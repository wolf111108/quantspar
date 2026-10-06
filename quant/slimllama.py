"""Slim-Llama output reuse and S-LUT scheduling, independent of bit sparsity ratios.

This is an analytical extension to FP8/INT4, not an RTL or native FP8 simulator.
Actual integer vectors are prototypes; residuals are never requantized.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from .bitlet import OPERATORS, _add_counts, _canonical, _operand_b, _storage_bits, validate_other_latency
from .bitwave import _bit_length, _precision, fixed_point_groups
from .cim_stats import encode_bits
from .quant_spec import parse_quant_spec

PAPER_URL = "https://doi.org/10.1109/ISSCC49661.2025.10904761"


@dataclass(frozen=True)
class SlimLlamaConfig:
    sbc_clusters: int = 8
    sbcs_per_cluster: int = 8
    columns_per_sbc: int = 8
    sluts_per_column: int = 8
    frequency_hz: float = 200e6
    sram_bytes: int = 500*1024
    dram_bytes_per_second: float = 1.6e9
    # Internal SRAM/NoC bandwidth is not published in the digest.
    sram_bytes_per_second: float | None = None
    output_storage_bytes: int = 2
    accumulator_storage_bytes: int = 4
    output_reuse: bool = True
    weight_clusters: int = 128
    clustering_features: int = 64
    clustering_chunk_vectors: int = 128
    sparse_threshold: float = .38
    lut_setup_cycles: int = 0
    mode_switch_cycles: int = 0
    group_setup_cycles: int = 0
    reuse_outputs_per_cycle: int = 512
    prefill_sample_waves: int = 64
    decode_sample_waves: int = 8
    chunk_waves: int = 2
    sample_seed: int = 23

    def __post_init__(self):
        for key in ("sbc_clusters", "sbcs_per_cluster", "columns_per_sbc", "sluts_per_column",
                    "sram_bytes", "output_storage_bytes", "accumulator_storage_bytes",
                    "weight_clusters", "clustering_features", "clustering_chunk_vectors",
                    "reuse_outputs_per_cycle", "chunk_waves"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Slim-Llama {key} must be a positive integer")
        for key in ("lut_setup_cycles", "mode_switch_cycles", "group_setup_cycles",
                    "prefill_sample_waves", "decode_sample_waves", "sample_seed"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Slim-Llama {key} must be a nonnegative integer")
        for key in ("frequency_hz", "dram_bytes_per_second", "sram_bytes_per_second"):
            value = getattr(self, key)
            if value is None and key == "sram_bytes_per_second":
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value <= 0):
                raise ValueError(f"Slim-Llama {key} must be finite and positive")
        if (isinstance(self.sparse_threshold, bool) or
                not isinstance(self.sparse_threshold, (int, float)) or
                not math.isfinite(self.sparse_threshold) or not 0 <= self.sparse_threshold <= 1):
            raise ValueError("Slim-Llama sparse_threshold must lie in [0,1]")
        if not isinstance(self.output_reuse, bool):
            raise ValueError("Slim-Llama output_reuse must be Boolean")

    @property
    def sbcs(self):
        return self.sbc_clusters*self.sbcs_per_cluster

    @property
    def columns(self):
        return self.sbcs*self.columns_per_sbc

    @classmethod
    def from_dict(cls, values):
        values = dict(values or {})
        for key in ("enabled", "trace_path"):
            values.pop(key, None)
        return cls(**values)


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big") % (2**63)


def prepare_weight_plan(matrix, spec, config, seed=None):
    """B=[K,N]. Deterministic feature-Hamming assignment to actual vectors.

    Centers are sampled without replacement, not synthetic centroids. Feature
    sampling affects grouping quality only: every computed delta uses full
    original code values, so the decomposition remains numerically exact.
    """
    spec = parse_quant_spec(spec)
    if spec.kind != "int" or not 2 <= spec.bits <= 16 or matrix.ndim != 2:
        raise ValueError("Slim-Llama static output reuse collector requires INT2..16 [K,N] codes")
    started = time.perf_counter()
    k, n = matrix.shape
    if min(k, n) <= 0:
        raise ValueError("Slim-Llama weight plan needs positive K/N")
    generator = torch.Generator(device="cpu").manual_seed(config.sample_seed if seed is None else seed)
    features = torch.randperm(k, generator=generator)[:min(k, config.clustering_features)]
    centers = torch.randperm(n, generator=generator)[:min(n, config.weight_clusters)]
    values = matrix[features.to(matrix.device)].t().detach().float().cpu()
    encode_bits(values, spec, "storage")
    values = values.to(torch.int32)
    prototypes = values[centers]
    assignments = torch.empty(n, dtype=torch.long)
    similarity = torch.empty(n, dtype=torch.float64)
    for start in range(0, n, config.clustering_chunk_vectors):
        stop = min(n, start+config.clustering_chunk_vectors)
        distances = (values[start:stop, None] != prototypes[None]).sum(-1)
        best = distances.argmin(-1)
        assignments[start:stop] = best
        similarity[start:stop] = 1-distances.gather(1, best[:, None]).flatten()/len(features)
    # A sampled prototype always supplies its own output, including tied vectors.
    assignments[centers] = torch.arange(len(centers))
    similarity[centers] = 1
    remaining = torch.ones(n, dtype=torch.bool)
    remaining[centers] = False
    sparse = remaining & (similarity >= config.sparse_threshold)
    dense = remaining & ~sparse
    probe_flat = torch.randint(k*n, (min(k*n, 64),), generator=generator)
    probe_k, probe_n = probe_flat//n, probe_flat % n
    probe_values = matrix[probe_k.to(matrix.device), probe_n.to(matrix.device)].detach().float().cpu()
    digest = hashlib.sha256()
    for value in (features, centers, assignments):
        digest.update(value.numpy().tobytes())
    metadata = dict(method="sampled actual prototypes; nearest feature-Hamming; lowest-ID ties",
        clusters=len(centers), feature_count=len(features), output_vectors=n, input_channels=k,
        mixed_residual_vectors=int(dense.sum()), buffer_residual_vectors=int(sparse.sum()),
        feature_residual_zero_fraction=float(similarity[remaining].mean()) if remaining.any() else None,
        feature_fraction_is_not_full_weight_sparsity=True,
        assignment_sha256=digest.hexdigest(), host_preprocessing_seconds=time.perf_counter()-started,
        host_time_excluded_from_hardware_latency=True,
        original_signed_bits=spec.bits, residual_signed_bits=spec.bits+1,
        residual_magnitude_bits_at_most=spec.bits,
        assumes_immutable_quantized_weights=True, mutation_probe_elements=len(probe_flat))
    return dict(centers=centers, assignments=assignments,
                mixed=torch.where(dense)[0], buffer=torch.where(sparse)[0],
                probe_k=probe_k, probe_n=probe_n, probe_values=probe_values,
                metadata=metadata, shape=(k, n), spec=spec.name())


def validate_weight_plan(plan, matrix, spec):
    if tuple(matrix.shape) != plan["shape"] or parse_quant_spec(spec).name() != plan["spec"]:
        raise ValueError("Slim-Llama cached weight shape/format changed; start a new manager")
    got = matrix[plan["probe_k"].to(matrix.device), plan["probe_n"].to(matrix.device)].detach().float().cpu()
    if not torch.equal(got, plan["probe_values"]):
        raise ValueError("Slim-Llama static weight probe changed; start a new manager")


def slut_group_work(a, b, a_spec, b_spec, valid_a, valid_b, config, mode):
    """A=[W,Mu,L,G], B=[W,Nu,L,G]; return synchronized column-wave work.

    Buffer mode has eight operands/two nonzero reads per cycle. Mixed mode
    reserves four registers for a three-operand LUT, four for raw operands:
    G=7. A non-dense prefix is explicitly switched to the full buffer.
    """
    group = 7 if mode == "mixed" else 8
    if mode not in ("mixed", "buffer") or a.shape[-1] != group or b.shape[-1] != group:
        raise ValueError("Slim-Llama mixed/buffer groups must contain 7/8 positions")
    am, aneg, _, _, _, _ = fixed_point_groups(a, a_spec, valid_a)
    bm, _, _, _, _, limit = fixed_point_groups(b, b_spec, valid_b)
    positive = torch.where(~aneg, am, 0).amax(-1)
    negative = torch.where(aneg, am, 0).amax(-1)
    signed_width = torch.maximum(_bit_length(positive), _bit_length((negative-1).clamp_min(0)))+1
    # A conservative exact extension: every wide digit is signed INT4 (±7).
    # -8 fits natively. Zero A does not bypass a nonzero B operation.
    passes = torch.where(signed_width <= 4, 1, (_bit_length(am.amax(-1))+2)//3)
    passes = torch.where(valid_a.any(-1), passes, 0)
    active_b = valid_b.any(-1)
    any_lut = torch.zeros_like(active_b)
    switches = torch.zeros_like(bm[..., 0])
    width = int(_bit_length(bm.amax(-1)).max())
    if width:
        planes = ((bm[..., None] >> torch.arange(width, device=b.device)) & 1).bool()
        nnz = planes.sum(-2)
        if mode == "mixed":
            lut = planes[..., :3, :].all(-2) & valid_b[..., :3].all(-1)[..., None]
            suffix = planes[..., 3:, :].sum(-2)
            plane_work = torch.where(lut, 1+(suffix+1)//2, (nnz+1)//2)
            switches = ((~lut) & (nnz != 0)).sum(-1)
            any_lut = lut.any(-1)
        else:
            lut = torch.zeros_like(nnz, dtype=torch.bool)
            plane_work = (nnz+1)//2
        plane_work += ((~lut) & (nnz != 0)).long()*config.mode_switch_cycles if mode == "mixed" else 0
        lut_reads = int(lut.sum())
        buffer_reads = int(torch.where(lut, (planes[..., 3:, :].sum(-2)+1)//2,
                                       (nnz+1)//2).sum())
    else:
        lut_reads = buffer_reads = 0
    # Each significance plane completes before the next; no free per-S-LUT
    # accumulation of differently shifted planes is assumed.
    mapped = ((passes[:, :, None, :, None]*plane_work[:, None]).amax(dim=(1, 2, 3)).sum(-1)
              if width else torch.zeros(a.shape[0], dtype=torch.long, device=a.device))
    setup = any_lut.long()*config.lut_setup_cycles+active_b.long()*config.group_setup_cycles
    mapped += (passes[:, :, None]*setup[:, None]).amax(dim=(1, 2, 3))
    # Dense reference keeps the same aligned A work and full magnitude planes.
    valid_count = valid_b.sum(-1)
    if mode == "mixed":
        full_prefix = valid_b[..., :3].all(-1)
        dense_group = torch.where(full_prefix, 1+(valid_count-3+1)//2, (valid_count+1)//2)
        dense_cost = dense_group*limit+full_prefix.long()*config.lut_setup_cycles
    else:
        dense_cost = (valid_count+1)//2*limit
    dense_cost += active_b.long()*config.group_setup_cycles
    dense = (passes[:, :, None]*dense_cost[:, None]).amax(dim=(1, 2, 3))
    observed = dict(observed_A_groups=int(valid_a.any(-1).sum()),
        observed_A_groups_exceeding_int4=int(((signed_width > 4) & valid_a.any(-1)).sum()),
        observed_B_coefficients=int(valid_b.sum()), observed_B_zero_coefficients=int(((bm == 0) & valid_b).sum()),
        observed_B_magnitude_width_histogram=torch.bincount(
            _bit_length(bm.amax(-1))[active_b], minlength=limit+1).cpu().tolist(),
        observed_A_pass_histogram=torch.bincount(passes[valid_a.any(-1)], minlength=1).cpu().tolist(),
        observed_lut_reads=lut_reads, observed_buffer_read_cycles=buffer_reads,
        observed_mode_switches=int(switches.sum()))
    # Each sampled wave reloads its extended tile; counts are local scenarios.
    local_a = (valid_a.sum(-1)*passes).sum(dim=(1, 2))  # one signed INT4 digit in a byte
    if parse_quant_spec(a_spec).kind == "fp":
        local_a += valid_a.any(-1).sum(dim=(1, 2))*2  # group exponent metadata
    magnitude_width = _bit_length(bm.amax(-1))
    local_b = (valid_b.sum(-1)*(magnitude_width+1)).sum(dim=(1, 2))/8
    if parse_quant_spec(b_spec).kind == "fp":
        local_b += active_b.sum(dim=(1, 2))*2  # full fixedpoint, sign and group metadata
    return dict(mapped=mapped, dense=dense, observed=observed,
                local_A_bytes=local_a, local_B_bytes=local_b)


def _stage(x, matrix, a_spec, b_spec, config, phase, seed, mode, vectors, plan=None, residual=False):
    operands, m, k = x.shape
    count = len(vectors)
    if not count:
        return None
    group = 7 if mode == "mixed" else 8
    mu = min(m, max(1, config.sbcs//math.ceil(count/config.columns_per_sbc)))
    nu = min(count, (config.sbcs//mu)*config.columns_per_sbc)
    span = group*config.sluts_per_column
    mt, nt, kr = math.ceil(m/mu), math.ceil(count/nu), math.ceil(k/span)
    total = mt*nt*kr
    budget = config.decode_sample_waves if phase == "decode" else config.prefill_sample_waves
    sampled = budget > 0 and total > budget
    observations = budget if sampled else total
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vectors = vectors.to(matrix.device)
    center_for_vector = (plan["centers"][plan["assignments"]].to(matrix.device) if residual else None)
    sums = dict(mapped=0., dense=0., local_A_bytes=0., local_B_bytes=0.)
    observed, variance = {}, 0.
    for operand in range(operands):
        source = _operand_b(matrix, operand)
        ids = torch.randint(total, (observations,), generator=generator) if sampled else None
        first = second = 0.
        operand_sums = {key: 0. for key in sums}
        for start in range(0, observations, config.chunk_waves):
            stop = min(observations, start+config.chunk_waves)
            wave = (ids[start:stop] if sampled else torch.arange(start, stop)).to(x.device)
            kg, nr, mr = wave % kr, (wave//kr) % nt, wave//(kr*nt)
            ki = kg[:, None, None]*span+torch.arange(span, device=x.device).reshape(config.sluts_per_column, group)
            mi = mr[:, None]*mu+torch.arange(mu, device=x.device)
            ni = nr[:, None]*nu+torch.arange(nu, device=x.device)
            va = (mi[:, :, None, None] < m) & (ki[:, None] < k)
            vb = (ni[:, :, None, None] < count) & (ki[:, None] < k)
            cols = vectors[ni.clamp_max(count-1)]
            a = x[operand, mi.clamp_max(m-1)[:, :, None, None], ki.clamp_max(k-1)[:, None]]
            w = source[ki.clamp_max(k-1)[:, None], cols[:, :, None, None]]
            spec = b_spec
            if residual:
                center_ids = center_for_vector[cols]
                # Float32 exactly represents INT1..16 and their differences.
                w = w.float()-source[ki.clamp_max(k-1)[:, None], center_ids[:, :, None, None]].float()
                spec = parse_quant_spec(b_spec).bits+1
            work = slut_group_work(a, w, a_spec, spec, va, vb, config, mode)
            moments = torch.stack([work[key].double().sum() for key in sums] +
                                  [work["mapped"].double().square().sum()]).cpu().tolist()
            for key, value in zip(operand_sums, moments):
                operand_sums[key] += value
            first += moments[0]
            second += moments[-1]
            _add_counts(observed, work["observed"])
        for key, value in operand_sums.items():
            sums[key] += value*(total/observations)
        if sampled and observations > 1:
            variance += total**2*max(0., (second-first**2/observations)/(observations-1))/observations
    # Capacity includes worst-case extended operands plus accumulator tiles.
    a_limit, _ = _precision(a_spec)
    b_limit = _precision(b_spec)[0]+1  # sign-magnitude; INT4 delta needs at most 4+sign
    input_tile = mu*span*max(1, math.ceil(a_limit/3))
    weight_tile = math.ceil(nu*span*b_limit/8)+nu*config.accumulator_storage_bytes
    if parse_quant_spec(a_spec).kind == "fp":
        input_tile += mu*config.sluts_per_column*2
    if parse_quant_spec(b_spec).kind == "fp":
        weight_tile += nu*config.sluts_per_column*2
    error = math.sqrt(variance) if not sampled or observations > 1 else None
    return dict(mode=mode, residual=residual, vectors=count,
        cycles=sums["mapped"], dense_reference_cycles=sums["dense"], observed=observed,
        local_A_bytes=sums["local_A_bytes"], local_B_bytes=sums["local_B_bytes"],
        original_A_read_bytes_streaming=math.ceil(operands*m*k*nt*_storage_bits(a_spec)/8),
        layout=dict(m_tile=mu, n_tile=nu, group_size=group, k_span=span,
                    m_tiles=mt, n_tiles=nt, k_tiles=kr, active_sbc_slots=mu*math.ceil(nu/config.columns_per_sbc)),
        minimum_tile_bytes=input_tile+weight_tile,
        sampling=dict(method="uniform_with_replacement" if sampled else "exact", exact=not sampled,
                      seed=seed, sample_waves=operands*observations, total_waves=operands*total,
                      mapped_cycles_standard_error=error,
                      mapped_cycles_approximate_95pct_interval=(
                          [max(0., sums["mapped"]-1.96*error), sums["mapped"]+1.96*error] if error is not None else None)))


def measure_slimllama_mapping(activation, weight, a_spec, b_spec, k, n, config,
                              phase="prefill", seed=None, weight_plan=None):
    x, b = _canonical(activation, weight, k, n)
    seed = config.sample_seed if seed is None else seed
    static = weight.ndim == 2 and parse_quant_spec(b_spec).kind == "int"
    plan = weight_plan
    if static and config.output_reuse and plan is None:
        plan = prepare_weight_plan(_operand_b(b, 0), b_spec, config, _seed(config.sample_seed, k, n))
    if plan is not None:
        validate_weight_plan(plan, _operand_b(b, 0), b_spec)
    stages = {}
    if plan is not None and config.output_reuse:
        definitions = (("centers", "mixed", plan["centers"], False),
                       ("mixed_residuals", "mixed", plan["mixed"], True),
                       ("buffer_residuals", "buffer", plan["buffer"], True))
    else:
        definitions = (("direct", "mixed" if static else "buffer", torch.arange(n), False),)
    for name, mode, vectors, residual in definitions:
        stage = _stage(x, b, a_spec, b_spec, config, phase, _seed(seed, name), mode, vectors, plan, residual)
        if stage is not None:
            stages[name] = stage
    # Independent fixed-width, no-output-reuse reference; never borrow Bitlet/BW ratios.
    reference = _stage(x, b, a_spec, b_spec, config, phase, _seed(seed, "reference"),
                       "mixed" if static else "buffer", torch.arange(n))
    operands, m, _ = x.shape
    centers = len(plan["centers"]) if plan is not None and config.output_reuse else 0
    remaining = n-centers if centers else 0
    center_store = operands*math.ceil(m*centers/config.reuse_outputs_per_cycle) if centers else 0
    reuse = operands*math.ceil(m*remaining/config.reuse_outputs_per_cycle) if remaining else 0
    mapped = sum(s["cycles"] for s in stages.values())+center_store+reuse
    observed = {}
    for stage in stages.values():
        _add_counts(observed, stage["observed"])
    tile_bytes = max(s["minimum_tile_bytes"] for s in stages.values())
    columns = max(s["layout"]["n_tile"] for s in stages.values())
    per_row = (centers+columns)*config.accumulator_storage_bytes
    available = config.sram_bytes-tile_bytes
    if available < per_row:
        raise ValueError("Slim-Llama SRAM cannot fit one extended tile, center outputs and partial sums")
    rows_per_window = min(m, available//per_row)
    windows = math.ceil(m/rows_per_window)
    local = dict(activation_read_bytes=sum(s["local_A_bytes"] for s in stages.values()),
                 B_read_bytes=sum(s["local_B_bytes"] for s in stages.values()),
                 center_output_write_bytes=operands*m*centers*config.accumulator_storage_bytes,
                 center_output_read_bytes=operands*m*remaining*config.accumulator_storage_bytes,
                 output_write_bytes=operands*m*n*config.output_storage_bytes)
    return dict(operand_shape=list(x.shape), weight_shape=list(weight.shape),
        layout=dict(m=m, k=k, n=n, operands=operands, sbcs=config.sbcs, columns=config.columns,
                    sluts=config.columns*config.sluts_per_column), stages=stages,
        dense_reference=reference,
        cycles=dict(mapped_tiles=mapped, dense_tiles=reference["dense_reference_cycles"],
                    center_store=center_store, output_reuse=reuse,
                    center_compute=stages.get("centers", {}).get("cycles", 0.),
                    mixed_residual_compute=stages.get("mixed_residuals", {}).get("cycles", 0.),
                    buffer_residual_compute=stages.get("buffer_residuals", {}).get("cycles", 0.),
                    direct_compute=stages.get("direct", {}).get("cycles", 0.)),
        compute_seconds=dict(mapped_tiles=mapped/config.frequency_hz,
                             dense_tiles=reference["dense_reference_cycles"]/config.frequency_hz),
        observed=observed, local_traffic=local,
        weight_reuse=dict(enabled=bool(centers), centers=centers, remaining_vectors=remaining,
                          preprocessing=plan["metadata"] if centers else None),
        memory=dict(minimum_tile_bytes=tile_bytes, accumulator_bytes_per_window_row=per_row,
                    rows_per_weight_window=rows_per_window, weight_windows=windows,
                    center_outputs_spilled=False, policy="tile weights; window rows and center/partial outputs"),
        sampling=dict(exact=all(s["sampling"]["exact"] for s in [*stages.values(), reference]),
                      note="stage-stratified wave samples; reference sampled separately; observations are not full tensors"),
        dense_equivalent_operations=2*operands*m*k*n,
        mapped_compute_speedup=reference["dense_reference_cycles"]/mapped if mapped else None,
        alignment="lossless group fixedpoint; native signed INT4 or signed 3-magnitude-bit slices")


def traffic_and_latency(result, activation, weight, a_spec, b_spec, name, context, config):
    a_min = math.ceil(activation.numel()*_storage_bits(a_spec)/8)
    b_min = math.ceil(weight.numel()*_storage_bits(b_spec)/8)
    attention = name in ("qk_matmul", "pv_matmul")
    append = 0
    if attention:
        for key in ("kv_heads", "head_dim", "cache_length_before", "query_length"):
            if key not in context:
                raise ValueError("Slim-Llama attention traffic requires cache/head provenance")
        cells = activation.shape[0]*context["kv_heads"]*context["head_dim"]
        b_min = math.ceil(cells*context["cache_length_before"]*_storage_bits(b_spec)/8)
        append = math.ceil(cells*context["query_length"]*_storage_bits(b_spec)/8)
    layout, reuse, local = result["layout"], result["weight_reuse"], result["local_traffic"]
    centers, remaining = reuse["centers"], reuse["remaining_vectors"]
    # Keep original W4 transport. No free off-chip residual compression: the
    # conditional schedule rereads centers to reconstruct exact differences.
    extra_centers = math.ceil(layout["operands"]*centers*layout["k"]*_storage_bits(b_spec)/8) if remaining else 0
    cluster_ids = math.ceil(layout["operands"]*remaining*math.ceil(math.log2(centers))/8) if remaining else 0
    windows = result["memory"]["weight_windows"]
    a_stream = sum(s["original_A_read_bytes_streaming"] for s in result["stages"].values())
    b_window = (b_min+extra_centers+cluster_ids)*windows
    output = local["output_write_bytes"]
    minimum_bytes = a_min+b_min+output+append
    window_bytes = a_stream+b_window+output+append
    traffic = dict(activation_read_bytes_minimum=a_min, B_read_bytes_minimum=b_min,
        packed_weight_read_bytes_minimum=0 if attention else b_min,
        fp8_kv_read_bytes_minimum=b_min if attention else 0, fp8_kv_append_bytes=append,
        output_write_bytes=output, center_coefficients_extra_bytes_per_window=extra_centers,
        cluster_id_bytes_per_window=cluster_ids, activation_read_bytes_window_schedule=a_stream,
        B_read_bytes_window_schedule=b_window, dram_bytes_minimum=minimum_bytes,
        dram_bytes_window_schedule=window_bytes, **{f"local_{key}": value for key, value in local.items()})
    compute = result["compute_seconds"]["mapped_tiles"]
    minimum = minimum_bytes/config.dram_bytes_per_second
    window = window_bytes/config.dram_bytes_per_second
    local_bytes = sum(local.values())
    local_seconds = local_bytes/config.sram_bytes_per_second if config.sram_bytes_per_second is not None else 0.
    latency = dict(mapped_compute_seconds=compute, minimum_dram_seconds=minimum,
        window_dram_seconds=window, modeled_local_sram_seconds=local_seconds,
        minimum_traffic_full_overlap_seconds=max(compute, minimum, local_seconds),
        capacity_window_no_overlap_seconds=compute+window+local_seconds)
    return traffic, latency


class SlimLlamaStats:
    def __init__(self, config, trace_path=None):
        self.config = config
        self.phases, self.layers, self.steps, self.weight_plans = {}, {}, {}, {}
        self.trace_path = Path(trace_path) if trace_path else None
        self.trace = None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace = self.trace_path.open("x", encoding="utf-8")

    def close(self):
        if self.trace:
            self.trace.close()
            self.trace = None

    def validate_coverage(self, layers, decode_steps=0):
        expected = {f"prefill:{name}_{index}": 1 for index in range(layers) for name in OPERATORS}
        if decode_steps:
            expected.update({f"decode:{name}_{index}": decode_steps for index in range(layers) for name in OPERATORS})
        if {key: value["calls"] for key, value in self.layers.items()} != expected:
            raise ValueError("Incomplete Slim-Llama per-layer/operator/phase coverage")

    def collect(self, name, index, activation, weight, a_spec, b_spec, k, n, phase, context):
        if name not in OPERATORS:
            raise ValueError(f"Unsupported Slim-Llama operator: {name}")
        meta = dict(context)
        attention = name in ("qk_matmul", "pv_matmul")
        if attention and (activation.ndim != 4 or weight.ndim != 4):
            raise ValueError("Slim-Llama attention requires [B,H,M,K] operands")
        x = activation if attention else activation.reshape(-1, k)
        physical_weight = weight
        if attention and meta.get("shared_kv_gqa"):
            heads, kv = meta["q_heads"], meta["kv_heads"]
            if activation.shape[1] != heads or heads % kv:
                raise ValueError("Slim-Llama GQA metadata disagrees with attention operands")
            group = heads//kv
            batch, _, m, _ = activation.shape
            x = activation.reshape(batch, kv, group*m, k)
            physical_weight = weight[:, ::group]
        plan = None
        if not attention and self.config.output_reuse:
            key = f"{name}_{index}"
            if key not in self.weight_plans:
                _, matrix = _canonical(x, physical_weight, k, n)
                self.weight_plans[key] = prepare_weight_plan(_operand_b(matrix, 0), b_spec, self.config,
                                                            _seed(self.config.sample_seed, name, index, k, n))
            plan = self.weight_plans[key]
        seed = _seed(self.config.sample_seed, phase, name, index, meta.get("step", 0))
        from .mapping import Mapping_stat_slimllama
        result = Mapping_stat_slimllama(x, physical_weight, a_spec, b_spec, k, n,
                                       self.config, phase, seed, plan)
        traffic, latency = traffic_and_latency(result, x, physical_weight, a_spec, b_spec, name, meta, self.config)
        counts = dict(calls=1, exact_calls=int(result["sampling"]["exact"]),
            sampled_calls=int(not result["sampling"]["exact"]),
            output_reuse_calls=int(result["weight_reuse"]["enabled"]), cycles=result["cycles"],
            observed=result["observed"], dense_equivalent_operations=result["dense_equivalent_operations"],
            traffic=traffic, latency=latency,
            stages={key: dict(calls=1, estimated_compute_cycles=stage["cycles"], observed=stage["observed"])
                    for key, stage in result["stages"].items()})
        _add_counts(self.phases.setdefault(phase, {}), counts)
        _add_counts(self.layers.setdefault(f"{phase}:{name}_{index}", {}), counts)
        step = self.steps.setdefault((phase, meta.get("step", 0)), dict(phase=phase, context=meta, counts={}))
        _add_counts(step["counts"], counts)
        # Host preprocessing duration appears once in summary, never each call's mapping.
        if result["weight_reuse"]["preprocessing"] is not None:
            result["weight_reuse"]["preprocessing"] = {
                key: value for key, value in plan["metadata"].items() if key != "host_preprocessing_seconds"}
        record = dict(result, operator=name, layer=index, phase=phase, context=meta,
                      activation_format=parse_quant_spec(a_spec).name(), B_format=parse_quant_spec(b_spec).name(),
                      traffic=traffic, latency=latency)
        if self.trace:
            self.trace.write(json.dumps(record, allow_nan=False)+"\n")
        return record

    def export(self, path, workload=None, other_latency=None):
        if self.trace:
            self.trace.flush()
        workload = workload or {}
        completed = workload.get("status") == "complete"
        scenarios = ("minimum_traffic_full_overlap_seconds", "capacity_window_no_overlap_seconds")
        phases = {phase: {key: value["latency"][key] for key in scenarios} for phase, value in self.phases.items()}
        total = {key: sum(value[key] for value in phases.values()) for key in scenarios}
        rest = None
        if other_latency is not None:
            validate_other_latency(other_latency, workload.get("decode_steps"))
            rest = dict(prefill_seconds=other_latency["prefill_seconds"],
                        decode_seconds=sum(other_latency["decode_step_seconds"][:workload.get("completed_decode_steps", 0)]))
        e2e = ({key: value+rest["prefill_seconds"]+rest["decode_seconds"] for key, value in total.items()}
               if completed and rest is not None else None)
        doc = dict(schema_version=1, backend="slimllama", config=asdict(self.config), sources=dict(isscc2025=PAPER_URL),
            paper_parameters=dict(sbc_clusters=8, sbcs_per_cluster=8, columns_per_sbc=8, sluts_per_column=8,
                slut_registers=8, slut_register_bits=7, slut_read_ports=2,
                frequency_range_hz=[25e6, 200e6], selected_frequency_hz=200e6,
                sram_reported="500 KB", sram_bytes_interpretation=500*1024,
                dram_bytes_per_second_at_200mhz=1.6e9, native_activation_bits=[4, 8, 16],
                native_weight_bits="INT1..16 or ternary", benchmark_weight_clusters=128,
                sram_bytes_per_second=None),
            workload=workload, phases=self.phases, layers=self.layers, steps=list(self.steps.values()),
            weight_preprocessing={key: plan["metadata"] for key, plan in self.weight_plans.items()},
            mapped_compute_speedups={phase: value["cycles"]["dense_tiles"]/value["cycles"]["mapped_tiles"]
                if value["cycles"]["mapped_tiles"] else None for phase, value in self.phases.items()},
            trace_file=self.trace_path.name if self.trace_path else None,
            latency=dict(scope="conditional S-LUT integer-slice and original-code transport extension",
                collection_complete=completed, per_phase_seconds=phases, gemm_and_io_seconds=total,
                supplied_other_latency=rest, conditional_e2e_seconds=e2e,
                internal_sram_bandwidth_supplied=self.config.sram_bytes_per_second is not None,
                missing_costs=([] if rest is not None else ["LM head", "Softmax/RMSNorm/RoPE/SiLU/residual"]) +
                    ["FP8 conversion/group scale and slice aggregation", "runtime residual formation", "NoC/control stalls"] +
                    ([] if self.config.sram_bytes_per_second is not None else ["internal SRAM bandwidth"]) +
                    (["LUT setup"] if self.config.lut_setup_cycles == 0 else [])),
            assumptions=[
                "Weight INT4 is within Fig.23.9.7 INT1..16; FP8 is not a native published activation format.",
                "Original quantized model operands and forward results are never changed by this collector.",
                "Output reuse is exact W=center+delta on integer codes with a shared scalar weight scale.",
                "The digest does not publish the clustering algorithm; sampled actual prototypes and feature-Hamming are an explicit alternative.",
                "Feature sparsity selects residual modes; observed full-code deltas, not the paper benchmark sparsity, determine sampled work.",
                "Static quantized weights are immutable within a run; a small probe is diagnostic, not a full mutation detector.",
                "INT4 differences use signed INT5 without clipping; signs and bit shifts preserve their exact numerical values.",
                "FP8 is losslessly aligned within each S-LUT group; wide A uses signed INT4 digits of at most three magnitude bits.",
                "Wide B is processed as signed-magnitude binary planes; sign/shift and exponent accumulation need an extension.",
                "Each magnitude plane has a synchronized wave barrier before the next plane; LUT rebuilding after mode switches is part of configurable control overhead.",
                "Buffer S-LUT stores eight A operands and consumes two nonzero weight bits per cycle.",
                "Mixed S-LUT uses four LUT entries for three dense operands and four raw buffer operands (seven K positions).",
                "A non-dense three-element prefix switches to a full buffer; mode switch and LUT setup costs are configurable assumptions.",
                "SBCs are partitioned over M and eight-column N tiles; their eight S-LUTs run in parallel with synchronized wave barriers.",
                "Centers complete before residual vectors for each row window; output stores/reads/additions use an assumed configurable vector throughput.",
                "Dynamic attention uses Buffer mode and no offline weight/output reuse; GQA transport counts physical KV heads.",
                "Index transition reordering is not modeled as a free latency improvement; no energy estimate is produced.",
                "No A-zero shortcut is assumed; zero B planes are skipped.",
                "The dense reference is full fixed-width magnitude-plane work without output reuse; it is not a cross-architecture speedup.",
                "SRAM capacity covers extended tiles and a window of center/partial outputs; bank placement, double buffering and register spills are not simulated.",
                "Minimum transport/full overlap is optimistic; capacity-window/no overlap rereads original W4 plus centers/IDs and streams FP8 A.",
                "Residuals are host-preprocessed for analysis, not an implemented off-chip compressed representation or device kernel.",
                "Internal SRAM/NoC bandwidth is unpublished; absent bandwidth contributes no modeled SRAM stall, not a measured zero cost.",
                "Default 200MHz/1.6GB/s corresponds to the same paper operating point; 4.69mW belongs to 25MHz and is not used here.",
                "Output two bytes, accumulator four bytes, setup zeros, sign/slice logic and control are modeling assumptions.",
                "Remaining operators are supplied separately; estimates are conditional scenarios, not hardware E2E measurements or physical bounds.",
            ])
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(doc, indent=2, allow_nan=False)+"\n")
        return doc
