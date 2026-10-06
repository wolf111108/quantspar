"""Bitlet bit-column work and conditional GEMM/IO latency estimates.

Paper facts and modeling assumptions are kept separate. FP8/INT4 codes are
promoted to the paper's FP32 BCE representation for statistics only; this is
an analytical mixed-format extension, not a measured native FP8 datapath.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from .cim_stats import encode_bits
from .quant_spec import parse_quant_spec


PAPER_URL = "https://luhang-hpu.github.io/files/bitlet-MICRO21.pdf"
EXTENSION_URL = "https://luhang-hpu.github.io/files/bitlet-TCAD24.pdf"
OPERATORS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
             "up_proj", "down_proj", "qk_matmul", "pv_matmul")


@dataclass(frozen=True)
class BitletConfig:
    pe_count: int = 32
    group_size: int = 64
    frequency_hz: float = 1.0e9
    mantissa_bits: int = 24
    activation_dma_bytes_per_second: float = 12.8e9
    weight_dma_bytes_per_second: float = 12.8e9
    local_buffer_bytes_per_second: float = 25.6e9
    # Neither cited paper specifies a local-buffer capacity.
    local_buffer_bytes: int | None = None
    output_storage_bytes: int = 2
    group_setup_cycles: int = 0
    prefill_sample_waves: int = 64
    decode_sample_waves: int = 8
    chunk_waves: int = 8
    sample_seed: int = 23

    def __post_init__(self):
        for key in ("pe_count", "group_size", "mantissa_bits", "chunk_waves",
                    "output_storage_bytes"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Bitlet {key} must be a positive integer")
        if self.mantissa_bits != 24:
            raise ValueError("This Bitlet adapter models the paper's 24-lane BCE")
        for key in ("group_setup_cycles", "prefill_sample_waves",
                    "decode_sample_waves", "sample_seed"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Bitlet {key} must be a nonnegative integer")
        for key in ("frequency_hz", "activation_dma_bytes_per_second",
                    "weight_dma_bytes_per_second", "local_buffer_bytes_per_second"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Bitlet {key} must be finite and positive")
        if self.local_buffer_bytes is not None:
            value = self.local_buffer_bytes
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("local_buffer_bytes must be positive or null (unreported)")

    @classmethod
    def from_dict(cls, values):
        values = dict(values or {})
        for key in ("enabled", "trace_path"):
            values.pop(key, None)
        return cls(**values)


def _storage_bits(spec):
    spec = parse_quant_spec(spec)
    if spec.kind == "int":
        if spec.bits > 24:
            raise ValueError("The paper supports fixed-point widths up to 24 bits")
        return spec.bits
    if spec.fmt in ("e4m3", "e5m2"):
        return 8
    raise ValueError("Bitlet collection supports quantized INT and FP8 codes")


def _components(values, spec, lanes):
    """Normalized FP32 significand, exponent, raw storage code and sign."""
    raw, width = encode_bits(values, spec, "storage")
    _storage_bits(spec)
    magnitude = values.detach().float().abs()
    fraction, exponent = torch.frexp(magnitude)
    # frexp returns [0.5,1); the implicit one is included in this integer.
    significand = (fraction * (1 << lanes)).long()
    # IEEE FP32 zero has exponent field zero and an all-zero significand.
    # Keep its effective exponent (-126): a zero A must not invent an
    # undocumented A-value bypass in a B-bit-interleaved BCE.
    exponent = torch.where(magnitude == 0, -126, exponent.long()-1)
    return significand, exponent, raw, width, values.signbit()


def group_column_work(a, b, a_spec, b_spec, config, valid=None):
    """One BCE per [...,G] group; zero padding never counts as real work.

    Floating/mixed mode follows Fig.5: Ei=EAi+EWi, Emax=max(Ei), shift
    the W significand right in a 24-lane BCE, then count each bit column.
    Integer/integer mode bypasses exponent matching and interleaves magnitude
    bits; product sign controls addition/subtraction, not a sparse bit lane.
    """
    if a.shape != b.shape or a.shape[-1] != config.group_size:
        raise ValueError("Bitlet pairs must have equal shape [...,group_size]")
    a_spec, b_spec = parse_quant_spec(a_spec), parse_quant_spec(b_spec)
    valid = torch.ones_like(a, dtype=torch.bool) if valid is None else valid
    if valid.shape != a.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a Boolean mask with the pair shape")
    am, ae, ar, aw, aneg = _components(a, a_spec, config.mantissa_bits)
    bm, be, br, bw, bneg = _components(b, b_spec, config.mantissa_bits)
    active = valid & (a != 0) & (b != 0)
    if a_spec.kind == b_spec.kind == "int":
        lanes = b_spec.bits
        if lanes > config.mantissa_bits:
            raise ValueError("Integer B width exceeds the configured BCE lanes")
        aligned = torch.where(valid, b.abs().long(), 0)
        dropped = torch.zeros_like(active)
        span = torch.zeros_like(active[..., 0], dtype=torch.long)
    else:
        lanes = config.mantissa_bits
        pair_exponent = ae + be
        high = torch.where(valid, pair_exponent, -1000).amax(-1, keepdim=True)
        low = torch.where(active, pair_exponent, 1000).amin(-1, keepdim=True)
        span = torch.where(active.any(-1), (high-low).squeeze(-1), 0)
        shifts = (high - pair_exponent).clamp(0, 63)
        aligned = torch.where(valid, bm >> shifts, 0)
        discarded_mask = (1 << shifts.clamp_max(lanes)) - 1
        dropped = active & ((bm & discarded_mask) != 0)
    # The bit axis belongs only to a bounded tile chunk, never the full
    # attention tensor. Vectorization avoids dozens of CUDA synchronizations.
    bit_positions = torch.arange(lanes, device=aligned.device)
    populations = ((aligned.unsqueeze(-1) >> bit_positions) & 1).sum(-2)
    cycles = populations.amax(-1)
    real_groups = valid.any(-1)
    setup = real_groups.long() * config.group_setup_cycles
    cycles += setup
    dense = valid.sum(-1) + setup
    column_population = populations[real_groups]
    a_bits = ((ar.unsqueeze(-1) >> torch.arange(aw, device=ar.device)) & 1)
    b_bits = ((br.unsqueeze(-1) >> torch.arange(bw, device=br.device)) & 1)
    scalars = torch.stack([
        real_groups.sum(), valid.sum(), (valid & ~active).sum(),
        (active & (aneg ^ bneg)).sum(), dropped.sum(), dropped.any(-1).sum(),
        span[real_groups].sum(), (a_bits*valid.unsqueeze(-1)).sum(),
        (b_bits*valid.unsqueeze(-1)).sum(),
    ]).cpu().tolist()
    counts = dict(
        observed_groups=scalars[0], observed_elements=scalars[1],
        observed_zero_products=scalars[2], observed_negative_products=scalars[3],
        observed_truncated_products=scalars[4], observed_truncated_groups=scalars[5],
        observed_alignment_span_sum=scalars[6],
        observed_source_A_one_bits=scalars[7], observed_source_B_one_bits=scalars[8],
        observed_source_A_bits=scalars[1]*aw, observed_source_B_bits=scalars[1]*bw,
        column_population_histogram=torch.bincount(
            column_population.reshape(-1), minlength=config.group_size+1).cpu().tolist(),
        group_cycles_histogram=torch.bincount(
            cycles[real_groups], minlength=config.group_size+config.group_setup_cycles+1
        ).cpu().tolist(),
    )
    return cycles, dense, counts


def _add_counts(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            _add_counts(target.setdefault(key, {}), value)
        elif isinstance(value, list):
            dest = target.setdefault(key, [0] * len(value))
            if len(dest) < len(value):
                dest.extend([0] * (len(value)-len(dest)))
            for index, count in enumerate(value):
                dest[index] += count
        else:
            target[key] = target.get(key, 0) + value


def _canonical(activation, weight, k, n):
    if min(k, n) <= 0:
        raise ValueError("K and N must be positive")
    if activation.ndim == 2 and weight.ndim == 2:
        if tuple(weight.shape) != (n, k):
            raise ValueError("Linear weights must be [N,K]")
        x = activation.unsqueeze(0)
        b = weight.transpose(-2, -1).unsqueeze(0)
    elif activation.ndim == weight.ndim and activation.ndim in (3, 4):
        if activation.shape[:-2] != weight.shape[:-2] or weight.shape[-2:] != (k, n):
            raise ValueError("Attention operands must be [...,M,K] and [...,K,N]")
        # Reshape only A, which is small in decode. Do not flatten a transposed
        # full KV tensor, as doing so can materialize another cache-sized copy.
        x = activation.reshape(-1, activation.shape[-2], k)
        b = weight
    else:
        raise ValueError("Unsupported Bitlet operand shapes")
    if x.shape[-1] != k or x.shape[-2] <= 0:
        raise ValueError("Activation shape disagrees with M/K")
    return x, b


def _operand_b(weight, index):
    if weight.ndim == 3:
        return weight[index]
    heads = weight.shape[1]
    return weight[index // heads, index % heads]


def measure_bitlet_mapping(activation, weight, a_spec, b_spec, k, n,
                           config, phase="prefill", seed=None):
    """Stream exact or uniformly sampled synchronized PE tiles.

    A wave broadcasts one G-entry A slice to up to PE_count B columns.
    Each independent batch/KV-head operand is scheduled separately. Sampling
    uses replacement with a stable seed; estimates and standard errors apply
    to tile work, not to complete hardware timing.
    """
    x, b = _canonical(activation, weight, k, n)
    operands, m, _ = x.shape
    pe, group = config.pe_count, config.group_size
    nt, kr = math.ceil(n/pe), math.ceil(k/group)
    waves_per_operand = m * nt * kr
    budget = config.decode_sample_waves if phase == "decode" else config.prefill_sample_waves
    sampled = budget > 0 and waves_per_operand > budget
    observations = min(budget, waves_per_operand) if sampled else waves_per_operand
    totals = dict(mapped_tiles=0.0, ideal_pe_balance=0.0,
                  dense_tiles=float(operands*m*nt*(k+kr*config.group_setup_cycles)))
    variance = 0.0
    counts, observed_waves = {}, 0
    unclipped_mapped_cycles = 0.0
    generator = torch.Generator(device="cpu").manual_seed(
        config.sample_seed if seed is None else seed)
    for operand in range(operands):
        sampled_ids = (torch.randint(waves_per_operand, (observations,), generator=generator)
                       if sampled else None)
        saving_sum = saving_square_sum = pe_saving_sum = 0
        matrix = _operand_b(b, operand)
        for start in range(0, observations, config.chunk_waves):
            stop = min(observations, start+config.chunk_waves)
            ids = (sampled_ids[start:stop] if sampled else torch.arange(start, stop)).to(x.device)
            kg = ids % kr
            tile = (ids // kr) % nt
            row = ids // (kr*nt)
            ki = kg[:, None] * group + torch.arange(group, device=x.device)
            ni = tile[:, None] * pe + torch.arange(pe, device=x.device)
            valid = (ki[:, None, :] < k) & (ni[:, :, None] < n)
            a = x[operand, row[:, None], ki.clamp_max(k-1)][:, None, :].expand(-1, pe, -1)
            w = matrix[ki.clamp_max(k-1)[:, None, :], ni.clamp_max(n-1)[:, :, None]]
            costs, dense, chunk_counts = group_column_work(a, w, a_spec, b_spec, config, valid)
            # Dense work is known exactly, including K/N tails. Estimate its
            # nonnegative savings to avoid inventing work by oversampling a
            # full K group. This is a control-variate estimator.
            saving = dense.amax(-1) - costs.amax(-1)
            moments = torch.stack([saving.sum(), saving.square().sum(),
                                   (dense-costs).sum()]).cpu().tolist()
            saving_sum += moments[0]
            saving_square_sum += moments[1]
            pe_saving_sum += moments[2]
            _add_counts(counts, chunk_counts)
        scale = waves_per_operand / observations
        operand_dense = m*nt*(k+kr*config.group_setup_cycles)
        raw_mapped = operand_dense - saving_sum*scale
        mapped = max(0.0, raw_mapped)
        unclipped_mapped_cycles += raw_mapped
        totals["mapped_tiles"] += mapped
        dense_balance = m*n*(k+kr*config.group_setup_cycles)/pe
        totals["ideal_pe_balance"] += max(0.0, min(mapped,
            dense_balance-pe_saving_sum*scale/pe))
        if sampled and observations > 1:
            sample_variance = max(0, (saving_square_sum-saving_sum**2/observations) /
                                  (observations-1))
            variance += waves_per_operand**2 * sample_variance / observations
        observed_waves += observations
    standard_error = math.sqrt(variance) if not sampled or observations > 1 else None
    estimate = totals["mapped_tiles"]
    interval = (None if standard_error is None else
                [max(0, min(totals["dense_tiles"], unclipped_mapped_cycles-1.96*standard_error)),
                 max(0, min(totals["dense_tiles"], unclipped_mapped_cycles+1.96*standard_error))])
    a_bits, b_bits = _storage_bits(a_spec), _storage_bits(b_spec)
    tile_bytes = (math.ceil(min(k, group)*a_bits/8) +
                  math.ceil(min(n, pe)*min(k, group)*b_bits/8) +
                  min(n, pe)*config.output_storage_bytes)
    if config.local_buffer_bytes is not None and config.local_buffer_bytes < tile_bytes:
        raise ValueError("Configured buffer cannot hold one packed Bitlet tile")
    return dict(
        operand_shape=list(x.shape), weight_shape=list(weight.shape),
        layout=dict(operands=operands, m=m, k=k, n=n, output_tiles=nt,
                    k_groups=kr, pe_count=pe, group_size=group,
                    scheduling="independent operands; synchronized output-column tiles"),
        sampling=dict(method="uniform_with_replacement" if sampled else "exact",
                      estimator="known_dense_minus_mean_savings_clipped_nonnegative" if sampled else "exact",
                      seed=config.sample_seed if seed is None else seed,
                      sample_waves=observed_waves,
                      total_waves=operands*waves_per_operand, exact=not sampled,
                      mapped_cycles_standard_error=standard_error,
                      mapped_cycles_before_nonnegative_clipping=unclipped_mapped_cycles,
                      mapped_cycles_approximate_95pct_interval=interval),
        cycles=totals, observed=counts,
        mapped_compute_speedup=totals["dense_tiles"]/estimate if estimate > 0 else None,
        compute_seconds={key: value/config.frequency_hz for key, value in totals.items()},
        dense_equivalent_operations=2*operands*m*k*n,
        minimum_packed_tile_bytes=tile_bytes,
        alignment="fixed_integer_magnitude" if
            parse_quant_spec(a_spec).kind == parse_quant_spec(b_spec).kind == "int"
            else "normalized_codes_promoted_to_fp32_BCE",
    )


def traffic_and_latency(result, activation, weight, a_spec, b_spec, name, context, config):
    """Traffic scenarios, not a capacity-validated memory simulation."""
    layout = result["layout"]
    operands, m, k, n = (layout[key] for key in ("operands", "m", "k", "n"))
    a_bits, b_bits = _storage_bits(a_spec), _storage_bits(b_spec)
    a_read = math.ceil(activation.numel()*a_bits/8)
    b_read = math.ceil(weight.numel()*b_bits/8)
    append = 0
    if name in ("qk_matmul", "pv_matmul"):
        required = ("kv_heads", "head_dim", "cache_length_before", "query_length")
        if not all(key in context for key in required):
            raise ValueError("Bitlet attention traffic requires cache/head provenance")
        batch = activation.shape[0]
        before, query = context["cache_length_before"], context["query_length"]
        cache_cells = batch * context["kv_heads"] * context["head_dim"]
        # The new K/V slice is already on chip; each QK/PV accounts one of K/V.
        b_read = math.ceil(cache_cells * before * b_bits/8)
        append = math.ceil(cache_cells * query * b_bits/8)
    output = operands*m*n*config.output_storage_bytes
    local_a = math.ceil(operands*m*k*layout["output_tiles"]*a_bits/8)
    local_b = math.ceil(weight.numel()*b_bits/8)
    traffic = dict(
        activation_read_bytes_minimum=a_read, B_read_bytes_minimum=b_read,
        output_write_bytes=output, fp8_kv_append_bytes=append,
        local_activation_broadcast_bytes=local_a,
        local_B_read_bytes_resident=local_b,
        local_B_read_bytes_streaming=local_b*m,
        packed_weight_read_bytes_minimum=b_read if name not in ("qk_matmul", "pv_matmul") else 0,
        fp8_kv_read_bytes_minimum=b_read if name in ("qk_matmul", "pv_matmul") else 0,
    )
    a_dma = (a_read+output+append)/config.activation_dma_bytes_per_second
    b_dma = b_read/config.weight_dma_bytes_per_second
    local_resident = (local_a+local_b+output)/config.local_buffer_bytes_per_second
    local_streaming = (local_a+local_b*m+output)/config.local_buffer_bytes_per_second
    compute = result["compute_seconds"]["mapped_tiles"]
    latency = dict(
        mapped_compute_seconds=compute, activation_dma_seconds=a_dma,
        weight_dma_seconds=b_dma, local_resident_seconds=local_resident,
        local_streaming_seconds=local_streaming,
        resident_full_overlap_seconds=max(compute, a_dma, b_dma, local_resident),
        streaming_no_overlap_seconds=compute+a_dma+b_dma+local_streaming,
    )
    return traffic, latency


class BitletStats:
    def __init__(self, config, trace_path=None):
        self.config = config
        self.phases, self.layers, self.steps = {}, {}, {}
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
        """Require each of the nine operators in every actual model layer."""
        expected = {f"prefill:{name}_{index}": 1
                    for index in range(layers) for name in OPERATORS}
        if decode_steps:
            expected.update({f"decode:{name}_{index}": decode_steps
                             for index in range(layers) for name in OPERATORS})
        observed = {key: value["calls"] for key, value in self.layers.items()}
        if observed != expected:
            raise ValueError("Incomplete Bitlet per-layer/operator/phase coverage")

    def collect(self, name, index, activation, weight, a_spec, b_spec, k, n, phase, context):
        if name not in OPERATORS:
            raise ValueError(f"Unsupported Bitlet operator: {name}")
        meta = dict(context)
        attention = name in ("qk_matmul", "pv_matmul")
        if attention and (activation.ndim != 4 or weight.ndim != 4):
            raise ValueError("Bitlet attention collection requires [B,H,M,K] operands")
        x = activation if attention else activation.reshape(-1, k)
        physical_weight = weight
        if attention and meta.get("shared_kv_gqa"):
            heads, kv = meta["q_heads"], meta["kv_heads"]
            if activation.ndim != 4 or activation.shape[1] != heads or heads % kv:
                raise ValueError("Bitlet GQA metadata disagrees with attention operands")
            group = heads//kv
            batch, _, m, _ = activation.shape
            x = activation.reshape(batch, kv, group*m, k)
            physical_weight = weight[:, ::group]
        seed_key = f"{self.config.sample_seed}:{phase}:{name}:{index}:{meta.get('step', 0)}"
        seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "big") % (2**63)
        from .mapping import Mapping_stat_bitlet
        result = Mapping_stat_bitlet(x, physical_weight, a_spec, b_spec, k, n,
                                     self.config, phase, seed)
        traffic, latency = traffic_and_latency(result, x, physical_weight,
                                               a_spec, b_spec, name, meta, self.config)
        counts = dict(calls=1, exact_calls=int(result["sampling"]["exact"]),
                      sampled_calls=int(not result["sampling"]["exact"]),
                      cycles=result["cycles"], observed=result["observed"],
                      dense_equivalent_operations=result["dense_equivalent_operations"],
                      traffic=traffic, latency=latency)
        key = f"{phase}:{name}_{index}"
        _add_counts(self.phases.setdefault(phase, {}), counts)
        _add_counts(self.layers.setdefault(key, {}), counts)
        step_key = (phase, meta.get("step", 0))
        step = self.steps.setdefault(step_key, dict(phase=phase, context=meta, counts={}))
        _add_counts(step["counts"], counts)
        record = dict(result, operator=name, layer=index, phase=phase, context=meta,
                      activation_format=parse_quant_spec(a_spec).name(),
                      B_format=parse_quant_spec(b_spec).name(),
                      traffic=traffic, latency=latency)
        if self.trace:
            self.trace.write(json.dumps(record, allow_nan=False)+"\n")
        return record

    def export(self, path, workload=None, other_latency=None):
        if self.trace:
            self.trace.flush()
        workload = workload or {}
        scenarios = ("resident_full_overlap_seconds", "streaming_no_overlap_seconds")
        phase_latency = {
            phase: {key: result["latency"][key] for key in scenarios}
            for phase, result in self.phases.items()
        }
        total = {key: sum(value[key] for value in phase_latency.values()) for key in scenarios}
        completed = workload.get("status") == "complete"
        rest = None
        if other_latency is not None:
            validate_other_latency(other_latency, workload.get("decode_steps"))
            prefill = other_latency["prefill_seconds"]
            decode = other_latency["decode_step_seconds"]
            done = workload.get("completed_decode_steps", 0)
            rest = dict(prefill_seconds=prefill, decode_seconds=sum(decode[:done]))
        e2e = (None if rest is None or not completed else
               {key: value+rest["prefill_seconds"]+rest["decode_seconds"]
                for key, value in total.items()})
        decode_steps = sum(phase == "decode" for phase, _ in self.steps)
        doc = dict(
            schema_version=1, backend="bitlet", config=asdict(self.config),
            sources=dict(micro2021=PAPER_URL, tcad2024=EXTENSION_URL),
            paper_parameters=dict(pe_count=32, group_size=64, frequency_hz=1e9,
                mantissa_bits=24, activation_dma_bytes_per_second=12.8e9,
                weight_dma_bytes_per_second=12.8e9, local_buffer_bytes_per_second=25.6e9,
                local_buffer_bytes=None, buffer_capacity_status="not reported in cited papers"),
            mapped_compute_speedups={phase: value["cycles"]["dense_tiles"]/value["cycles"]["mapped_tiles"]
                if value["cycles"]["mapped_tiles"] > 0 else None
                for phase, value in self.phases.items()},
            workload=workload, phases=self.phases, layers=self.layers,
            steps=list(self.steps.values()), trace_file=self.trace_path.name if self.trace_path else None,
            latency=dict(
                scope="conditional GEMM and operand-traffic scenarios; not measured hardware E2E",
                collection_complete=completed, per_phase_seconds=phase_latency,
                gemm_and_io_seconds=total, supplied_other_latency=rest,
                conditional_e2e_seconds=e2e,
                average_decode_gemm_and_io_seconds=(
                    {key: phase_latency["decode"][key]/decode_steps for key in scenarios}
                    if decode_steps else None),
                missing_costs=[] if rest is not None else
                    ["LM head", "Softmax/RMSNorm/RoPE/SiLU/residual", "scale/format conversion"],
            ),
            assumptions=[
                "FP8/INT4 mixed format uses normalized codes promoted to the paper FP32 BCE for statistics.",
                "Floating path uses product exponents EA+EB and full significands including hidden one.",
                "Bits shifted beyond the configured mantissa lanes are counted as truncated; forward is unchanged.",
                "Signs control addition/subtraction; INT4 storage remains full two's complement.",
                "B bits are interleaved; zero A values do not add an undocumented value-skip shortcut.",
                "Synchronized output-column tiles and independent operand scheduling are analytical assumptions.",
                "Uniform wave sampling uses replacement; observed histograms are samples, not full tensor totals.",
                "Sampling subtracts estimated savings from exact dense work; negative estimates are clipped to zero.",
                "Group setup defaults to zero because cycle overhead is not reported; it is configurable.",
                "Resident/full-overlap and streaming/no-overlap are scenarios, not guaranteed physical bounds.",
                "Paper buffer bandwidth is used; capacity, partial-sum spills and control stalls are unreported.",
                "Packed W4 and FP8 traffic is an extension; no native packed kernel or RTL is implemented.",
                "Output/cache writes share the activation DMA in this analytical transport model.",
                "A supplied remaining-operator latency is added serially; it must include the LM head.",
            ],
        )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(doc, indent=2, allow_nan=False)+"\n")
        return doc


def validate_other_latency(document, decode_steps):
    if document.get("schema_version") != 1 or document.get("includes_lm_head") is not True:
        raise ValueError("Other latency needs schema_version=1 and includes_lm_head=true")
    if not isinstance(document.get("decode_step_seconds"), list):
        raise ValueError("decode_step_seconds must be a list")
    if len(document["decode_step_seconds"]) != decode_steps:
        raise ValueError("Other latency must cover every requested decode forward")
    values = [document.get("prefill_seconds"), *document["decode_step_seconds"]]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Other latency values must be finite and nonnegative")
