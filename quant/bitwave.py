"""BitWave B-column skipping, independent of Bitlet bit interleaving.

FP8 is converted losslessly in numeric value to group fixed-point integers.
Wide A groups are expressed as signed 7-bit-magnitude slices that fit an INT8
input. This is an analytical extension, not the paper's measured FP8 datapath.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from .bitlet import (OPERATORS, _add_counts, _canonical, _operand_b,
                     _storage_bits, validate_other_latency)
from .cim_stats import encode_bits
from .quant_spec import parse_quant_spec

PAPER_URL = "https://arxiv.org/abs/2507.12444"
# Table I: Cu -> GEMM K group, OXu -> GEMM M tile, Ku -> GEMM N tile.
# Last two columns are weight/activation SRAM-to-engine bits per cycle.
DATAFLOWS = {
    "SU1": (8, 16, 32, 256, 1024),
    "SU2": (16, 8, 32, 512, 1024),
    "SU3": (32, 4, 32, 1024, 1024),
    "SU4": (8, 1, 128, 1024, 64),
    "SU5": (16, 1, 64, 1024, 128),
    "SU6": (32, 1, 32, 1024, 256),
}


@dataclass(frozen=True)
class BitWaveConfig:
    bce_count: int = 512
    frequency_hz: float = 250e6
    activation_sram_bytes: int = 256*1024
    weight_sram_bytes: int = 256*1024
    sram_banks: int = 16
    sram_bank_bits: int = 64
    dram_bytes_per_second: float | None = None
    output_storage_bytes: int = 2
    group_setup_cycles: int = 0
    dataflow: str = "auto"
    prefill_sample_waves: int = 64
    decode_sample_waves: int = 8
    chunk_waves: int = 4
    sample_seed: int = 23
    bit_flip: bool = False

    def __post_init__(self):
        for key in ("bce_count", "activation_sram_bytes", "weight_sram_bytes",
                    "sram_banks", "sram_bank_bits", "output_storage_bytes", "chunk_waves"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"BitWave {key} must be a positive integer")
        if (self.bce_count, self.sram_banks, self.sram_bank_bits) != (512, 16, 64):
            raise ValueError("Table-I dataflows require 512 BCEs and 16x64-bit SRAM banks")
        for key in ("group_setup_cycles", "prefill_sample_waves",
                    "decode_sample_waves", "sample_seed"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"BitWave {key} must be a nonnegative integer")
        for value in (self.frequency_hz, self.dram_bytes_per_second):
            if value is not None and (isinstance(value, bool) or
                    not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                raise ValueError("BitWave frequency/bandwidth must be finite and positive")
        if self.frequency_hz is None:
            raise ValueError("frequency_hz is required")
        if self.dataflow not in ("auto", *DATAFLOWS):
            raise ValueError("BitWave dataflow must be auto or SU1..SU6 (SU7 is depthwise only)")
        if self.bit_flip is not False:
            raise ValueError("Joint profiling keeps operands unchanged; Bit-Flip needs separate accuracy validation")

    @classmethod
    def from_dict(cls, values):
        values = dict(values or {})
        for key in ("enabled", "trace_path"):
            values.pop(key, None)
        return cls(**values)


def _precision(spec):
    spec = parse_quant_spec(spec)
    _storage_bits(spec)
    if spec.kind == "int":
        return spec.bits, 0
    return (18, 9) if spec.fmt == "e4m3" else (32, 16)


def _bit_length(magnitude):
    return torch.where(magnitude > 0, torch.frexp(magnitude.float())[1].long(), 0)


def fixed_point_groups(values, spec, valid):
    """Return magnitudes and a common power-of-two scale per [...,G].

    Raw FP8 bits are never treated as integer magnitudes. The minimum quantum
    is exact (2^-9 for E4M3FN, 2^-16 for E5M2); common trailing zero removal
    reduces the integer range without changing any nonzero numeric value.
    """
    raw, width = encode_bits(values, spec, "storage")
    limit, fractional = _precision(spec)
    magnitude = (values.detach().float().abs()*(1 << fractional)).long()
    magnitude = torch.where(valid, magnitude, 0)
    negative = valid & (values < 0)
    exponent = torch.zeros_like(magnitude[..., 0])
    if fractional:
        lowbit = magnitude & -magnitude
        trailing = _bit_length(lowbit)-1
        common = torch.where(magnitude != 0, trailing, 100).amin(-1)
        common = torch.where((magnitude != 0).any(-1), common, 0)
        magnitude = magnitude >> common.unsqueeze(-1)
        exponent = torch.where((magnitude != 0).any(-1), common-fractional, 0)
    return magnitude, negative, exponent, raw, width, limit


def column_work(a, b, a_spec, b_spec, valid_a, valid_b):
    """A=[...,Mu,G], B=[...,Nu,G]; count B columns, not their populations."""
    am, aneg, ae, _, _, _ = fixed_point_groups(a, a_spec, valid_a)
    bm, bneg, be, raw, storage_width, limit = fixed_point_groups(b, b_spec, valid_b)
    a_group = valid_a.any(-1)
    b_group = valid_b.any(-1)
    a_magnitude_bits = _bit_length(am.amax(-1))
    positive = torch.where(~aneg, am, 0).amax(-1)
    negative = torch.where(aneg, am, 0).amax(-1)
    # Signed INT8 includes -128, while +128 requires a wider input.
    a_signed_bits = torch.maximum(_bit_length(positive),
                                  _bit_length((negative-1).clamp_min(0)))+1
    a_passes = torch.where(a_signed_bits <= 8, 1,
                          torch.div(a_magnitude_bits+6, 7, rounding_mode="floor"))
    a_passes = torch.where(a_group, a_passes, 0)
    positions = torch.arange(limit, device=b.device)
    populations = ((bm.unsqueeze(-1) >> positions) & 1).sum(-2)
    nonzero = populations != 0
    nz_columns = nonzero.sum(-1)
    b_magnitude_bits = _bit_length(bm.amax(-1))
    b_passes = torch.div(b_magnitude_bits+6, 7, rounding_mode="floor").clamp_min(1)
    sign_columns = torch.zeros_like(b_passes)
    for word in range(math.ceil(limit/7)):
        sign_columns += (bneg & (((bm >> (7*word)) & 127) != 0)).any(-1)
    # Each INT8 ZCIP index is 8 bits, including the sign-column flag.
    # FP8 extension adds an exponent and word-count byte per real group.
    float_metadata = 16 if parse_quant_spec(b_spec).kind == "fp" else 0
    compressed_bits = (8*b_passes + b.shape[-1]*(nz_columns+sign_columns)+float_metadata)
    compressed_bits = torch.where(b_group, compressed_bits, 0)
    dense_bits = torch.where(b_group, b.shape[-1]*(limit+1)+float_metadata, 0)
    wave = a_passes.amax(-1)*nz_columns.amax(-1)
    dense = a_passes.amax(-1)*limit
    ideal = (a_passes.sum(-1)*nz_columns.sum(-1)).double()/(a.shape[-2]*b.shape[-2])
    raw_bits = ((raw.unsqueeze(-1) >> torch.arange(storage_width, device=b.device)) & 1)
    scalars = torch.stack([
        b_group.sum(), valid_b.sum(), (valid_b & (bm == 0)).sum(),
        (valid_b & bneg).sum(), sign_columns[b_group].sum(),
        (nonzero*b_group.unsqueeze(-1)).sum(), (~nonzero*b_group.unsqueeze(-1)).sum(),
        (raw_bits*valid_b.unsqueeze(-1)).sum(), a_group.sum(),
        (a_group & (a_signed_bits > 8)).sum(),
        (b_group & (b_magnitude_bits > 7)).sum(),
    ]).cpu().tolist()
    keys = ("observed_B_groups", "observed_B_elements", "observed_B_zeros",
            "observed_B_negative_values", "observed_sign_columns",
            "observed_nonzero_magnitude_columns", "observed_zero_magnitude_columns",
            "observed_source_B_one_bits", "observed_A_groups",
            "observed_A_groups_exceeding_int8", "observed_B_groups_exceeding_native_magnitude")
    observed = dict(zip(keys, scalars))
    observed["observed_source_B_bits"] = scalars[1]*storage_width
    for key, tensor, size in (
        ("nonzero_columns_histogram", nz_columns[b_group], limit+1),
        ("A_signed_width_histogram", a_signed_bits[a_group], 34),
        ("B_magnitude_width_histogram", b_magnitude_bits[b_group], 33),
        ("column_population_histogram", populations[b_group].reshape(-1), b.shape[-1]+1),
    ):
        observed[key] = torch.bincount(tensor, minlength=size).cpu().tolist()
    # A is loaded as signed INT8 slices, broadcast across all Nu output columns.
    return dict(cycles=wave, dense=dense, ideal=ideal, observed=observed,
                local_A_bytes=(a_passes.sum(-1)*a.shape[-1]+
                    (2*a_group.sum(-1) if parse_quant_spec(a_spec).kind == "fp" else 0)).double(),
                compressed_B_bytes=compressed_bits.sum(-1).double()/8,
                dense_B_bytes=dense_bits.sum(-1).double()/8,
                A_quantum_exponents=ae, B_quantum_exponents=be)


def _measure_dataflow(x, b, a_spec, b_spec, config, su, phase, seed):
    operands, m, k = x.shape
    n = b.shape[-1]
    group, mu, nu, w_bw, a_bw = DATAFLOWS[su]
    mt, nt, kr = math.ceil(m/mu), math.ceil(n/nu), math.ceil(k/group)
    total_per_operand = mt*nt*kr
    budget = config.decode_sample_waves if phase == "decode" else config.prefill_sample_waves
    sampled = budget > 0 and total_per_operand > budget
    observations = budget if sampled else total_per_operand
    generator = torch.Generator(device="cpu").manual_seed(seed)
    totals = dict(mapped_tiles=0., dense_tiles=0., ideal_pe_balance=0.,
                  local_A_read_bytes=0., local_B_read_bytes_streaming=0.,
                  dense_B_read_bytes_streaming=0.)
    observed, variance = {}, 0.
    for operand in range(operands):
        matrix = _operand_b(b, operand)
        sampled_ids = (torch.randint(total_per_operand, (observations,), generator=generator)
                       if sampled else None)
        sums = [0.]*len(totals)
        square_sum = 0.
        for start in range(0, observations, config.chunk_waves):
            stop = min(observations, start+config.chunk_waves)
            ids = (sampled_ids[start:stop] if sampled else torch.arange(start, stop)).to(x.device)
            kg, col, row = ids % kr, (ids//kr) % nt, ids//(kr*nt)
            ki = kg[:, None]*group+torch.arange(group, device=x.device)
            mi = row[:, None]*mu+torch.arange(mu, device=x.device)
            ni = col[:, None]*nu+torch.arange(nu, device=x.device)
            va = (mi[:, :, None] < m) & (ki[:, None, :] < k)
            vb = (ni[:, :, None] < n) & (ki[:, None, :] < k)
            a = x[operand, mi.clamp_max(m-1)[:, :, None], ki.clamp_max(k-1)[:, None, :]]
            w = matrix[ki.clamp_max(k-1)[:, None, :], ni.clamp_max(n-1)[:, :, None]]
            result = column_work(a, w, a_spec, b_spec, va, vb)
            mapped = result["cycles"]+config.group_setup_cycles
            dense = result["dense"]+config.group_setup_cycles
            ideal = result["ideal"]+config.group_setup_cycles*va.any(-1).sum(-1)*vb.any(-1).sum(-1)/(mu*nu)
            moments = torch.stack([mapped.sum(), dense.sum(), ideal.sum(),
                result["local_A_bytes"].sum(), result["compressed_B_bytes"].sum(),
                result["dense_B_bytes"].sum(), mapped.double().square().sum()]).cpu().tolist()
            sums = [left+right for left, right in zip(sums, moments[:-1])]
            square_sum += moments[-1]
            _add_counts(observed, result["observed"])
        scale = total_per_operand/observations
        for key, value in zip(totals, sums):
            totals[key] += value*scale
        if sampled and observations > 1:
            variance += total_per_operand**2*max(
                0., (square_sum-sums[0]**2/observations)/(observations-1))/observations
    standard_error = math.sqrt(variance) if not sampled or observations > 1 else None
    cycle_count = totals["mapped_tiles"]
    local_a = totals.pop("local_A_read_bytes")
    local_b = totals.pop("local_B_read_bytes_streaming")
    dense_b = totals.pop("dense_B_read_bytes_streaming")
    output = operands*m*n*config.output_storage_bytes
    local = dict(activation_read_bytes=local_a,
                 B_read_bytes_resident=local_b/mt,
                 B_read_bytes_streaming=local_b,
                 output_write_bytes=output)
    # This fit check covers one tile only, not all temporal reuse/spills.
    a_limit, _ = _precision(a_spec)
    b_limit, _ = _precision(b_spec)
    a_tile = mu*group*max(1, math.ceil(a_limit/7))+mu*nu*config.output_storage_bytes
    b_tile = math.ceil(nu*(group*(b_limit+math.ceil(b_limit/7))+
                          8*math.ceil(b_limit/7)+16)/8)
    if a_tile > config.activation_sram_bytes or b_tile > config.weight_sram_bytes:
        raise ValueError(f"{su}: SRAM cannot fit one extended input/output tile")
    compute = cycle_count/config.frequency_hz
    a_seconds = local_a/(a_bw/8*config.frequency_hz)
    b_seconds = local_b/(w_bw/8*config.frequency_hz)
    return dict(
        layout=dict(dataflow=su, group_size=group, m_tile=mu, n_tile=nu,
                    m=m, n=n, k=k, operands=operands, m_tiles=mt, n_tiles=nt, k_groups=kr,
                    active_bce_slots=mu*nu*(group//8),
                    weight_bits_per_cycle=w_bw, activation_bits_per_cycle=a_bw),
        cycles=totals, compute_seconds={key: value/config.frequency_hz for key, value in totals.items()},
        observed=observed, local_traffic=local,
        compression=dict(estimated_BCS_B_bytes=local_b/mt,
                         estimated_dense_fixedpoint_B_bytes=dense_b/mt,
                         BCS_vs_dense_fixedpoint_ratio=dense_b/local_b if local_b else None,
                         BCS_vs_original_storage_ratio=math.ceil(operands*k*n*_storage_bits(b_spec)/8)/(local_b/mt)
                         if local_b else None,
                         includes_signs_indexes_and_FP8_metadata=True),
        sampling=dict(method="uniform_with_replacement" if sampled else "exact",
                      exact=not sampled, seed=seed, sample_waves=operands*observations,
                      total_waves=operands*total_per_operand,
                      mapped_cycles_standard_error=standard_error,
                      mapped_cycles_approximate_95pct_interval=(
                          [max(0., cycle_count-1.96*standard_error), cycle_count+1.96*standard_error]
                          if standard_error is not None else None)),
        minimum_tile_bytes=dict(activation=a_tile, weight=b_tile),
        selection_compute_plus_local_streaming_seconds=compute+a_seconds+b_seconds,
    )


def measure_bitwave_mapping(activation, weight, a_spec, b_spec, k, n,
                            config, phase="prefill", seed=None):
    x, b = _canonical(activation, weight, k, n)
    seed = config.sample_seed if seed is None else seed
    candidates = {}
    for su in (DATAFLOWS if config.dataflow == "auto" else (config.dataflow,)):
        su_seed = int.from_bytes(hashlib.sha256(f"{seed}:{su}".encode()).digest()[:8], "big") % (2**63)
        try:
            candidates[su] = _measure_dataflow(x, b, a_spec, b_spec, config, su, phase, su_seed)
        except ValueError as error:
            if "SRAM cannot fit" not in str(error):
                raise
            if config.dataflow != "auto":
                raise
    if not candidates:
        raise ValueError("No BitWave dataflow fits the configured SRAM")
    selected = min(candidates, key=lambda key: candidates[key]["selection_compute_plus_local_streaming_seconds"])
    result = candidates[selected]
    result.update(operand_shape=list(x.shape), weight_shape=list(weight.shape),
                  dense_equivalent_operations=2*x.shape[0]*x.shape[1]*k*n,
                  mapped_compute_speedup=(result["cycles"]["dense_tiles"]/result["cycles"]["mapped_tiles"]
                      if result["cycles"]["mapped_tiles"] else None),
                  integer_operand_formats=parse_quant_spec(a_spec).kind == parse_quant_spec(b_spec).kind == "int",
                  alignment="lossless_group_fixedpoint_and_signed_int8_magnitude_slices",
                  candidates={su: {key: value[key] for key in ("layout", "cycles", "sampling", "compression",
                        "selection_compute_plus_local_streaming_seconds")}
                              for su, value in candidates.items()},
                  dataflow_selection="minimum estimated compute plus streaming local reads; not ZigZag")
    return result


def traffic_and_latency(result, activation, weight, a_spec, b_spec, name, context, config):
    a_read = math.ceil(activation.numel()*_storage_bits(a_spec)/8)
    b_read = math.ceil(weight.numel()*_storage_bits(b_spec)/8)
    append = 0
    attention = name in ("qk_matmul", "pv_matmul")
    if attention:
        for key in ("kv_heads", "head_dim", "cache_length_before", "query_length"):
            if key not in context:
                raise ValueError("BitWave attention traffic requires cache/head provenance")
        cells = activation.shape[0]*context["kv_heads"]*context["head_dim"]
        b_read = math.ceil(cells*context["cache_length_before"]*_storage_bits(b_spec)/8)
        append = math.ceil(cells*context["query_length"]*_storage_bits(b_spec)/8)
    local = result["local_traffic"]
    output = local["output_write_bytes"]
    traffic = dict(activation_read_bytes_minimum=a_read, B_read_bytes_minimum=b_read,
                   output_write_bytes=output, fp8_kv_append_bytes=append,
                   packed_weight_read_bytes_minimum=0 if attention else b_read,
                   fp8_kv_read_bytes_minimum=b_read if attention else 0,
                   **{f"local_{key}": value for key, value in local.items()})
    su = result["layout"]
    a_seconds = local["activation_read_bytes"]/(su["activation_bits_per_cycle"]/8*config.frequency_hz)
    b_seconds = local["B_read_bytes_resident"]/(su["weight_bits_per_cycle"]/8*config.frequency_hz)
    bs_seconds = local["B_read_bytes_streaming"]/(su["weight_bits_per_cycle"]/8*config.frequency_hz)
    out_seconds = output/(config.sram_banks*config.sram_bank_bits/8*config.frequency_hz)
    compute = result["compute_seconds"]["mapped_tiles"]
    latency = dict(mapped_compute_seconds=compute, local_activation_seconds=a_seconds,
                   local_B_resident_seconds=b_seconds, local_B_streaming_seconds=bs_seconds,
                   local_output_write_seconds=out_seconds)
    if config.dram_bytes_per_second is not None:
        dram = (a_read+b_read+output+append)/config.dram_bytes_per_second
        latency.update(dram_seconds=dram,
            paper_equation5_resident_seconds=dram+out_seconds+max(compute, a_seconds, b_seconds),
            streaming_no_overlap_seconds=dram+out_seconds+compute+a_seconds+bs_seconds)
    return traffic, latency


class BitWaveStats:
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
        expected = {f"prefill:{name}_{index}": 1 for index in range(layers) for name in OPERATORS}
        if decode_steps:
            expected.update({f"decode:{name}_{index}": decode_steps
                             for index in range(layers) for name in OPERATORS})
        if {key: value["calls"] for key, value in self.layers.items()} != expected:
            raise ValueError("Incomplete BitWave per-layer/operator/phase coverage")

    def collect(self, name, index, activation, weight, a_spec, b_spec, k, n, phase, context):
        if name not in OPERATORS:
            raise ValueError(f"Unsupported BitWave operator: {name}")
        meta = dict(context)
        attention = name in ("qk_matmul", "pv_matmul")
        if attention and (activation.ndim != 4 or weight.ndim != 4):
            raise ValueError("BitWave attention requires [B,H,M,K] operands")
        x = activation if attention else activation.reshape(-1, k)
        physical_weight = weight
        if attention and meta.get("shared_kv_gqa"):
            heads, kv = meta["q_heads"], meta["kv_heads"]
            if activation.shape[1] != heads or heads % kv:
                raise ValueError("BitWave GQA metadata disagrees with attention operands")
            group = heads//kv
            batch, _, m, _ = activation.shape
            x = activation.reshape(batch, kv, group*m, k)
            physical_weight = weight[:, ::group]
        key = f"{self.config.sample_seed}:{phase}:{name}:{index}:{meta.get('step', 0)}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**63)
        from .mapping import Mapping_stat_bitwave
        result = Mapping_stat_bitwave(x, physical_weight, a_spec, b_spec, k, n,
                                      self.config, phase, seed)
        traffic, latency = traffic_and_latency(result, x, physical_weight, a_spec, b_spec, name, meta, self.config)
        counts = dict(calls=1, exact_calls=int(result["sampling"]["exact"]),
            sampled_calls=int(not result["sampling"]["exact"]),
            integer_operand_calls=int(result["integer_operand_formats"]),
            selected_dataflows={result["layout"]["dataflow"]: 1},
            cycles=result["cycles"], observed=result["observed"],
            dense_equivalent_operations=result["dense_equivalent_operations"],
            traffic=traffic, latency=latency)
        _add_counts(self.phases.setdefault(phase, {}), counts)
        _add_counts(self.layers.setdefault(f"{phase}:{name}_{index}", {}), counts)
        step = self.steps.setdefault((phase, meta.get("step", 0)), dict(phase=phase, context=meta, counts={}))
        _add_counts(step["counts"], counts)
        record = dict(result, operator=name, layer=index, phase=phase, context=meta,
                      activation_format=parse_quant_spec(a_spec).name(),
                      B_format=parse_quant_spec(b_spec).name(), traffic=traffic, latency=latency)
        if self.trace:
            self.trace.write(json.dumps(record, allow_nan=False)+"\n")
        return record

    def export(self, path, workload=None, other_latency=None):
        if self.trace:
            self.trace.flush()
        workload = workload or {}
        completed = workload.get("status") == "complete"
        scenarios = ("paper_equation5_resident_seconds", "streaming_no_overlap_seconds")
        io_available = self.config.dram_bytes_per_second is not None
        phases = ({phase: {key: counts["latency"][key] for key in scenarios}
                   for phase, counts in self.phases.items()} if io_available else None)
        total = ({key: sum(value[key] for value in phases.values()) for key in scenarios}
                 if io_available else None)
        rest = None
        if other_latency is not None:
            validate_other_latency(other_latency, workload.get("decode_steps"))
            rest = dict(prefill_seconds=other_latency["prefill_seconds"],
                        decode_seconds=sum(other_latency["decode_step_seconds"][
                            :workload.get("completed_decode_steps", 0)]))
        e2e = ({key: value+rest["prefill_seconds"]+rest["decode_seconds"] for key, value in total.items()}
               if completed and total is not None and rest is not None else None)
        doc = dict(
            schema_version=1, backend="bitwave", config=asdict(self.config),
            sources=dict(hpca2024=PAPER_URL),
            paper_parameters=dict(bce_count=512, smm_count=4096, frequency_hz=250e6,
                native_activation_bits=8, native_weight_bits=8,
                activation_sram_bytes=256*1024, weight_sram_bytes=256*1024,
                sram_banks=16, sram_bank_bits=64, dataflows=DATAFLOWS,
                dram_bytes_per_second=None, dram_bandwidth_status="not specified in cited paper"),
            workload=workload, phases=self.phases, layers=self.layers, steps=list(self.steps.values()),
            mapped_compute_speedups={phase: value["cycles"]["dense_tiles"]/value["cycles"]["mapped_tiles"]
                if value["cycles"]["mapped_tiles"] else None for phase, value in self.phases.items()},
            trace_file=self.trace_path.name if self.trace_path else None,
            latency=dict(scope="conditional integer-slice extension; not measured FP8 BitWave latency",
                collection_complete=completed, dram_bandwidth_supplied=io_available,
                per_phase_seconds=phases, gemm_and_io_seconds=total,
                supplied_other_latency=rest, conditional_e2e_seconds=e2e,
                missing_costs=([] if rest is not None else
                    ["LM head", "Softmax/RMSNorm/RoPE/SiLU/residual", "scale/format conversion"]) +
                    ([] if io_available else ["DRAM bandwidth"])),
            assumptions=[
                "The paper's native arithmetic is INT8 activation with sign-magnitude bit-column weights.",
                "FP8 uses exact numeric group fixed-point alignment; negative zero becomes integer zero.",
                "Wide activations use signed 7-bit-magnitude slices; wider B uses multiple INT8 index words.",
                "Slice accumulation and group exponent conversion costs are not published and must be supplied.",
                "Bit-Flip is disabled so both architectures observe identical quantized model operands.",
                "B work is the number of nonzero magnitude columns, not the maximum column population.",
                "Sign columns cost storage/loading and control product signs; they are not magnitude MAC cycles.",
                "Compression includes 8-bit indexes, word signs, padded K groups, and two FP8 metadata bytes.",
                "DRAM transport stays packed W4 and FP8 KV; BCS compression is a diagnostic and local-read scenario.",
                "Auto SU selection minimizes estimated compute plus streaming local reads; it is not ZigZag.",
                "Sampling and dataflow selection can introduce uncertainty and selection bias; histograms are observed samples.",
                "Tile fit is checked against separate SRAMs; full reuse, register transfers and partial-sum spills are not simulated.",
                "Eq.5-like latency adds DRAM and output writes before max of compute and SRAM input reads.",
                "Register traffic, format conversion, index control and setup stalls may add latency.",
                "Bandwidth overrides and output_storage_bytes=2/group_setup_cycles=0 are explicit modeling assumptions.",
            ])
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(doc, indent=2, allow_nan=False)+"\n")
        return doc
