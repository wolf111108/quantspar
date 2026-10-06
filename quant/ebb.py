"""MulTCIM-inspired effective-bit statistics and conditional compute bounds.

The paper implements INT8/INT16, not FP8. FP8 groups below are converted
losslessly to shared-exponent integers. The ideal equalizer is a lower bound,
not a reconstruction of the unpublished routing/scheduling controller.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from .quant_spec import parse_quant_spec, fp8_dtype


@dataclass(frozen=True)
class EBBConfig:
    input_bits: int = 16
    linear_weight_bits: int = 8
    attention_weight_bits: int = 16
    frequency_hz: float = 160e6
    macros: int = 128
    arrays_per_macro: int = 32
    banks: int = 8
    rows: int = 4
    group_size: int = 8
    chunk_rows: int = 32
    signed_encoding: str = "twos_complement"
    fp8_alignment: str = "exact_dyadic"
    group_setup_cycles: int = 0

    def __post_init__(self):
        if self.input_bits not in (8, 16):
            raise ValueError("EBB input_bits must be 8 or 16")
        if any(v not in (8, 16) for v in
               (self.linear_weight_bits, self.attention_weight_bits)):
            raise ValueError("EBB physical weight words must be 8 or 16 bits")
        if (self.banks, self.rows, self.group_size) != (8, 4, 8):
            raise ValueError("This EBB model uses paper geometry: 8 banks, 4 rows, group 8")
        if min(self.macros, self.arrays_per_macro, self.chunk_rows) <= 0:
            raise ValueError("EBB geometry and chunk_rows must be positive")
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("EBB frequency must be finite and positive")
        if self.signed_encoding not in ("twos_complement", "sign_magnitude"):
            raise ValueError("Unknown EBB signed_encoding")
        if self.fp8_alignment != "exact_dyadic":
            raise ValueError("Only lossless exact_dyadic FP8 alignment is implemented")
        if not isinstance(self.group_setup_cycles, int) or self.group_setup_cycles < 0:
            raise ValueError("group_setup_cycles must be a nonnegative integer")

    @classmethod
    def from_dict(cls, value):
        value = dict(value or {})
        for name in ("enabled", "trace_path"):
            value.pop(name, None)
        return cls(**value)


def _bit_length(values):
    return torch.where(values > 0,
                       torch.floor(torch.log2(values.clamp_min(1).double())).long() + 1,
                       torch.zeros_like(values))


def _hist(values, size=48):
    return torch.bincount(values.reshape(-1), minlength=size).cpu().tolist()


def add_counts(target, source):
    """Add integer counters/histograms without averaging per-layer ratios."""
    for key, value in source.items():
        if isinstance(value, list):
            old = target.setdefault(key, [0] * len(value))
            if len(old) < len(value):
                old.extend([0] * (len(value) - len(old)))
            for i, count in enumerate(value):
                old[i] += count
        elif isinstance(value, dict):
            add_counts(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0) + value


def effective_bit_groups(values, spec, config):
    """Group adjacent K entries; never join different rows or head operands.

    For E4M3, codes * 2**9 are exact integers (E5M2 uses 2**16).
    Align to the smallest encoded LSB exponent of nonzero entries in the group.
    Keep trailing zero significand positions: 1.0 has significand 1000, not 1.
    This keeps the complete significand, exponent span and sign. No clipping
    or requantization is applied to the model's numerical forward pass.
    """
    spec = parse_quant_spec(spec)
    if values.ndim != 2 or values.shape[-1] <= 0:
        raise ValueError("EBB expects nonempty [rows,K] operands")
    x = values.detach().float()
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite EBB input")
    if spec.kind == "int":
        if ((x != x.round()) | (x < -(1 << (spec.bits - 1))) |
                (x >= (1 << (spec.bits - 1)))).any():
            raise ValueError("EBB integer inputs must be quantized codes")
        integer = x.long()
    elif spec.kind == "fp" and spec.fmt in ("e4m3", "e5m2"):
        encoded = x.to(fp8_dtype(spec))
        if (encoded.float() != x).any():
            raise ValueError("EBB FP8 inputs must be actual normalized quantized codes")
        integer = (x * (512 if spec.fmt == "e4m3" else 65536)).long()
        mb, eb = (3, 4) if spec.fmt == "e4m3" else (2, 5)
        exponent = (encoded.contiguous().view(torch.uint8).long() >> mb) & ((1 << eb) - 1)
        fp_shift = (exponent - 1).clamp_min(0)
    else:
        raise ValueError("EBB supports integer codes and E4M3/E5M2 codes")

    valid = torch.ones_like(integer, dtype=torch.bool)
    tail = (-x.shape[-1]) % config.group_size
    if tail:
        integer = torch.nn.functional.pad(integer, (0, tail))
        valid = torch.nn.functional.pad(valid, (0, tail), value=False)
        if spec.kind == "fp":
            fp_shift = torch.nn.functional.pad(fp_shift, (0, tail))
    q = integer.reshape(x.shape[0], -1, config.group_size)
    valid = valid.reshape_as(q)
    if spec.kind == "fp":
        shifts = fp_shift.reshape_as(q)
        shifts = torch.where(q != 0, shifts, torch.full_like(shifts, 63))
        common = shifts.amin(-1, keepdim=True).clamp_max(62)
        q = q >> common
    magnitude = q.abs()
    mag_bits = _bit_length(magnitude)
    # Smallest signed two's-complement width, including the asymmetric -128.
    signed_bits = torch.where(q < 0, _bit_length((magnitude - 1).clamp_min(0)) + 1,
                              mag_bits + (q > 0).long())
    required = signed_bits.amax(-1)
    if config.signed_encoding == "sign_magnitude":
        required = (mag_bits + (q != 0).long()).amax(-1)
    # Overflow is retained and reported, rather than silently clipped to 16b.
    words = required.clamp_min(config.input_bits)
    if config.signed_encoding == "twos_complement":
        eb = torch.where(q < 0, words.unsqueeze(-1), mag_bits)
    else:
        eb = mag_bits + (q != 0).long()  # an explicit serial sign-slot assumption
    eb = torch.where(valid, eb, 0)
    maximum = eb.amax(-1)
    ideal = (eb.sum(-1) + config.group_size - 1) // config.group_size
    minimum = eb.amin(-1)
    counts = dict(
        elements=int(valid.sum()), zero_elements=int(((q == 0) & valid).sum()),
        negative_elements=int(((q < 0) & valid).sum()), groups=required.numel(),
        all_zero_groups=int((maximum == 0).sum()),
        imbalanced_groups=int((maximum != minimum).sum()),
        overflow_int8_groups=int((required > 8).sum()),
        overflow_int16_groups=int((required > 16).sum()),
        overflow_configured_groups=int((required > config.input_bits).sum()),
        effective_bits_histogram=_hist(eb[valid]),
        required_signed_bits_histogram=_hist(required),
        group_max_bits_histogram=_hist(maximum),
        dense_group_cycles=int(words.sum()),
        leading_zero_group_cycles=int(maximum.sum()),
        ideal_balanced_group_cycles=int(ideal.sum()),
    )
    return dict(dense=words, leading_zero=maximum, ideal_balanced=ideal), counts


def _layout(config, m, k, n, weight_bits):
    # INT16 weight uses two 8b bank passes, halves resident K capacity.
    passes = weight_bits // 8
    groups = math.ceil(k / config.group_size)
    resident_groups = config.rows // passes
    kt = math.ceil(groups / resident_groups)
    nt = math.ceil(n / config.arrays_per_macro)
    factors = [i for i in range(1, config.macros + 1) if config.macros % i == 0]
    best = None
    for kf in factors:
        if kf > kt:
            continue
        for nf in factors:
            if nf > nt or kf * nf > config.macros:
                continue
            mf = min(m, config.macros // (kf * nf))
            lane_groups = math.ceil(kt / kf) * resident_groups
            score = math.ceil(m / mf) * math.ceil(nt / nf) * lane_groups
            candidate = (score, -(kf * mf * nf), kf, mf, nf, lane_groups)
            if best is None or candidate < best:
                best = candidate
    _, _, kf, mf, nf, lane_groups = best
    return dict(k_parallel=kf, token_parallel=mf, n_parallel=nf,
                groups_per_lane=lane_groups, n_rounds=math.ceil(nt / nf),
                weight_bank_passes=passes)


def measure_ebb_mapping(values, spec, in_features, out_features, config,
                        weight_bits=8):
    """A fixed dense-selected macro layout, synchronous token-wave barriers.

    Independent head operands run sequentially. The caller may join the five
    query heads sharing one KV head into rows, explicitly modeling GQA reuse.
    K groups on each macro sum their costs; each wave uses its slowest macro.
    """
    if values.ndim == 2:
        x = values.unsqueeze(0)
    elif values.ndim in (3, 4):
        x = values.reshape(-1, values.shape[-2], values.shape[-1])
    else:
        raise ValueError("EBB mapping expects [M,K], [B,M,K] or [B,H,M,K]")
    if x.shape[-1] != in_features or min(in_features, out_features, x.shape[-2]) <= 0:
        raise ValueError("EBB mapping dimensions disagree with operands")
    layout = _layout(config, x.shape[-2], in_features, out_features, weight_bits)
    mf, kf = layout["token_parallel"], layout["k_parallel"]
    capacity = kf * layout["groups_per_lane"]
    # Align chunks to token waves so chunking does not add artificial barriers.
    chunk = max(mf, (config.chunk_rows // mf) * mf)
    counts = {}
    cycles = dict(dense=0, leading_zero=0, ideal_balanced=0)
    for operand in x:
        for start in range(0, operand.shape[0], chunk):
            costs, part_counts = effective_bit_groups(operand[start:start + chunk], spec, config)
            add_counts(counts, part_counts)
            for name, cost in costs.items():
                cost = cost * layout["weight_bank_passes"] + config.group_setup_cycles
                cost = torch.nn.functional.pad(cost, (0, capacity - cost.shape[-1],
                                                       0, (-cost.shape[0]) % mf))
                lane = cost.reshape(-1, mf, kf, layout["groups_per_lane"]).sum(-1)
                cycles[name] += int(lane.amax(dim=(1, 2)).sum()) * layout["n_rounds"]
    return dict(layout=layout, cycles=cycles, activation=counts,
                input_shape=list(values.shape), operand_shape=list(x.shape),
                in_features=in_features, out_features=out_features,
                compute_seconds={key: val / config.frequency_hz for key, val in cycles.items()},
                ideal_speedup=cycles["dense"] / cycles["ideal_balanced"]
                if cycles["ideal_balanced"] else None,
                leading_zero_speedup=cycles["dense"] / cycles["leading_zero"]
                if cycles["leading_zero"] else None)


def operand_counts(values, spec, config):
    rows = values.reshape(-1, values.shape[-1])
    counts = {}
    for start in range(0, rows.shape[0], config.chunk_rows):
        _, part = effective_bit_groups(rows[start:start + config.chunk_rows], spec, config)
        add_counts(counts, part)
    return counts


class EBBStats:
    def __init__(self, config, trace_path=None):
        self.config = config
        self.layers = {}
        self.phases = {}
        self.steps = {}
        self.static_weights = {}
        self.dynamic_cache = {}
        self.trace_path = Path(trace_path) if trace_path else None
        self.trace = None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace = self.trace_path.open("x", encoding="utf-8")

    def close(self):
        if self.trace:
            self.trace.close()
            self.trace = None

    def _weight_counts(self, name, index, phase, weight, spec, context):
        """Incremental validation of immutable FP8 cache entries, not sampling.

        QK groups run along head_dim, so each new token adds complete rows.
        PV groups run along cache length: keep the completed group prefix and
        recount the last (at most seven-token) partial group on every append.
        Historical normalized codes stay unchanged under the fixed B scales.
        """
        attention = name in ("qk_matmul", "pv_matmul")
        key = f"{phase}:{name}_{index}"
        if not attention:
            if key not in self.static_weights:
                self.static_weights[key] = operand_counts(weight, spec, self.config)
            return self.static_weights[key]
        rows = weight.transpose(-2, -1)
        if context.get("kv_cache_encoding") != "fp8_static_dequantized_emulation":
            return operand_counts(rows, spec, self.config)
        seq = rows.shape[-2] if name == "qk_matmul" else rows.shape[-1]
        signature = tuple(rows.shape[:-2]) + (rows.shape[-1],) if name == "qk_matmul" else tuple(rows.shape[:-1])
        cache_key = (name, index, spec.name(), signature)
        state = self.dynamic_cache.get(cache_key)
        if state is None or seq < state["last_seq"] or context.get("cache_length_before", 0) == 0:
            state = dict(completed=0, last_seq=0, counts={})
            self.dynamic_cache[cache_key] = state
        if name == "qk_matmul":
            if seq > state["completed"]:
                add_counts(state["counts"], operand_counts(rows[..., state["completed"]:seq, :],
                                                           spec, self.config))
            state["completed"] = seq
            result = state["counts"]
        else:
            completed = seq // self.config.group_size * self.config.group_size
            if completed > state["completed"]:
                add_counts(state["counts"], operand_counts(rows[..., state["completed"]:completed],
                                                           spec, self.config))
            state["completed"] = completed
            result = {}
            add_counts(result, state["counts"])
            if seq > completed:
                add_counts(result, operand_counts(rows[..., completed:seq], spec, self.config))
        state["last_seq"] = seq
        return result

    def collect(self, name, index, activation, weight, a_spec, w_spec,
                k, n, phase, context):
        config = self.config
        attention = name in ("qk_matmul", "pv_matmul")
        meta = dict(context)
        x = activation if attention else activation.reshape(-1, k)
        physical_weight = weight
        shared = attention and bool(meta.get("shared_kv_gqa"))
        if shared:
            heads, kv = meta["q_heads"], meta["kv_heads"]
            if activation.ndim != 4 or activation.shape[1] != heads or heads % kv:
                raise ValueError("EBB GQA metadata disagrees with attention input")
            group = heads // kv
            b, _, m, _ = activation.shape
            x = activation.reshape(b, kv, group * m, k)
            physical_weight = weight[:, ::group]
        word_bits = config.attention_weight_bits if attention else config.linear_weight_bits
        from .mapping import Mapping_stat_ebb
        result = Mapping_stat_ebb(x, a_spec, k, n, config, word_bits)
        key = f"{phase}:{name}_{index}"
        weight_counts = self._weight_counts(name, index, phase, physical_weight, w_spec, meta)
        histogram = weight_counts["required_signed_bits_histogram"]
        weight_overflows = sum(histogram[word_bits + 1:])
        traffic = dict(packed_weight_read_bytes_minimum=0,
                       fp8_kv_read_bytes_minimum=0, fp8_kv_append_bytes=0)
        if attention:
            before = meta.get("cache_length_before", 0)
            batch = activation.shape[0]
            kv = meta.get("kv_heads", physical_weight.shape[1])
            dim = meta.get("head_dim", 0)
            traffic["fp8_kv_read_bytes_minimum"] = batch * kv * dim * before
            traffic["fp8_kv_append_bytes"] = batch * kv * dim * meta.get("query_length", 0)
        else:
            traffic["packed_weight_read_bytes_minimum"] = math.ceil(weight.numel() * w_spec.bits / 8)
        counters = dict(calls=1, cycles=result["cycles"], traffic=traffic,
                        activation=result["activation"],
                        weight_overflow_occurrences=weight_overflows,
                        activation_overflow_occurrences=result["activation"]["overflow_configured_groups"])
        layer = self.layers.setdefault(key, dict(layer_name=name, layer_idx=index, phase=phase,
                     activation_format=a_spec.name(), weight_format=w_spec.name(),
                     physical_weight_bits=word_bits, counts={}))
        add_counts(layer["counts"], counters)
        # Dynamic weight histograms count read occurrences; static histograms are unique.
        if attention or "weight" not in layer:
            add_counts(layer.setdefault("weight", {}), weight_counts)
        add_counts(self.phases.setdefault(phase, {}), counters)
        step_key = f"{phase}:{meta.get('step', 0)}"
        step = self.steps.setdefault(step_key, dict(phase=phase, context=meta.copy(), counts={}))
        add_counts(step["counts"], {key: value for key, value in counters.items() if key != "activation"})
        record = dict(layer_name=name, layer_idx=index, phase=phase,
                      context=meta, input_shape=list(activation.shape),
                      operand_shape=result["operand_shape"], in_features=k, out_features=n,
                      layout=result["layout"], cycles=result["cycles"], traffic=traffic,
                      activation_overflow_groups=counters["activation_overflow_occurrences"],
                      weight_overflow_groups=weight_overflows)
        if self.trace:
            self.trace.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        return record

    def export(self, path, workload=None):
        if self.trace:
            self.trace.flush()
        phases = {}
        for phase, counts in self.phases.items():
            phases[phase] = dict(counts=counts, compute_seconds={
                name: count / self.config.frequency_hz for name, count in counts["cycles"].items()},
                configured_word_coverage_complete=not (
                    counts["activation_overflow_occurrences"] or counts["weight_overflow_occurrences"]))
        doc = dict(schema_version=1, backend="multcim_ebb_bounds",
                   paper_doi="10.1109/JSSC.2023.3305663", config=asdict(self.config),
                   workload=workload or {}, phases=phases, layers=self.layers,
                   steps=list(self.steps.values()),
                   trace_file=self.trace_path.name if self.trace_path else None,
                   scope="conditional GEMM compute bounds; not measured FP8 MulTCIM or E2E latency",
                   assumptions=[
                       "FP8 alignment is a lossless software extension; the paper supports INT8/INT16.",
                       "Ideal group equalization is a lower bound; routing constraints are not reconstructed.",
                       "Leading-zero-only execution is an upper bound under the same fixed macro layout.",
                       "INT16 weights use two 8b bank passes and half resident K capacity (analytical assumption).",
                       "Negative two's-complement inputs retain the full word; sign-magnitude is optional.",
                       "A overflow extends serial widths; B overflow invalidates the assumed weight storage.",
                       "No exponent/scale/sign conversion overhead, output accumulation, IO, or CIM writes included.",
                       "Traffic is a compulsory minimum; tiling reloads and scale metadata are excluded.",
                   ], remaining_system_inputs=["offchip_bandwidth", "cim_write_bandwidth_or_time",
                                              "reload_schedule", "compute_transfer_overlap",
                                              "LLMCompass_non_gemm_latencies", "FP8_conversion_cost",
                                              "exact_EBB_controller_schedule"])
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(destination)
        return doc
