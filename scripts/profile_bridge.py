"""Portable measured-cycle coefficients; standard library, no model dependency."""
from copy import deepcopy
from dataclasses import asdict

from .estimate_paper_latency import Workload, number


OPERATORS = dict(q_proj="q", k_proj="k", v_proj="v", o_proj="o",
                 gate_proj="gate", up_proj="up", down_proj="down",
                 qk_matmul="qk", pv_matmul="pv")
DIMENSIONS = dict(layers="layers", hidden_size="d_model", intermediate_size="ffn_dim",
                  query_heads="q_heads", kv_heads="kv_heads", head_dim="head_dim")


def workload_from_profile(workload, prefill_length=None, decode_steps=None):
    if workload.get("batch_size") != 1:
        raise ValueError("The offline bridge requires batch_size=1")
    return Workload(**{key: workload[value] for key, value in DIMENSIONS.items()},
                    prefill_length=workload["prefill_length"] if prefill_length is None else prefill_length,
                    decode_steps=workload["decode_steps"] if decode_steps is None else decode_steps)


def build_profile(summary):
    """Export one coefficient per actual layer/operator/phase; never phase averages."""
    workload = summary["workload"]
    if (workload.get("status") != "complete" or
            workload.get("completed_decode_steps") != workload.get("decode_steps")):
        raise ValueError("Only complete profiling runs can calibrate the offline model")
    work = workload_from_profile(workload)
    backend = summary["backend"]
    ebb = backend == "multcim_ebb_bounds"
    name = "ebb" if ebb else backend
    if name not in ("ebb", "bitlet", "bitwave", "slimllama"):
        raise ValueError("Unsupported profile backend")
    number(summary["config"]["frequency_hz"], "source frequency")
    macs = {}
    for item in work.operations():
        key = (item["phase"], item["operator"])
        macs[key] = macs.get(key, 0) + item["flops"] // (2*work.layers)
    coefficients = {}
    expected = {(phase, op, layer) for phase, op in macs
                for layer in range(work.layers)}
    for key, raw in summary["layers"].items():
        phase, suffix = key.split(":", 1)
        operator, layer = suffix.rsplit("_", 1)
        layer = int(layer)
        op = OPERATORS[operator]
        if (phase, op, layer) not in expected:
            raise ValueError("Unexpected layer/operator/phase in profile")
        expected.remove((phase, op, layer))
        counts = raw["counts"] if ebb else raw
        calls = 1 if phase == "prefill" else work.decode_steps
        if counts["calls"] != calls:
            raise ValueError("Incomplete per-layer call coverage")
        measured_macs = macs[phase, op]
        if not ebb and counts["dense_equivalent_operations"] != 2*measured_macs:
            raise ValueError("Profile operation counts disagree with model dimensions")
        cycles = counts["cycles"]
        # EBB has two conditional mappings, rather than one measured sparse count.
        selected = ({"dense": "dense", "online_mapped": "ideal_balanced",
                     "online_leading_zero": "leading_zero"} if ebb else
                    {"dense": "dense_tiles", "online_mapped": "mapped_tiles"})
        normalized = {}
        for case, field in selected.items():
            number(cycles[field], "cycles", positive=False)
            normalized[case] = cycles[field]/measured_macs
        coefficients[f"{phase}:{op}:{layer}"] = dict(
            source_calls=calls, source_macs=measured_macs,
            cycles_per_mac=normalized, source_cycles={case: cycles[field] for case, field in selected.items()},
            source_speedup=(cycles[selected["dense"]]/cycles[selected["online_mapped"]]
                            if cycles[selected["online_mapped"]] else None),
            sampled_calls=counts.get("sampled_calls", 0),
            observations=deepcopy(counts.get("observed", counts.get("activation", {}))),
            selected_dataflows=counts.get("selected_dataflows", {}))
    if expected:
        raise ValueError("Missing per-layer/operator/phase coverage")
    word_coverage = all(p["configured_word_coverage_complete"] for p in summary["phases"].values()) if ebb else None
    return dict(schema_version=1, method="online_cycles_per_mac", architecture=name,
                source_workload=deepcopy(workload), source_dimensions=asdict(work),
                hardware=deepcopy(summary["config"]), coefficients=coefficients,
                configured_word_coverage_complete=word_coverage,
                source_assumptions=deepcopy(summary.get("assumptions", [])),
                transfer_policy="Recompute target packed W4 / physical FP8 KV; do not scale source traffic.",
                extrapolation_assumptions=[
                    "Per-layer/operator/phase cycles per MAC, FP8 width expansion, tiling utilization and sparse savings remain equal to the short run.",
                    "Decode coefficients are MAC-weighted across collected steps; they are not long-context measurements.",
                    "A different target shape can change tile tails, selected dataflow, utilization and SRAM refill; none is guaranteed invariant.",
                    "Sample histograms describe observed waves; do not interpret them as whole-tensor element sparsity.",
                    "Compute coefficients preserve source hardware resources. Only frequency may be rescaled; changing PE geometry requires a new profile or mapping model.",
                ])


def validate_profile(profile):
    if profile.get("schema_version") != 1 or profile.get("method") != "online_cycles_per_mac":
        raise ValueError("Expected an online_cycles_per_mac profile")
    work = workload_from_profile(profile["source_workload"])
    number(profile["hardware"]["frequency_hz"], "profile frequency")
    expected = {f"{phase}:{op}:{layer}" for phase in ("prefill", "decode")
                if phase == "prefill" or work.decode_steps
                for op in OPERATORS.values() for layer in range(work.layers)}
    if set(profile["coefficients"]) != expected:
        raise ValueError("Missing or unexpected profile coefficients")
    for row in profile["coefficients"].values():
        for case in ("dense", "online_mapped"):
            number(row["cycles_per_mac"].get(case), "coefficient", positive=False)
        for value in row["cycles_per_mac"].values():
            number(value, "coefficient", positive=False)
    return work
