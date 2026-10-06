"""Offline short-profile extrapolation; recompute target GEMM/IO, no Torch/model."""
import argparse
import csv
import json
from pathlib import Path

from .estimate_paper_latency import SCENARIOS, number, validate_other
from .profile_bridge import build_profile, validate_profile, workload_from_profile


def estimate(profile, prefill_length=2048, decode_steps=256, *, frequency_hz=None,
             external_bandwidth=None, other=None, traffic_mode="weights_kv",
             paper_speedup=None, allow_incomplete_word_coverage=False):
    source = validate_profile(profile)
    if decode_steps and not source.decode_steps:
        raise ValueError("Decode estimation requires collected decode coefficients")
    if profile.get("configured_word_coverage_complete") is False and not allow_incomplete_word_coverage:
        raise ValueError("EBB word coverage incomplete; explicitly allow this unsupported-width scenario to extrapolate")
    work = workload_from_profile(profile["source_workload"], prefill_length, decode_steps)
    if other is not None and other.get("workload") is None:
        raise ValueError("Remaining operator latency must specify the target workload lengths")
    validate_other(other, work)
    if traffic_mode not in ("weights_kv", "materialized_once"):
        raise ValueError("Unknown traffic mode")
    frequency = profile["hardware"]["frequency_hz"] if frequency_hz is None else frequency_hz
    # Do not conflate Bitlet's two DMA ports or local SRAM with one DRAM channel.
    bw = profile["hardware"].get("dram_bytes_per_second") if external_bandwidth is None else external_bandwidth
    number(frequency, "target frequency")
    number(bw, "external bandwidth", nullable=True)
    if paper_speedup is not None:
        number(paper_speedup, "paper speedup")
    cases = ["dense", "online_mapped"]
    if profile["architecture"] == "ebb":
        cases.append("online_leading_zero")
    if paper_speedup is not None:
        cases.append("paper_speedup")
    rows = {}
    for case in cases:
        phases, steps = {}, {}
        for op in work.operations():
            phase, step = op["phase"], op["step"]
            transfer = sum(op[k] for k in ("weight_bytes", "kv_read_bytes", "kv_write_bytes"))
            if traffic_mode == "materialized_once":
                transfer += op["activation_io_bytes"]
            item = dict(compute_seconds=0., modeled_transfer_bytes=transfer,
                        transfer_seconds=None if bw is None else transfer/bw,
                        gemm_and_io_seconds={s: None if bw is None else 0. for s in SCENARIOS})
            for layer in range(work.layers):
                coefficient = profile["coefficients"][f"{phase}:{op['operator']}:{layer}"]["cycles_per_mac"]
                cycles = coefficient["dense"]/paper_speedup if case == "paper_speedup" else coefficient[case]
                compute = op["flops"]/(2*work.layers)*cycles/frequency
                item["compute_seconds"] += compute
                if bw is not None:
                    io = transfer/work.layers/bw
                    item["gemm_and_io_seconds"]["full_overlap"] += max(compute, io)
                    item["gemm_and_io_seconds"]["no_overlap"] += compute+io
            bucket = phases.setdefault(phase, dict(compute_seconds=0., transfer_seconds=None if bw is None else 0.,
                modeled_transfer_bytes=0, gemm_and_io_seconds={s: None if bw is None else 0. for s in SCENARIOS}, operators={}))
            step_bucket = steps.setdefault(step, dict(phase=phase, step=step, compute_seconds=0.,
                gemm_and_io_seconds={s: None if bw is None else 0. for s in SCENARIOS}))
            operator = bucket["operators"].setdefault(op["operator"], dict(compute_seconds=0., modeled_transfer_bytes=0))
            operator["compute_seconds"] += item["compute_seconds"]
            operator["modeled_transfer_bytes"] += transfer
            bucket["compute_seconds"] += item["compute_seconds"]
            bucket["modeled_transfer_bytes"] += transfer
            step_bucket["compute_seconds"] += item["compute_seconds"]
            if bw is not None:
                bucket["transfer_seconds"] += item["transfer_seconds"]
                for schedule in SCENARIOS:
                    bucket["gemm_and_io_seconds"][schedule] += item["gemm_and_io_seconds"][schedule]
                    step_bucket["gemm_and_io_seconds"][schedule] += item["gemm_and_io_seconds"][schedule]
        for step, bucket in steps.items():
            rest = None if other is None else other["prefill_seconds"] if step == 0 else other["decode_step_seconds"][step-1]
            bucket["conditional_e2e_seconds"] = {
                s: None if rest is None or bw is None else bucket["gemm_and_io_seconds"][s]+rest for s in SCENARIOS}
        total = {s: None if bw is None else sum(b["gemm_and_io_seconds"][s] for b in phases.values()) for s in SCENARIOS}
        e2e = {s: None if other is None or bw is None else total[s]+other["prefill_seconds"]+sum(other["decode_step_seconds"])
               for s in SCENARIOS}
        rows[case] = dict(phases=phases, steps=list(steps.values()), gemm_and_io_seconds=total,
                          conditional_e2e_seconds=e2e,
                          average_decode_compute_seconds=phases["decode"]["compute_seconds"]/decode_steps if decode_steps else None)
    return dict(schema_version=1, method="short_profile_calibrated_workload_extrapolation",
                architecture=profile["architecture"], source_workload=profile["source_workload"],
                target_workload=dict(prefill_length=prefill_length, decode_steps=decode_steps,
                                     decode_definition="N decode forwards after prefill, including the last forward"),
                frequency_hz=frequency, external_bandwidth_bytes_per_second=bw,
                traffic_mode=traffic_mode, paper_speedup=paper_speedup, cases=rows,
                source_word_coverage_complete=profile.get("configured_word_coverage_complete"),
                assumptions=profile["extrapolation_assumptions"]+[
                    "Target shapes/work and GQA physical KV traffic are recalculated; no whole-run latency multiplier is used.",
                    "weights_kv transfers each layer weight once per forward and physical KV once; materialized_once adds one activation/output IO per GEMM.",
                    "This minimum-traffic model omits SRAM-capacity reloads, CIM refill replication, metadata, bank conflicts and spills.",
                    "Full/no overlap are conditional per-layer/operator schedules, not guaranteed physical bounds.",
                    "Remaining LM head, normalization/Softmax/RoPE/SiLU, format/control and unmodeled memory costs must be supplied for a conditional E2E scenario.",
                    "A paper speedup, if explicitly supplied, divides the online-calibrated dense cycles once; it does not multiply the measured sparse speedup.",
                ], missing_inputs=([] if bw is not None else ["external bandwidth"])+
                ([] if other is not None else ["target remaining-operator costs including LM head"]))


def overrides(items, names, label, multiplier=1):
    result = {}
    for item in items:
        name, value = item.split("=", 1)
        if name not in names or name in result:
            raise ValueError(f"Unknown or duplicate architecture in {label}: {name}")
        value = float(value)*multiplier
        number(value, label)
        result[name] = value
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile-dir", type=Path, required=True)
    p.add_argument("--architectures", nargs="+", choices=("bitlet", "bitwave", "slimllama", "ebb"))
    p.add_argument("--prefill-length", type=int, default=2048)
    p.add_argument("--decode-steps", type=int, default=256)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frequency-mhz", action="append", default=[], help="architecture=MHz; resources stay fixed")
    p.add_argument("--external-bandwidth-gbps", action="append", default=[], help="architecture=decimal GB/s")
    p.add_argument("--shared-external-bandwidth-gbps", type=float)
    p.add_argument("--paper-speedup", action="append", default=[], help="architecture=ratio; optional dense calibration case")
    p.add_argument("--other-latency-json", action="append", default=[], help="architecture=path; target workload costs")
    p.add_argument("--traffic-mode", choices=("weights_kv", "materialized_once"), default="weights_kv")
    p.add_argument("--allow-incomplete-word-coverage", action="store_true")
    args = p.parse_args(argv)
    names = args.architectures or [n for n in ("bitlet", "bitwave", "slimllama", "ebb")
                                  if (args.profile_dir/f"{n}_summary.json").exists()]
    if not names or len(names) != len(set(names)):
        raise ValueError("Select one or more distinct architectures with completed profiles")
    frequency = overrides(args.frequency_mhz, names, "frequency", 1e6)
    bandwidth = overrides(args.external_bandwidth_gbps, names, "bandwidth", 1e9)
    speedups = overrides(args.paper_speedup, names, "paper speedup")
    if args.shared_external_bandwidth_gbps is not None:
        number(args.shared_external_bandwidth_gbps, "shared bandwidth")
        bandwidth = dict({n: args.shared_external_bandwidth_gbps*1e9 for n in names}, **bandwidth)
    others = {}
    for item in args.other_latency_json:
        name, path = item.split("=", 1)
        if name not in names or name in others:
            raise ValueError("Unknown/duplicate remaining-operator architecture")
        others[name] = json.loads(Path(path).read_text())
    results = {}
    for name in names:
        # Rebuild from authoritative summary, accepting existing completed runs too.
        profile = build_profile(json.loads((args.profile_dir/f"{name}_summary.json").read_text()))
        if profile["architecture"] != name:
            raise ValueError("Summary backend disagrees with filename")
        results[name] = estimate(profile, args.prefill_length, args.decode_steps,
            frequency_hz=frequency.get(name), external_bandwidth=bandwidth.get(name), other=others.get(name),
            traffic_mode=args.traffic_mode, paper_speedup=speedups.get(name),
            allow_incomplete_word_coverage=args.allow_incomplete_word_coverage)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"profile_latency_summary.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    with (args.output_dir/"profile_latency_comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["architecture", "case", "prefill_compute_s", "decode_compute_s", "gemm_io_overlap_s", "conditional_e2e_overlap_s"])
        for name, result in results.items():
            for case, row in result["cases"].items():
                writer.writerow([name, case, row["phases"]["prefill"]["compute_seconds"],
                    row["phases"].get("decode", {}).get("compute_seconds", 0), row["gemm_and_io_seconds"]["full_overlap"],
                    row["conditional_e2e_seconds"]["full_overlap"]])
            row = result["cases"]["online_mapped"]
            print(f"{name}: conditional compute={sum(b['compute_seconds'] for b in row['phases'].values()):.6f} s; "
                  f"GEMM/IO={row['gemm_and_io_seconds']['full_overlap']}; E2E={row['conditional_e2e_seconds']['full_overlap']}")


if __name__ == "__main__":
    main()
