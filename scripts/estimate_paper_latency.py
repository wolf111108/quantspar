"""Offline Fig.16 capacity/traffic scenarios; Python standard library only.

No checkpoint, quantization, sparse TOPS or measured GPU wall time is used.
Missing operating points stay null. Native precision references are explicit
capacity proxies; this is not a native FP8 timing implementation.
"""
import argparse
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/qwen14b_fig16_capacity.json"
SCENARIOS = ("full_overlap", "no_overlap")


def number(value, label, positive=True, nullable=False):
    if value is None and nullable:
        return
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")


@dataclass(frozen=True)
class Workload:
    prefill_length: int = 2048
    decode_steps: int = 256
    layers: int = 48
    hidden_size: int = 5120
    intermediate_size: int = 13824
    query_heads: int = 40
    kv_heads: int = 8
    head_dim: int = 128
    activation_bytes: int = 1
    kv_bytes: int = 1
    weight_bits: int = 4
    output_bytes: int = 2

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "decode_steps" else 1):
                raise ValueError(f"{name} has an invalid integer value")
        if self.query_heads * self.head_dim != self.hidden_size or self.query_heads % self.kv_heads:
            raise ValueError("Inconsistent GQA/head dimensions")

    def linear_shapes(self):
        d, f, kv = self.hidden_size, self.intermediate_size, self.kv_heads*self.head_dim
        return {"q": (d, d), "k": (d, kv), "v": (d, kv), "o": (d, d),
                "gate": (d, f), "up": (d, f), "down": (f, d)}

    def operations(self):
        """Group equal-shape calls across layers; this preserves sum_l max(t_l).

        Prefill attention computes full SxS, including masked entries. Decode
        attention includes the new token; only historical physical KV is read
        from external memory, and new KV is written once.
        """
        for step in range(self.decode_steps + 1):
            phase = "prefill" if step == 0 else "decode"
            m = self.prefill_length if step == 0 else 1
            context = self.prefill_length if step == 0 else self.prefill_length + step
            before = context if step == 0 else context - 1
            for op, (k, n) in self.linear_shapes().items():
                yield dict(phase=phase, step=step, operator=op,
                           flops=2*self.layers*m*k*n,
                           weight_bytes=self.layers*((k*n*self.weight_bits+7)//8),
                           kv_read_bytes=0, kv_write_bytes=0,
                           activation_io_bytes=self.layers*(m*k*self.activation_bytes+m*n*self.output_bytes))
            kv_read = self.layers*self.kv_heads*before*self.head_dim*self.kv_bytes
            kv_write = self.layers*self.kv_heads*m*self.head_dim*self.kv_bytes
            for op in ("qk", "pv"):
                # QK output is FP16; PV reads a quantized FP8 attention matrix.
                a_elements = self.query_heads*m*(self.head_dim if op == "qk" else context)
                c_elements = self.query_heads*m*(context if op == "qk" else self.head_dim)
                yield dict(phase=phase, step=step, operator=op,
                           flops=2*self.layers*self.query_heads*m*context*self.head_dim,
                           weight_bytes=0, kv_read_bytes=kv_read, kv_write_bytes=kv_write,
                           activation_io_bytes=self.layers*(a_elements*self.activation_bytes+c_elements*self.output_bytes))


def validate_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != 1 or config.get("model") != "Qwen2.5-14B":
        raise ValueError("Expected a Qwen2.5-14B schema_version=1 capacity config")
    if not isinstance(config.get("architectures"), dict) or not config["architectures"]:
        raise ValueError("architectures must be a nonempty mapping")
    for key, arch in config["architectures"].items():
        if not isinstance(arch, dict):
            raise ValueError(f"{key} must be an architecture mapping")
        for field in ("speedup", "pe_count", "products_per_pe_per_cycle", "dense_cycles_per_mac",
                      "prefill_utilization", "decode_utilization"):
            number(arch.get(field), f"{key}.{field}")
        for field in ("frequency_hz", "external_bandwidth_bytes_per_second"):
            number(arch.get(field), f"{key}.{field}", nullable=True)
        for field in ("pe_count", "products_per_pe_per_cycle", "dense_cycles_per_mac"):
            if not isinstance(arch[field], int):
                raise ValueError(f"{key}.{field} must be an integer")
        for phase in ("prefill", "decode"):
            if arch[f"{phase}_utilization"] > 1:
                raise ValueError("utilization must be in (0,1]")


def validate_other(other, workload):
    if other is None:
        return
    if not isinstance(other, dict) or other.get("schema_version") != 1 or other.get("includes_lm_head") is not True:
        raise ValueError("Other latency requires schema_version=1 and includes_lm_head=true")
    steps = other.get("decode_step_seconds")
    if not isinstance(steps, list) or len(steps) != workload.decode_steps:
        raise ValueError("Other latency must cover every requested decode forward")
    for value in [other.get("prefill_seconds"), *steps]:
        number(value, "other latency", positive=False)
    provenance = other.get("workload")
    if provenance is not None:
        if (not isinstance(provenance, dict) or
                provenance.get("prefill_length") != workload.prefill_length or
                provenance.get("decode_steps") != workload.decode_steps):
            raise ValueError("Other latency workload does not match the requested lengths")


def estimate(config, workload, other=None, traffic_mode="weights_kv"):
    validate_config(config)
    validate_other(other, workload)
    if traffic_mode not in ("weights_kv", "materialized_once"):
        raise ValueError("Unknown traffic mode")
    records = list(workload.operations())
    counts = {phase: dict(flops=0, weight_bytes=0, kv_read_bytes=0, kv_write_bytes=0,
                         activation_io_bytes=0, modeled_transfer_bytes=0, operators={})
              for phase in ("prefill", "decode")}
    for record in records:
        transfer = sum(record[k] for k in ("weight_bytes", "kv_read_bytes", "kv_write_bytes"))
        if traffic_mode == "materialized_once":
            transfer += record["activation_io_bytes"]
        record["modeled_transfer_bytes"] = transfer
        phase = counts[record["phase"]]
        op = phase["operators"].setdefault(record["operator"], dict(flops=0, transfer_bytes=0))
        op["flops"] += record["flops"]
        op["transfer_bytes"] += transfer
        for key in phase:
            if key != "operators":
                phase[key] += record[key]
    params = workload.layers*sum(k*n for k, n in workload.linear_shapes().values())
    result = dict(schema_version=1, method="fig16_dense_capacity_and_traffic",
                  scope="conditional capacity/traffic scenarios, not measured or cycle-accurate native FP8 E2E",
                  source=config["source"], source_context_length=config["source_context_length"],
                  workload=dict(model=config["model"], batch_size=1, **asdict(workload),
                                linear_parameters=params, decode_definition=f"{workload.decode_steps} decode forwards after prefill, including the last forward"),
                  traffic_mode=traffic_mode, workload_counts=counts, supplied_other_latency=other,
                  other_workload_verified=(other is not None and other.get("workload") is not None),
                  assumptions=[
                      "One MAC = two operations. Fig.16 ratios multiply dense capacity once; no reported sparse TOPS are used.",
                      "The Qwen14B average Fig.16 speedup is extrapolated from 8192 tokens to every GEMM in both phases, including decode M=1.",
                      "Asyn's Fig.16 ratio 4.01 matches Fig.14 after-I/O prefill throughput / dense throughput; using it as compute capacity is a calibration proxy and can retain some source I/O loss.",
                      "Utilization=1 is an optimistic capacity assumption, not another measured utilization factor. Per-phase overrides are explicit sensitivity scenarios.",
                      "Packed W4 is external transport only. KV is FP8 and GQA traffic uses 8 physical KV heads, while compute uses 40 query heads.",
                      "Each linear weight matrix is read once per prefill or decode forward. Prefill KV is written and read once; decode reads history and writes new KV.",
                      "weights_kv omits other activation IO; materialized_once adds one activation read/output write per GEMM, without tiling/refill/metadata or full spill simulation.",
                      "Buffer sizes and on-chip bandwidths are documented but do not establish residency or an internal traffic model; repeated refill and conversion costs remain missing.",
                      "Full/no overlap are conditional schedules under the same specified traffic, not guaranteed physical bounds.",
                      "Remaining LM head, Softmax, RMSNorm, RoPE, SiLU and conversion/control costs must be supplied separately; absent costs never become zero E2E.",
                  ], architectures={})
    for key, arch in config["architectures"].items():
        missing = [field for field in ("frequency_hz", "external_bandwidth_bytes_per_second") if arch[field] is None]
        dense = None if arch["frequency_hz"] is None else (
            2*arch["pe_count"]*arch["products_per_pe_per_cycle"]*arch["frequency_hz"]/arch["dense_cycles_per_mac"]/1e12)
        row = dict(config=deepcopy(arch), dense_tops=dense, missing_inputs=missing, cases={})
        for case, speedup in (("dense", 1), ("fig16", arch["speedup"])):
            phases = {}
            decode = []
            for step in range(workload.decode_steps+1):
                phase = "prefill" if step == 0 else "decode"
                rate = None if dense is None else dense*1e12*speedup*arch[f"{phase}_utilization"]
                operations = records[step*9:(step+1)*9]
                compute = None if rate is None else sum(r["flops"]/rate for r in operations)
                bw = arch["external_bandwidth_bytes_per_second"]
                io = None if bw is None else sum(r["modeled_transfer_bytes"]/bw for r in operations)
                times = {name: None for name in SCENARIOS}
                if compute is not None and io is not None:
                    times = dict(
                        full_overlap=sum(max(r["flops"]/rate, r["modeled_transfer_bytes"]/bw) for r in operations),
                        no_overlap=compute+io)
                rest = None if other is None else (other["prefill_seconds"] if step == 0 else other["decode_step_seconds"][step-1])
                e2e = {name: None if seconds is None or rest is None else seconds+rest for name, seconds in times.items()}
                item = dict(compute_seconds=compute, transfer_seconds=io,
                            gemm_and_io_seconds=times, conditional_e2e_seconds=e2e)
                bucket = phases.setdefault(phase, dict(compute_seconds=compute, transfer_seconds=io,
                                                       gemm_and_io_seconds={name: 0.0 if times[name] is not None else None for name in SCENARIOS},
                                                       conditional_e2e_seconds={name: 0.0 if e2e[name] is not None else None for name in SCENARIOS}))
                if step > 0 and len(decode) > 0:
                    for field, value in (("compute_seconds", compute), ("transfer_seconds", io)):
                        if value is not None:
                            bucket[field] += value
                for field in ("gemm_and_io_seconds", "conditional_e2e_seconds"):
                    for name in SCENARIOS:
                        if item[field][name] is not None:
                            bucket[field][name] += item[field][name]
                if step > 0:
                    decode.append(dict(step=step, cache_length_before=workload.prefill_length+step-1,
                                       cache_length_after=workload.prefill_length+step, **item))
            totals = {}
            for field in ("gemm_and_io_seconds", "conditional_e2e_seconds"):
                totals[field] = {name: (sum(p[field][name] for p in phases.values())
                                      if all(p[field][name] is not None for p in phases.values()) else None)
                                 for name in SCENARIOS}
            totals["average_decode_gemm_and_io_seconds"] = {
                name: (phases["decode"]["gemm_and_io_seconds"][name]/workload.decode_steps
                       if workload.decode_steps and phases["decode"]["gemm_and_io_seconds"][name] is not None else None)
                for name in SCENARIOS}
            totals["average_decode_conditional_e2e_seconds"] = {
                name: (phases["decode"]["conditional_e2e_seconds"][name]/workload.decode_steps
                       if workload.decode_steps and phases["decode"]["conditional_e2e_seconds"][name] is not None else None)
                for name in SCENARIOS}
            row["cases"][case] = dict(speedup=speedup, phases=phases, decode_steps=decode, **totals)
        result["architectures"][key] = row
    return result


def apply_overrides(config, args):
    config = deepcopy(config)
    config["overrides"] = []
    archs = config["architectures"]
    if args.shared_external_bandwidth_gbps is not None:
        number(args.shared_external_bandwidth_gbps, "shared bandwidth")
        for arch in archs.values():
            arch["external_bandwidth_bytes_per_second"] = args.shared_external_bandwidth_gbps*1e9
        config["overrides"].append(f"Shared external bandwidth={args.shared_external_bandwidth_gbps} GB/s (assumption, overrides native values)")
    for option, field, multiplier in (
            ("external_bandwidth_gbps", "external_bandwidth_bytes_per_second", 1e9),
            ("frequency_mhz", "frequency_hz", 1e6),
            ("prefill_utilization", "prefill_utilization", 1),
            ("decode_utilization", "decode_utilization", 1)):
        for text in getattr(args, option):
            try:
                key, raw = text.split("=", 1)
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"--{option.replace('_', '-')} expects architecture=value") from exc
            if key not in archs:
                raise ValueError(f"Unknown architecture {key}; Bitlet/Slim-Llama have no Fig.16 ratios")
            number(value, field)
            archs[key][field] = value*multiplier
            config["overrides"].append(f"{key}.{field}={value*multiplier} (explicit assumption)")
    validate_config(config)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefill-length", type=int, default=2048)
    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument("--traffic-mode", choices=("weights_kv", "materialized_once"), default="weights_kv")
    parser.add_argument("--other-latency-json", type=Path)
    parser.add_argument("--shared-external-bandwidth-gbps", type=float)
    for name in ("external-bandwidth-gbps", "frequency-mhz", "prefill-utilization", "decode-utilization"):
        parser.add_argument(f"--{name}", action="append", default=[], metavar="ARCH=VALUE")
    args = parser.parse_args(argv)
    workload = Workload(prefill_length=args.prefill_length, decode_steps=args.decode_steps)
    config = apply_overrides(json.loads(args.config.read_text()), args)
    other = None if args.other_latency_json is None else json.loads(args.other_latency_json.read_text())
    result = estimate(config, workload, other, args.traffic_mode)
    result["overrides"] = config["overrides"]
    try:
        result["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result["source_commit"] = None
    # Refuse to overwrite inputs or results; all computation finishes first.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"paper_latency_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    with (args.output_dir/"paper_latency_comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["architecture", "dense_tops", "fig16_speedup", "prefill_compute_s", "decode_compute_s",
                         "gemm_io_full_overlap_s", "gemm_io_no_overlap_s", "e2e_full_overlap_s", "e2e_no_overlap_s",
                         "decode_gemm_io_full_overlap_s_per_forward", "missing_inputs"])
        for key, row in result["architectures"].items():
            case = row["cases"]["fig16"]
            phases = case["phases"]
            writer.writerow([key, row["dense_tops"], row["config"]["speedup"],
                             phases["prefill"]["compute_seconds"], phases.get("decode", {}).get("compute_seconds"),
                             *case["gemm_and_io_seconds"].values(), *case["conditional_e2e_seconds"].values(),
                             case["average_decode_gemm_and_io_seconds"]["full_overlap"], ";".join(row["missing_inputs"])])
            latency = case["gemm_and_io_seconds"]["full_overlap"]
            rendered = "unavailable: "+", ".join(row["missing_inputs"]) if latency is None else f"{latency:.6f} s GEMM/IO (conditional)"
            e2e = case["conditional_e2e_seconds"]["full_overlap"]
            print(f"{key}: {rendered}; E2E="+("unavailable without complete inputs" if e2e is None else f"{e2e:.6f} s (conditional)"))
    return result


if __name__ == "__main__":
    main()
