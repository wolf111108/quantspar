#!/usr/bin/env python3
"""Isolated, short FineWeb PPL ablations for Qwen2.5-7B Linear quantization."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "config/diagnostics/qwen7b_linear_scale_fast.yaml"
ENTRY = ROOT / "0103_quant_pipeline_main.py"
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CASES = {
    "bf16_raw": ("scalar", "bf16", "bf16", "none"),
    "scalar_awo_fp8": ("scalar", "e4m3", "e4m3", "e4m3"),
    "channel_awo_fp8": ("output_channel", "e4m3", "e4m3", "e4m3"),
    "scalar_aw_fp8_o_raw": ("scalar", "e4m3", "e4m3", "none"),
    "channel_aw_fp8_o_raw": ("output_channel", "e4m3", "e4m3", "none"),
    "channel_a_raw_w_fp8": ("output_channel", "bf16", "e4m3", "none"),
    "channel_a_fp8_w_raw": ("output_channel", "e4m3", "bf16", "none"),
}
DEFAULT_CASES = tuple(CASES)[:5]
FIELDS = ("case", "status", "ppl", "delta_vs_scalar", "attempt", "report", "log", "error")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local Qwen2.5-7B checkpoint")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/qwen7b_linear_scale_fast")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=DEFAULT_CASES)
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--calibration-seq-length", type=int, default=1024)
    parser.add_argument("--eval-tokens", type=int, default=8192)
    parser.add_argument("--eval-seq-length", type=int, default=2048)
    parser.add_argument("--min-doc-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true", help="Reuse only matching successful reports")
    parser.add_argument("--dry-run", action="store_true", help="Show cases without creating files or loading models")
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases):
        parser.error("--cases must not contain duplicates")
    for name in ("calibration_samples", "calibration_seq_length", "eval_tokens", "eval_seq_length", "min_doc_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def case_config(template, name, args, scale_dir):
    config = json.loads(json.dumps(template))
    granularity, a_bit, w_bit, o_bit = CASES[name]
    q = config["quantization"]
    q["weight_scale_granularity"] = granularity
    q["scale_dir"] = str(scale_dir)
    q["calibration_policy"]["default"] = "recalibrate"
    for layer in LINEARS:
        q[layer].update(a_bit=a_bit, w_bit=w_bit, o_bit=o_bit)
    config["calibration"].update(num_samples=args.calibration_samples, seq_length=args.calibration_seq_length)
    config["evaluation"].update(seq_length=args.eval_seq_length, max_eval_tokens=args.eval_tokens,
                                min_doc_tokens=args.min_doc_tokens)
    for dataset in config["evaluation"]["datasets"]:
        dataset.update(seq_length=args.eval_seq_length, max_eval_tokens=args.eval_tokens,
                       min_doc_tokens=args.min_doc_tokens)
    return config


def fingerprint(config, args):
    # Exclude output path so a matching attempt can be resumed in its own directory.
    canonical = json.loads(json.dumps(config))
    canonical["quantization"].pop("scale_dir")
    data = {"config": canonical, "model_path": str(Path(args.model_path).expanduser().resolve()),
            "device": args.device, "entry_sha256": hashlib.sha256(ENTRY.read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def existing_result(attempt, name, expected, args):
    report = attempt / "results" / f"{name}_evaluation.json"
    if not report.is_file():
        return None
    try:
        doc = json.loads(report.read_text())
        metric = doc["ppl"]["fineweb"]
        value = float(metric["perplexity"])
        if (doc["run_name"] != name or doc["eval_flow"] != "ppl" or
                doc["model_path"] != args.model_path or
                doc["config"]["experiment"]["fingerprint"] != expected or
                metric["status"] != "ok" or not math.isfinite(value) or value <= 0):
            return None
        return {"case": name, "status": "reused", "ppl": value, "attempt": str(attempt),
                "report": str(report), "log": str(attempt / "run.log"), "error": ""}
    except (KeyError, TypeError, ValueError, OSError):
        return None


def next_attempt(case_dir):
    numbers = [int(p.name[8:]) for p in case_dir.glob("attempt_[0-9][0-9][0-9][0-9]") if p.is_dir()]
    return case_dir / f"attempt_{max(numbers, default=0) + 1:04d}"


def command(args, name, attempt):
    cmd = [sys.executable, "-u", str(ENTRY), "--config", str(attempt / "config.yaml"),
           "--model-path", args.model_path, "--device", args.device, "--eval-flow", "ppl",
           "--ppl-only-no-stats", "--no-unit-sparsity", "--run-name", name,
           "--results-dir", str(attempt / "results")]
    if name == "bf16_raw":
        cmd.append("--fp-baseline")
    return cmd


def save_summary(rows, root):
    scalar = next((r["ppl"] for r in rows if r["case"] == "scalar_awo_fp8" and r["ppl"] is not None), None)
    for row in rows:
        row["delta_vs_scalar"] = (row["ppl"] - scalar) if row["ppl"] is not None and scalar is not None else None
    (root / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = arguments()
    template = yaml.safe_load(TEMPLATE.read_text())
    root = args.output_dir.expanduser().resolve()
    if args.dry_run:
        for name in args.cases:
            config = case_config(template, name, args, root / name / "attempt_0001" / "scales")
            print(f"{name}: Linear W scale={config['quantization']['weight_scale_granularity']}, "
                  f"A/W/O={tuple(config['quantization']['q_proj'][key] for key in ('a_bit', 'w_bit', 'o_bit'))}; "
                  f"cal={args.calibration_samples}x{args.calibration_seq_length}, "
                  f"eval={args.eval_tokens} tokens, seq={args.eval_seq_length}")
            print("  " + " ".join(command(args, name, root / name / "attempt_0001")))
        return 0

    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.cases:
        case_dir = root / name
        probe = case_config(template, name, args, case_dir / "scales")
        expected = fingerprint(probe, args)
        reused = None
        if args.resume:
            for attempt in sorted(case_dir.glob("attempt_[0-9][0-9][0-9][0-9]"), reverse=True):
                reused = existing_result(attempt, name, expected, args)
                if reused:
                    break
        if reused:
            row = reused
        else:
            attempt = next_attempt(case_dir)
            attempt.mkdir(parents=True)
            config = case_config(template, name, args, attempt / "scales")
            config["experiment"] = {"case": name, "fingerprint": expected, "ppl_only_no_stats": True}
            (attempt / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            log = attempt / "run.log"
            row = {"case": name, "status": "failed", "ppl": None, "attempt": str(attempt),
                   "report": "", "log": str(log), "error": ""}
            print(f"Running {name} -> {log}", flush=True)
            try:
                with log.open("w") as stream:
                    result = subprocess.run(command(args, name, attempt), cwd=ROOT, stdout=stream,
                                            stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    raise RuntimeError(f"exit code {result.returncode}")
                parsed = existing_result(attempt, name, expected, args)
                if parsed is None:
                    raise RuntimeError("missing, nonfinite, or mismatched PPL report")
                row = parsed
                row["status"] = "ok"
            except (OSError, RuntimeError) as exc:
                row["error"] = str(exc)
                print(f"{name}: {exc}; see {log}", file=sys.stderr)
        rows.append(row)
        save_summary(rows, root)
        print(f"{name}: {row['status']} PPL={row['ppl']}", flush=True)
    print(f"Summary: {root / 'summary.csv'}")
    return 1 if any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
