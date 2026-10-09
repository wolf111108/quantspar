"""Sequential FineWeb reruns for the three non-GPT models and five A/W formats."""

import argparse
import copy
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation_report import COUNT_FIELDS, file_fingerprint, runtime_versions, sum_counts, write_json


MODELS = {
    "opt-1.3b": ("opt_1.3b", "opt_1_3b_path"),
    "opt-6.7b": ("opt_6.7b", "opt_6_7b_path"),
    "qwen2.5-7b": ("qwen2.5_7b", "qwen_7b_path"),
}
FORMATS = ("bf16_bf16", "fp8_fp8", "int8_int8", "int8_int4", "fp8_int4")
PHASES = ("full_forward", "prefill", "decode")
OPERANDS = ("activation", "weight", "Q", "K", "attention_probs", "V")
METRIC = {
    "fp_bit_scope": "explicit_mantissa",
    "int_bit_scope": "twos_complement",
    "aggregation": "sum_zero_bits / sum_counted_bits",
    "mask_generated_zeros": "included",
    "high_precision_sidepath_counted": False,
    "unit_sparsity": False,
}


def outlier_ratio(value):
    ratio = float(value)
    if not math.isfinite(ratio) or not 0 <= ratio < 1:
        raise argparse.ArgumentTypeError("outlier ratio must be finite and in [0, 1)")
    return ratio


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt-1-3b-path", help="Local OPT-1.3B checkpoint directory")
    parser.add_argument("--opt-6-7b-path", help="Local OPT-6.7B checkpoint directory")
    parser.add_argument("--qwen-7b-path", help="Local Qwen2.5-7B checkpoint directory")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--eval-flow", choices=("all", "ppl"), default="all",
                        help="all: PPL + prefill/decode; ppl: PPL and full-forward bit stats")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sparsity_ppl_rerun"))
    parser.add_argument("--outlier-ratio", type=outlier_ratio, default=None,
                        help="Override Linear and QK/PV; an explicit QK/PV ratio takes precedence")
    parser.add_argument("--qk-pv-outlier-ratio", type=outlier_ratio, default=None,
                        help="Override QK/PV separately (default follows --outlier-ratio or template)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip only validated completed jobs with matching provenance")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plans without loading models or writing files")
    return parser.parse_args(argv)


def source_digest():
    paths = {ROOT / "0103_quant_pipeline_main.py", ROOT / "perplexity.py",
             Path(__file__).resolve(), ROOT / "scripts/evaluation_report.py"}
    for directory in ("quant", "others"):
        paths.update((ROOT / directory).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_jobs(args):
    jobs = []
    provenance = {"source_sha256": source_digest(), "runtime": runtime_versions()}
    for model_name, (stem, path_arg) in MODELS.items():
        if model_name not in args.models:
            continue
        checkpoint = getattr(args, path_arg)
        if checkpoint is None:
            if not args.dry_run:
                raise ValueError(f"Missing checkpoint for {model_name}; use --{path_arg.replace('_', '-')}")
            checkpoint = f"/path/to/{model_name}"
        checkpoint = str(Path(checkpoint).expanduser().resolve())
        if not args.dry_run and not Path(checkpoint).is_dir():
            raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
        for format_name in FORMATS:
            if format_name not in args.formats:
                continue
            run_name = f"{stem}_{format_name}"
            path = ROOT / "config/sparsity_ppl_rerun" / f"{run_name}.yaml"
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            if args.outlier_ratio is not None:
                for settings in config["quantization"].values():
                    if isinstance(settings, dict) and "a_bit" in settings:
                        settings["outlier_ratio"] = args.outlier_ratio
            qk_pv_ratio = (args.qk_pv_outlier_ratio if args.qk_pv_outlier_ratio is not None
                           else args.outlier_ratio)
            if qk_pv_ratio is not None:
                for settings in config["quantization"].values():
                    if isinstance(settings, dict) and "A_bit" in settings:
                        settings["outlier_ratio"] = qk_pv_ratio
            fingerprint_input = {
                **provenance, "config": config, "model_path": checkpoint,
                "device": args.device, "eval_flow": args.eval_flow,
            }
            fingerprint = hashlib.sha256(json.dumps(
                fingerprint_input, sort_keys=True, allow_nan=False,
            ).encode()).hexdigest()
            jobs.append({
                "run_name": run_name, "model": model_name, "format": format_name,
                "model_path": checkpoint, "template": str(path), "config": config,
                "fingerprint": fingerprint, "provenance": provenance,
                "device": args.device, "eval_flow": args.eval_flow,
            })
    return jobs


def command_for(job, attempt):
    return [sys.executable, "-u", str(ROOT / "0103_quant_pipeline_main.py"),
            "--config", str(attempt / "config.yaml"),
            "--model-path", job["model_path"], "--device", job["device"],
            "--eval-flow", job["eval_flow"], "--no-unit-sparsity",
            "--run-name", job["run_name"], "--results-dir", str(attempt / "results"),
            "--stats-output-dir", str(attempt / "stats")]


def validate_counts(row):
    for key in COUNT_FIELDS:
        if type(row.get(key)) is not int or row[key] < 0:
            raise ValueError(f"Invalid {key} count")
    if row["zero_bits"] > row["bits"] or row["zero_elements"] > row["elements"]:
        raise ValueError("Zero counts exceed total counts")


def read_report(job, attempt):
    path = attempt / "results" / f"{job['run_name']}_evaluation.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("schema_version") != 1 or doc.get("run_name") != job["run_name"]
            or doc.get("eval_flow") != job["eval_flow"] or doc.get("device") != job["device"]
            or doc.get("model_path") != job["model_path"] or doc.get("metric") != METRIC
            or doc.get("runtime") != job["provenance"]["runtime"]
            or doc.get("config", {}).get("experiment", {}).get("fingerprint") != job["fingerprint"]):
        raise ValueError("Evaluation report provenance or metric does not match this job")

    ppl = doc.get("ppl", {}).get("fineweb")
    if ppl is None:
        raise ValueError("Missing FineWeb PPL result")
    value = ppl.get("perplexity")
    finite_ppl = (ppl.get("status") == "ok" and type(value) in (int, float)
                  and math.isfinite(value) and value > 0)
    if not finite_ppl and (ppl.get("status") not in ("nonfinite", "invalid") or value is not None):
        raise ValueError("Invalid PPL result or status")

    summary = {"ppl": value, "ppl_status": ppl["status"], "ppl_raw_value": ppl.get("raw_value"),
               "ppl_time_seconds": ppl.get("time_seconds"), "report": str(path)}
    operands = []
    flows = {"full_forward": ("full_forward",)}
    if job["eval_flow"] == "all":
        flows["prefill_decode"] = ("prefill", "decode")
    for flow, phases in flows.items():
        snapshot = doc.get("bit_sparsity", {}).get(flow)
        if snapshot is None or snapshot.get("schema_version") != 2 or snapshot.get("bit_scope") != "mantissa":
            raise ValueError(f"Missing or incompatible {flow} bit statistics")
        for extension in ("json", "csv"):
            stats_path = attempt / "stats" / f"{job['run_name']}_{flow}_bit_sparsity.{extension}"
            if not stats_path.is_file() or stats_path.stat().st_size == 0:
                raise ValueError(f"Missing detailed bit statistics: {stats_path}")
            if doc.get("artifacts", {}).get(flow, {}).get(extension) != file_fingerprint(stats_path):
                raise ValueError(f"Detailed bit statistics checksum mismatch: {stats_path}")
        rows = snapshot["phase_operands"]
        for row in rows:
            validate_counts(row)
            if row["phase"] not in phases or row["operand"] not in OPERANDS:
                raise ValueError("Unexpected phase or operand in bit statistics")
        for phase in phases:
            phase_rows = [row for row in rows if row["phase"] == phase]
            summary[f"{phase}_total_bit_zero_ratio"] = sum_counts(phase_rows)["bit_zero_ratio"]
            for operand in OPERANDS:
                selected = [row for row in phase_rows if row["operand"] == operand]
                total = sum_counts(selected)
                if not total["bits"]:
                    raise ValueError(f"Missing {phase}/{operand} bit counts")
                summary[f"{phase}_{operand}_bit_zero_ratio"] = total["bit_zero_ratio"]
            # Retain masked and unmasked groups, with recomputed weighted ratios.
            operands.extend({**row, **sum_counts([row])} for row in phase_rows)
    return {"status": "completed" if finite_ppl else "nonfinite_ppl",
            "summary": summary, "operands": operands}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_results(output, manifest):
    write_json(output / "manifest.json", manifest)
    entries = list(manifest["jobs"].values())
    entries.sort(key=lambda row: (list(MODELS).index(row["model"]), FORMATS.index(row["format"])))
    rows = [{key: value for key, value in entry.items() if key not in ("summary", "operands")}
            | entry.get("summary", {}) for entry in entries]
    write_json(output / "summary.json", {"schema_version": 1, "metric": METRIC, "jobs": rows})
    fields = ["run_name", "model", "format", "status", "ppl", "ppl_status", "ppl_raw_value",
              "eval_flow", "device", "seq_length", "max_eval_tokens", "linear_outlier_ratio",
              "qk_pv_outlier_ratio", "elapsed_seconds", "ppl_time_seconds"]
    fields += [f"{phase}_total_bit_zero_ratio" for phase in PHASES]
    fields += [f"{phase}_{operand}_bit_zero_ratio" for phase in PHASES for operand in OPERANDS]
    fields += ["returncode", "error", "report", "log", "attempt", "fingerprint"]
    write_csv(output / "summary.csv", rows, fields)
    operand_rows = [{"run_name": entry["run_name"], "model": entry["model"],
                     "format": entry["format"], "status": entry["status"], **row}
                    for entry in entries for row in entry.get("operands", [])]
    write_csv(output / "operand_sparsity.csv", operand_rows,
              ["run_name", "model", "format", "status", "phase", "operand", "outlier_masked",
               *COUNT_FIELDS, "element_zero_ratio", "bit_zero_ratio"])


def run_matrix(args, executor=subprocess.run):
    jobs = build_jobs(args)
    output = args.output_dir.expanduser().resolve()
    print(f"{len(jobs)} jobs; FP explicit mantissa; mask zeros included; unit disabled", flush=True)
    if args.dry_run:
        for job in jobs:
            config = job["config"]
            print(f"\n{job['run_name']}: seq={config['evaluation']['seq_length']}, "
                  f"PPL tokens={config['evaluation']['max_eval_tokens']}, "
                  f"Linear outlier={config['quantization']['q_proj']['outlier_ratio']}")
            print(f"  template: {job['template']}")
            print("  " + shlex.join(command_for(job, output / "jobs" / job["run_name"] / "attempt_0001")))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "schema_version": 1, "matrix": "sparsity_ppl_rerun", "metric": METRIC, "jobs": {},
    }
    if manifest.get("schema_version") != 1 or manifest.get("matrix") != "sparsity_ppl_rerun":
        raise ValueError("Output directory contains an incompatible manifest; choose another directory")
    for job in jobs:
        manifest["jobs"].setdefault(job["run_name"], {
            "run_name": job["run_name"], "model": job["model"], "format": job["format"], "status": "pending",
        })
    save_results(output, manifest)
    failed = False
    for index, job in enumerate(jobs, 1):
        previous = manifest["jobs"][job["run_name"]]
        if args.resume and previous["status"] == "completed" and previous.get("fingerprint") == job["fingerprint"]:
            try:
                checked = read_report(job, Path(previous["attempt"]))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"[{index}/{len(jobs)}] {job['run_name']}: rerun; saved report invalid ({exc})", flush=True)
            else:
                if checked["status"] == "completed":
                    previous.update(checked)
                    print(f"[{index}/{len(jobs)}] {job['run_name']}: resume skipped validated result", flush=True)
                    save_results(output, manifest)
                    continue

        job_dir = output / "jobs" / job["run_name"]
        attempt_number = 1
        while (job_dir / f"attempt_{attempt_number:04d}").exists():
            attempt_number += 1
        attempt = job_dir / f"attempt_{attempt_number:04d}"
        attempt.mkdir(parents=True)
        config = copy.deepcopy(job["config"])
        config["quantization"]["scale_dir"] = str(attempt / "scales")
        config["experiment"] = {
            "matrix": "sparsity_ppl_rerun", "fingerprint": job["fingerprint"], **job["provenance"],
        }
        (attempt / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        entry = {
            "run_name": job["run_name"], "model": job["model"], "format": job["format"],
            "status": "running", "fingerprint": job["fingerprint"],
            "eval_flow": job["eval_flow"], "device": job["device"],
            "seq_length": config["evaluation"]["seq_length"],
            "max_eval_tokens": config["evaluation"]["max_eval_tokens"],
            "linear_outlier_ratio": config["quantization"]["q_proj"]["outlier_ratio"],
            "qk_pv_outlier_ratio": config["quantization"]["qk_matmul"]["outlier_ratio"],
            "attempt": str(attempt), "log": str(attempt / "run.log"),
        }
        manifest["jobs"][job["run_name"]] = entry
        save_results(output, manifest)
        command = command_for(job, attempt)
        print(f"[{index}/{len(jobs)}] {job['run_name']}: starting; log={entry['log']}", flush=True)
        started = time.monotonic()
        try:
            with (attempt / "run.log").open("w", encoding="utf-8") as log:
                log.write(shlex.join(command) + "\n")
                log.flush()
                process = executor(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=False)
            entry["returncode"] = process.returncode
            try:
                entry.update(read_report(job, attempt))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                entry.update(status="failed", error=str(exc))
            if process.returncode != 0:
                entry.update(status="failed", error=f"Pipeline exited with code {process.returncode}; see run.log")
        except KeyboardInterrupt:
            entry.update(status="interrupted", returncode=130, elapsed_seconds=time.monotonic() - started)
            save_results(output, manifest)
            print("Interrupted; partial attempt retained. Use --resume to continue.", flush=True)
            return 130
        except OSError as exc:
            entry.update(status="failed", returncode=None, error=str(exc))
        entry["elapsed_seconds"] = time.monotonic() - started
        failed |= entry["status"] != "completed"
        save_results(output, manifest)
        print(f"[{index}/{len(jobs)}] {job['run_name']}: {entry['status']}; "
              f"PPL={entry.get('summary', {}).get('ppl')}", flush=True)

    print(f"Summary: {output / 'summary.csv'}\nOperand counts: {output / 'operand_sparsity.csv'}", flush=True)
    return 1 if failed else 0


def main(argv=None):
    try:
        return run_matrix(parse_args(argv))
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
