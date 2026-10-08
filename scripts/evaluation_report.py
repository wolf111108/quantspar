"""Machine-readable evaluation reports; no model dependencies are imported."""

import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


COUNT_FIELDS = ("elements", "zero_elements", "bits", "zero_bits")


def runtime_versions():
    result = {"python": platform.python_version()}
    for package in ("torch", "transformers", "datasets", "PyYAML"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def sum_counts(rows):
    """Recompute ratios from integer counts, never from averages of ratios."""
    total = {key: sum(row[key] for row in rows) for key in COUNT_FIELDS}
    total["element_zero_ratio"] = (
        total["zero_elements"] / total["elements"] if total["elements"] else None
    )
    total["bit_zero_ratio"] = total["zero_bits"] / total["bits"] if total["bits"] else None
    return total


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_fingerprint(path):
    content = Path(path).read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def export_evaluation_report(args, config, results, sparsity_summaries):
    """Preserve full-precision PPL and phase snapshots before the manager resets."""
    ppl = {}
    for name, result in results.items():
        value = float(result["perplexity"])
        status = "ok" if math.isfinite(value) and value > 0 else "nonfinite"
        if math.isfinite(value) and value <= 0:
            status = "invalid"
        elapsed = float(result["time"])
        ppl[name] = {
            "perplexity": value if status == "ok" else None,
            "status": status,
            "raw_value": None if status == "ok" else str(value),
            "time_seconds": elapsed if math.isfinite(elapsed) else None,
        }

    stats = {}
    for flow, snapshot in sparsity_summaries.items():
        # The detailed layer/format records remain in the dedicated bit JSON/CSV.
        compact = {key: value for key, value in snapshot.items() if key != "records"}
        rows = snapshot["phase_operands"]
        compact["total"] = sum_counts(rows)
        compact["phase_totals"] = [
            {"phase": phase, **sum_counts([row for row in rows if row["phase"] == phase])}
            for phase in sorted({row["phase"] for row in rows})
        ]
        stats[flow] = compact

    run_name = args.run_name or Path(args.config).stem
    artifact_root = Path(getattr(args, "stats_output_dir", None) or args.results_dir)
    artifacts = {}
    for flow in sparsity_summaries:
        artifacts[flow] = {}
        for extension in ("json", "csv"):
            artifact_path = artifact_root / f"{run_name}_{flow}_bit_sparsity.{extension}"
            if artifact_path.is_file():
                artifacts[flow][extension] = file_fingerprint(artifact_path)
    doc = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "eval_flow": args.eval_flow,
        "device": args.device,
        "model_path": args.model_path,
        "config_file": str(args.config),
        "config": config,
        "runtime": runtime_versions(),
        "metric": {
            "fp_bit_scope": "explicit_mantissa",
            "int_bit_scope": "twos_complement",
            "aggregation": "sum_zero_bits / sum_counted_bits",
            "mask_generated_zeros": "included",
            "high_precision_sidepath_counted": False,
            "unit_sparsity": bool(args.unit_sparsity),
        },
        "ppl": ppl,
        "bit_sparsity": stats,
        "artifacts": artifacts,
    }
    path = Path(args.results_dir) / f"{run_name}_evaluation.json"
    write_json(path, doc)
    return path
