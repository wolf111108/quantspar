"""Add remaining operator latencies to an existing Bitlet summary without rerunning Qwen."""
import argparse
import json
from pathlib import Path

from quant.bitlet import validate_other_latency


def estimate(summary, other):
    if summary.get("backend") != "bitlet" or summary.get("schema_version") != 1:
        raise ValueError("Input must be a Bitlet schema_version=1 summary")
    workload = summary["workload"]
    if workload.get("status") != "complete" or not summary["latency"]["collection_complete"]:
        raise ValueError("Complete profiling is required before adding E2E costs")
    validate_other_latency(other, workload["decode_steps"])
    extra = other["prefill_seconds"]+sum(other["decode_step_seconds"])
    modeled = summary["latency"]["gemm_and_io_seconds"]
    e2e = {key: value+extra for key, value in modeled.items()}
    decode = summary["latency"]["per_phase_seconds"].get("decode", {})
    return dict(schema_version=1, backend="bitlet", source_commit=workload.get("source_commit"),
                workload=workload, config=summary["config"],
                scope="conditional E2E scenarios with supplied remaining operators; not measured hardware latency",
                gemm_and_io_seconds=modeled, supplied_other_latency=other,
                conditional_e2e_seconds=e2e,
                average_decode_seconds=(
                    {key: (value+sum(other["decode_step_seconds"]))/workload["decode_steps"]
                     for key, value in decode.items()} if workload["decode_steps"] else None),
                assumptions=summary["assumptions"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--other-latency-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() in (args.summary.resolve(), args.other_latency_json.resolve()):
        raise ValueError("Output must not overwrite either input")
    result = estimate(json.loads(args.summary.read_text()),
                      json.loads(args.other_latency_json.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for scenario, seconds in result["conditional_e2e_seconds"].items():
        print(f"{scenario}: {seconds:.6f} s (conditional)")


if __name__ == "__main__":
    main()
