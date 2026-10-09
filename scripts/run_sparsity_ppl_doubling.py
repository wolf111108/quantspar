"""Iteratively double outlier ratios until every group's PPL is below a threshold.

Round 1 runs the selected model/format matrix with template defaults, or with
--init-ratio/--init-qk-pv-ratio when continuing from a previous manual experiment
(e.g. --init-ratio 0.0004 --init-qk-pv-ratio 0.0004 resumes after a 4e-4 round;
round 2 then starts doubling from 8e-4). After each round, groups whose FineWeb
PPL exceeds --ppl-threshold get both their Linear and QK/PV outlier ratios
doubled (a zero ratio starts from --seed-ratio) and are rerun alone in a fresh
round_NN directory. The loop stops when every group is below the threshold,
when no failing group can still be doubled (--max-ratio), or after --max-rounds.
A zero ratio only starts from the seed; ratios never exceed --max-ratio. Each
failing job is dispatched as its own runner invocation (one model, one format,
one ratio pair) because ratios diverge per group across rounds; runner
invocations into the same round directory accumulate one manifest/summary.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_sparsity_ppl_matrix as runner

MODELS = runner.MODELS
FORMATS = runner.FORMATS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt-1-3b-path", help="Local OPT-1.3B checkpoint directory")
    parser.add_argument("--opt-6-7b-path", help="Local OPT-6.7B checkpoint directory")
    parser.add_argument("--qwen-7b-path", help="Local Qwen2.5-7B checkpoint directory")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--eval-flow", choices=("all", "ppl"), default="all")
    parser.add_argument("--ppl-threshold", type=float, default=20.0)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-ratio", type=float, default=0.1,
                        help="Hard cap for both outlier ratios; failing groups already at the cap stop")
    parser.add_argument("--seed-ratio", type=float, default=0.0001,
                        help="First ratio for a group whose previous ratio was zero")
    parser.add_argument("--init-ratio", type=runner.outlier_ratio, default=None,
                        help="Start round 1 with this Linear outlier ratio instead of template defaults")
    parser.add_argument("--init-qk-pv-ratio", type=runner.outlier_ratio, default=None,
                        help="Start round 1 with this QK/PV outlier ratio; defaults to --init-ratio")
    parser.add_argument("--init-results", type=Path, default=None,
                        help="Seed PPL values from a previous experiment's summary.csv matching "
                             "--init-ratio/--init-qk-pv-ratio, skipping a redundant round 1 re-run; "
                             "groups found there enter round 2 (doubling) directly, missing groups "
                             "still run round 1 with the init ratios")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sparsity_ppl_doubling"))
    return parser.parse_args(argv)


def invocation_argv(args, model, fmt, linear, qk_pv, round_dir):
    """Build one runner invocation covering a single (model, format) pair."""
    argv = ["--models", model, "--formats", fmt, "--device", args.device,
            "--eval-flow", args.eval_flow, "--output-dir", str(round_dir),
            f"--{MODELS[model][1].replace('_', '-')}", getattr(args, MODELS[model][1])]
    if linear is not None:
        argv += ["--outlier-ratio", repr(linear)]
    if qk_pv is not None:
        argv += ["--qk-pv-outlier-ratio", repr(qk_pv)]
    return argv


def read_round_ppl(round_dir):
    """Map run_name -> (ppl or None, status) from a round's summary.csv."""
    result = {}
    with (round_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("ppl")
            result[row["run_name"]] = (float(value) if value not in ("", None) else None, row["status"])
    return result


def seeded_ratios(round_dir):
    """Map run_name -> (linear_ratio, qk_pv_ratio) recorded in a summary.csv."""
    result = {}
    with (round_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                result[row["run_name"]] = (float(row["linear_outlier_ratio"]), float(row["qk_pv_outlier_ratio"]))
            except (KeyError, TypeError, ValueError):
                continue
    return result


def doubled(ratio, args):
    # None means the template default ran (Linear: 0.0001 quantized / 0 for BF16; QK/PV: 0);
    # treat it as unknown-but-small and start doubling from the seed like a zero ratio.
    value = ratio * 2 if ratio else args.seed_ratio
    return min(value, args.max_ratio)


def write_summary(path, state, threshold):
    fields = ["run_name", "model", "format", "final_ppl", "final_status", "meets_threshold",
              "rounds", "final_linear_outlier_ratio", "final_qk_pv_outlier_ratio",
              "ppl_history", "linear_history", "qk_pv_history", "exhausted"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in sorted(state):
            entry = state[name]
            writer.writerow({
                "run_name": name, "model": entry["model"], "format": entry["format"],
                "final_ppl": entry["ppl"], "final_status": entry["status"],
                "meets_threshold": entry["ppl"] is not None and entry["ppl"] <= threshold,
                "rounds": len(entry["ppl_history"]),
                "final_linear_outlier_ratio": entry["linear"],
                "final_qk_pv_outlier_ratio": entry["qk_pv"],
                "ppl_history": ";".join("None" if p is None else repr(p) for p in entry["ppl_history"]),
                "linear_history": ";".join(repr(r) for r in entry["linear_history"]),
                "qk_pv_history": ";".join(repr(r) for r in entry["qk_pv_history"]),
                "exhausted": entry.get("exhausted", False),
            })


def run_doubling(args, executor=None):
    executor = executor or subprocess.run
    for model in args.models:
        checkpoint = getattr(args, MODELS[model][1])
        if checkpoint is None or not Path(checkpoint).expanduser().is_dir():
            raise ValueError(f"Missing checkpoint directory for {model}; use --{MODELS[model][1].replace('_', '-')}")

    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    init = (args.init_ratio, args.init_qk_pv_ratio if args.init_qk_pv_ratio is not None else args.init_ratio)
    state = {f"{MODELS[model][0]}_{fmt}": {"model": model, "format": fmt, "linear": init[0], "qk_pv": init[1],
                                           "ppl": None, "status": "pending",
                                           "ppl_history": [], "linear_history": [], "qk_pv_history": []}
             for model in args.models for fmt in args.formats}
    seeded = set()
    if args.init_results is not None:
        if init[0] is None:
            raise ValueError("--init-results requires --init-ratio describing the seeded experiment's ratios")
        seeded_ppl = read_round_ppl(args.init_results.expanduser().resolve())
        for name, entry in state.items():
            if name not in seeded_ppl:
                continue
            ppl, status = seeded_ppl[name]
            if ppl is None:
                continue
            row_ratio = seeded_ratios(args.init_results.expanduser().resolve()).get(name)
            if row_ratio is not None and tuple(row_ratio) != init:
                raise ValueError(f"{name}: seeded ratios {row_ratio} do not match --init-ratio {init}")
            entry.update(ppl=ppl, status=status, seeded=True)
            entry["ppl_history"].append(ppl)
            entry["linear_history"].append(init[0])
            entry["qk_pv_history"].append(init[1])
            seeded.add(name)
        print(f"Seeded {len(seeded)} group(s) from {args.init_results}; they skip round 1 and start doubling",
              flush=True)
    threshold = args.ppl_threshold

    for round_index in range(1, args.max_rounds + 1):
        round_dir = root / f"round_{round_index:02d}"
        if round_index == 1:
            schedule = [name for name in state if name not in seeded]
            overrides = {name: init for name in schedule}
            for name in schedule:  # record the explicit starting ratios for round 2 doubling
                state[name]["linear"], state[name]["qk_pv"] = init
            if not schedule:  # everything seeded: run the first doubling round inside round_01
                failing = [name for name, entry in state.items()
                           if entry["ppl"] is None or entry["ppl"] > threshold]
                if not failing:
                    break
                schedule, overrides = [], {}
                for name in failing:
                    entry = state[name]
                    linear, qk_pv = doubled(entry["linear"], args), doubled(entry["qk_pv"], args)
                    if linear == entry["linear"] and qk_pv == entry["qk_pv"]:
                        entry["exhausted"] = True
                        print(f"{name}: at ratio cap {args.max_ratio} with PPL={entry['ppl']}; stop doubling",
                              flush=True)
                        continue
                    entry["linear"], entry["qk_pv"] = linear, qk_pv
                    schedule.append(name)
                    overrides[name] = (linear, qk_pv)
                if not schedule:
                    break
        else:
            failing = [name for name, entry in state.items()
                       if entry["ppl"] is None or entry["ppl"] > threshold]
            if not failing:
                break
            schedule, overrides = [], {}
            for name in failing:
                entry = state[name]
                linear, qk_pv = doubled(entry["linear"], args), doubled(entry["qk_pv"], args)
                if linear == entry["linear"] and qk_pv == entry["qk_pv"]:
                    entry["exhausted"] = True  # already at --max-ratio and still failing
                    print(f"{name}: at ratio cap {args.max_ratio} with PPL={entry['ppl']}; stop doubling",
                          flush=True)
                    continue
                entry["linear"], entry["qk_pv"] = linear, qk_pv
                schedule.append(name)
                overrides[name] = (linear, qk_pv)
            if not schedule:
                break

        for name in schedule:
            entry = state[name]
            linear, qk_pv = overrides[name]
            print(f"round {round_index}: {name}: linear={linear if linear is not None else 'default'} "
                  f"qk_pv={qk_pv if qk_pv is not None else 'default'}", flush=True)
            argv = invocation_argv(args, entry["model"], entry["format"], linear, qk_pv, round_dir)
            runner.run_matrix(runner.parse_args(argv), executor=executor)
        for name, (ppl, status) in read_round_ppl(round_dir).items():
            if name not in state:
                continue
            state[name].update(ppl=ppl, status=status)
        for name in schedule:
            entry = state[name]
            entry["ppl_history"].append(entry["ppl"])
            entry["linear_history"].append(entry["linear"] if entry["linear"] is not None else "default")
            entry["qk_pv_history"].append(entry["qk_pv"] if entry["qk_pv"] is not None else "default")

    write_summary(root / "doubling_summary.csv", state, threshold)
    pending = [name for name, entry in state.items()
               if entry["ppl"] is None or entry["ppl"] > threshold]
    for name in sorted(state):
        entry = state[name]
        print(f"{name}: PPL={entry['ppl']} linear={entry['linear']} qk_pv={entry['qk_pv']} "
              f"rounds={len(entry['ppl_history'])}{' EXHAUSTED' if entry.get('exhausted') else ''}", flush=True)
    print(f"Doubling summary: {root / 'doubling_summary.csv'}", flush=True)
    if pending:
        print(f"{len(pending)} group(s) still above PPL {threshold}: {', '.join(sorted(pending))}", flush=True)
        return 1
    return 0


def main(argv=None):
    try:
        return run_doubling(parse_args(argv))
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
