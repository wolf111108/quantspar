"""Config/report/launcher checks with synthetic reports, without model execution."""

import contextlib
import csv
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from scripts import evaluation_report as report
from scripts import run_sparsity_ppl_matrix as runner


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        self.output = self.root / "output"
        self.calls = []

    def args(self, *extra):
        return runner.parse_args([
            "--models", "opt-1.3b", "--formats", "bf16_bf16", "fp8_int4",
            "--opt-1-3b-path", str(self.checkpoint), "--output-dir", str(self.output), *extra,
        ])

    @staticmethod
    def snapshot(phases):
        rows = []
        for phase in phases:
            for operand in runner.OPERANDS:
                rows.append(dict(phase=phase, operand=operand, outlier_masked=False,
                                 elements=10, zero_elements=1, bits=100, zero_bits=10))
            # A second group has very different size and ratio; average=0.5 is wrong.
            rows.append(dict(phase=phase, operand="activation", outlier_masked=True,
                             elements=1, zero_elements=1, bits=10, zero_bits=9))
        return dict(schema_version=2, bit_scope="mantissa", records=[], phase_operands=rows,
                    total=dict(bits=1, zero_bits=1, bit_zero_ratio=1.0))

    def executor(self, command, *, cwd, stdout, stderr, check, outcomes=None):
        self.calls.append(command)
        self.assertEqual(cwd, str(runner.ROOT))
        self.assertEqual(stderr, subprocess.STDOUT)
        self.assertFalse(check)
        self.assertIn("--no-unit-sparsity", command)
        self.assertNotIn("--fp-baseline", command)
        self.assertNotIn("--skip-calibration", command)
        value = lambda flag: command[command.index(flag) + 1]
        run_name = value("--run-name")
        outcome = (outcomes or {}).get(run_name, 5.123456789)
        if outcome == "fail":
            return SimpleNamespace(returncode=7)
        config = yaml.safe_load(Path(value("--config")).read_text())
        self.assertEqual(config["quantization"]["calibration_policy"]["default"], "recalibrate")
        self.assertEqual(Path(config["quantization"]["scale_dir"]).parent,
                         Path(value("--config")).parent)
        args = SimpleNamespace(run_name=run_name, config=value("--config"),
                               model_path=value("--model-path"), device=value("--device"),
                               eval_flow=value("--eval-flow"), results_dir=value("--results-dir"),
                               stats_output_dir=value("--stats-output-dir"),
                               unit_sparsity=False)
        snapshots = {"full_forward": self.snapshot(["full_forward"])}
        if args.eval_flow == "all":
            snapshots["prefill_decode"] = self.snapshot(["prefill", "decode"])
        for flow, snapshot in snapshots.items():
            root = Path(value("--stats-output-dir"))
            root.mkdir(exist_ok=True)
            (root / f"{run_name}_{flow}_bit_sparsity.json").write_text(json.dumps(snapshot))
            (root / f"{run_name}_{flow}_bit_sparsity.csv").write_text("phase,operand,bits,zero_bits\n")
        report.export_evaluation_report(args, config, {"fineweb": {"perplexity": outcome, "time": 1.25}}, snapshots)
        return SimpleNamespace(returncode=0)

    def run_matrix(self, args, executor=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run_matrix(args, executor=executor or self.executor)

    def summary(self):
        with (self.output / "summary.csv").open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_all_fifteen_configs_match_real_wrapper_layers(self):
        expected = {"bf16_bf16": ("bf16", "bf16"), "fp8_fp8": ("e4m3", "e4m3"),
                    "int8_int8": (8, 8), "int8_int4": (8, 4), "fp8_int4": ("e4m3", 4)}
        paths = list((runner.ROOT / "config/sparsity_ppl_rerun").glob("*.yaml"))
        self.assertEqual(len(paths), 15)
        scale_dirs = set()
        for model, (stem, _) in runner.MODELS.items():
            opt = model.startswith("opt")
            names = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2") if opt else (
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
            for fmt, (a, w) in expected.items():
                with self.subTest(model=model, format=fmt):
                    config = yaml.safe_load((runner.ROOT / "config/sparsity_ppl_rerun" / f"{stem}_{fmt}.yaml").read_text())
                    q = config["quantization"]
                    self.assertEqual(config["model"]["name"], model)
                    self.assertEqual(config["model"]["dtype"], "bf16")
                    self.assertEqual(config["model"]["attn_implementation"], "eager")
                    self.assertEqual(q["calibration_policy"], {"default": "recalibrate"})
                    self.assertFalse(q["mixed_precision"])
                    self.assertFalse(config["unit_sparsity"]["enabled"])
                    self.assertFalse(q["lm_head"]["enabled"])
                    self.assertEqual({n for n, v in q.items() if isinstance(v, dict) and "a_bit" in v}, set(names))
                    for name in names:
                        self.assertEqual((q[name]["a_bit"], q[name]["w_bit"], q[name]["o_bit"]), (a, w, a))
                        self.assertEqual(q[name]["outlier_ratio"], 0 if fmt == "bf16_bf16" else 0.0001)
                    for name in ("qk_matmul", "pv_matmul"):
                        self.assertEqual((q[name]["A_bit"], q[name]["B_bit"], q[name]["O_bit"]), (a, a, a))
                        self.assertEqual(q[name]["outlier_ratio"], 0)
                    length = 2048 if opt else 8192
                    self.assertEqual(config["calibration"]["seq_length"], length)
                    self.assertEqual(config["calibration"]["num_samples"], 64)
                    self.assertEqual(config["calibration"]["seed"], 23)
                    self.assertEqual(config["evaluation"]["seq_length"], length)
                    dataset = config["evaluation"]["datasets"][0]
                    self.assertEqual((dataset["name"], dataset["config"], dataset["split"]),
                                     ("HuggingFaceFW/fineweb", "sample-10BT", "train"))
                    self.assertEqual(dataset["max_eval_tokens"], 65536)
                    self.assertEqual(config["evaluation"]["max_eval_tokens"], 65536)
                    profile = config["prefill_decode_profile"]
                    self.assertEqual(profile["decode_steps"], 64)
                    if opt:
                        self.assertLessEqual(profile["prefill_length"] + profile["decode_steps"], 2048)
                    self.assertNotIn(q["scale_dir"], scale_dirs)
                    scale_dirs.add(q["scale_dir"])

    def test_dry_run_has_fifteen_jobs_and_no_output(self):
        args = runner.parse_args(["--dry-run", "--output-dir", str(self.output)])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = runner.run_matrix(args, executor=lambda *a, **k: self.fail("Executed a model"))
        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue().count("  template:"), 15)
        self.assertFalse(self.output.exists())

    def test_report_recomputes_weighted_counts_and_preserves_nan_status(self):
        args = SimpleNamespace(run_name="synthetic", config="synthetic.yaml", model_path="synthetic",
                               eval_flow="ppl", device="cpu", results_dir=self.output, unit_sparsity=False)
        path = report.export_evaluation_report(args, {}, {"fineweb": {"perplexity": float("nan"), "time": 1}},
                                               {"full_forward": self.snapshot(["full_forward"])})
        doc = json.loads(path.read_text(), parse_constant=lambda value: self.fail(f"Nonstandard JSON: {value}"))
        self.assertIsNone(doc["ppl"]["fineweb"]["perplexity"])
        self.assertEqual(doc["ppl"]["fineweb"]["status"], "nonfinite")
        self.assertEqual(doc["ppl"]["fineweb"]["raw_value"], "nan")
        self.assertAlmostEqual(doc["bit_sparsity"]["full_forward"]["total"]["bit_zero_ratio"], 69 / 610)

    def test_success_resume_ratio_change_and_missing_file(self):
        args = self.args()
        self.assertEqual(self.run_matrix(args), 0)
        self.assertEqual(len(self.calls), 2)
        rows = self.summary()
        self.assertEqual({row["status"] for row in rows}, {"completed"})
        self.assertEqual(float(rows[0]["ppl"]), 5.123456789)
        self.assertAlmostEqual(float(rows[0]["full_forward_activation_bit_zero_ratio"]), 19 / 110)
        self.assertAlmostEqual(float(rows[0]["full_forward_total_bit_zero_ratio"]), 69 / 610)
        self.assertNotEqual(rows[0]["attempt"], rows[1]["attempt"])
        args.resume = True
        self.assertEqual(self.run_matrix(args), 0)
        self.assertEqual(len(self.calls), 2)
        args.outlier_ratio = 0.0
        self.assertEqual(self.run_matrix(args), 0)
        self.assertEqual(len(self.calls), 3)  # BF16 stays identical; FP8/INT4 changes.
        rows = self.summary()
        self.assertEqual(Path(rows[1]["attempt"]).name, "attempt_0002")
        stale = Path(rows[0]["attempt"]) / "stats/opt_1.3b_bf16_bf16_full_forward_bit_sparsity.csv"
        stale.write_text("")
        self.assertEqual(self.run_matrix(args), 0)
        self.assertEqual(len(self.calls), 4)  # Empty CSV must not be accepted by --resume.
        self.assertTrue(stale.exists())  # Previous attempt is retained.
        altered = Path(self.summary()[0]["attempt"]) / "stats/opt_1.3b_bf16_bf16_full_forward_bit_sparsity.json"
        altered.write_text('{"changed": true}')
        self.assertEqual(self.run_matrix(args), 0)
        self.assertEqual(len(self.calls), 5)  # Nonempty corruption is detected by checksum.

    def test_qk_pv_outlier_ratio_override_changes_fingerprint_and_config(self):
        base = self.args("--formats", "fp8_int4")
        self.assertEqual(self.run_matrix(base), 0)
        with_qk = self.args("--formats", "fp8_int4", "--qk-pv-outlier-ratio", "0.0002")
        self.assertEqual(self.run_matrix(with_qk), 0)
        self.assertEqual(len(self.calls), 2)
        jobs = sorted((self.output / "jobs").glob("opt_1.3b_fp8_int4/attempt_*"))
        self.assertEqual(len(jobs), 2)
        plain = yaml.safe_load((jobs[0] / "config.yaml").read_text())
        overridden = yaml.safe_load((jobs[1] / "config.yaml").read_text())
        self.assertEqual(plain["quantization"]["qk_matmul"]["outlier_ratio"], 0)
        self.assertEqual(overridden["quantization"]["qk_matmul"]["outlier_ratio"], 0.0002)
        self.assertEqual(overridden["quantization"]["pv_matmul"]["outlier_ratio"], 0.0002)
        self.assertEqual(overridden["quantization"]["q_proj"]["outlier_ratio"], 0.0001)  # Linear untouched
        rows = self.summary()
        self.assertEqual(len(rows), 1)  # Same run_name: the manifest keeps only the latest attempt.
        self.assertEqual(float(rows[0]["qk_pv_outlier_ratio"]), 0.0002)
        self.assertEqual(rows[0]["attempt"].endswith("attempt_0002"), True)

    def test_failure_and_nonfinite_ppl_continue_without_fake_zero(self):
        args = self.args("--formats", "bf16_bf16", "fp8_fp8", "int8_int8")
        outcomes = {"opt_1.3b_bf16_bf16": "fail", "opt_1.3b_fp8_fp8": float("inf")}
        executor = lambda *a, **k: self.executor(*a, **k, outcomes=outcomes)
        self.assertEqual(self.run_matrix(args, executor), 1)
        self.assertEqual(len(self.calls), 3)
        rows = self.summary()
        self.assertEqual([row["status"] for row in rows], ["failed", "nonfinite_ppl", "completed"])
        self.assertEqual(rows[0]["ppl"], "")
        self.assertEqual(rows[0]["full_forward_total_bit_zero_ratio"], "")
        self.assertEqual(rows[1]["ppl"], "")
        self.assertEqual(rows[1]["ppl_raw_value"], "inf")
        self.assertNotEqual(rows[1]["full_forward_total_bit_zero_ratio"], "")

    def test_interrupt_preserves_pending_jobs(self):
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt
        self.assertEqual(self.run_matrix(self.args(), interrupted), 130)
        self.assertEqual([row["status"] for row in self.summary()], ["interrupted", "pending"])

    def test_doubling_stops_when_all_groups_meet_threshold(self):
        from scripts import run_sparsity_ppl_doubling as doubling
        checkpoint = self.checkpoint
        script_ppl = {"opt_1.3b_bf16_bf16": 5.0, "opt_1.3b_fp8_int4": 60.0}
        invocations = []

        def executor(command, *, cwd, stdout, stderr, check):
            invocations.append(command)
            return self.executor(command, cwd=cwd, stdout=stdout, stderr=stderr, check=check)

        base = doubling.parse_args([
            "--models", "opt-1.3b", "--formats", "bf16_bf16", "fp8_int4",
            "--opt-1-3b-path", str(checkpoint), "--ppl-threshold", "20",
            "--output-dir", str(self.root / "doubling")])

        def patched_runner(argv=None):
            args = runner.parse_args(argv)
            outcomes = {name: script_ppl[name] for name in script_ppl
                        if runner.MODELS[args.models[0]][0] in name and args.formats[0] in name}
            # Only the selected single (model, format) job runs this invocation.
            selected = f"{runner.MODELS[args.models[0]][0]}_{args.formats[0]}"
            outcome = script_ppl[selected]
            def exec_one(command, *, cwd, stdout, stderr, check):
                self.calls.append(command)
                value = lambda flag: command[command.index(flag) + 1]
                run_name = value("--run-name")
                self.assertEqual(run_name, selected)
                return self.executor(command, cwd=cwd, stdout=stdout, stderr=stderr, check=check,
                                     outcomes={run_name: outcome})
            return runner.run_matrix(args, executor=exec_one)

        # Drive the loop with a stubbed runner entry.
        calls = []
        real_run_matrix = runner.run_matrix

        def value_of(command, flag):
            return command[command.index(flag) + 1]

        def fake_run_matrix(runner_args, executor=None):
            calls.append(runner_args)
            selected = f"{runner.MODELS[runner_args.models[0]][0]}_{runner_args.formats[0]}"
            def exec_one(command, *, cwd, stdout, stderr, check):
                self.calls.append(command)
                return self.executor(command, cwd=cwd, stdout=stdout, stderr=stderr, check=check,
                                     outcomes={value_of(command, "--run-name"): script_ppl[selected]})
            return real_run_matrix(runner_args, executor=exec_one)

        with mock.patch.object(doubling.runner, "run_matrix", new=fake_run_matrix), \
             contextlib.redirect_stdout(io.StringIO()):
            # Round 1: fp8_int4 fails (60 > 20); later rounds keep doubling while it stays 60.
            code = doubling.run_doubling(base)
            self.assertEqual(code, 1)  # bf16 fine, fp8_int4 never converges at fixed PPL 60
        # Round 1 dispatches both groups (2 calls); rounds 2..8 dispatch only fp8_int4 (7 calls).
        self.assertEqual(len(calls), 2 + 7)
        self.assertEqual([a.formats[0] for a in calls[:2]], ["bf16_bf16", "fp8_int4"])
        self.assertEqual([a.formats[0] for a in calls[2:]], ["fp8_int4"] * 7)
        self.assertEqual(calls[2].outlier_ratio, 0.0001)  # None default -> seed ratio
        self.assertEqual(calls[2].qk_pv_outlier_ratio, 0.0001)
        self.assertEqual(calls[3].outlier_ratio, 0.0002)
        self.assertEqual(calls[-1].outlier_ratio, min(0.0001 * 2 ** 6, 0.1))  # round 8 = 6th doubling
        rows = list(csv.DictReader((self.root / "doubling" / "doubling_summary.csv").open()))
        by_name = {r["run_name"]: r for r in rows}
        self.assertEqual(by_name["opt_1.3b_bf16_bf16"]["final_ppl"], "5.0")
        self.assertEqual(by_name["opt_1.3b_bf16_bf16"]["rounds"], "1")
        self.assertEqual(by_name["opt_1.3b_fp8_int4"]["final_ppl"], "60.0")
        self.assertEqual(by_name["opt_1.3b_fp8_int4"]["rounds"], "8")  # hit --max-rounds

        # Now make the second run converge: first invocation 60, later ones 15.
        script_ppl["opt_1.3b_fp8_int4"] = 15.0
        calls.clear()
        original_executor = self.executor
        attempt_count = {"n": 0}
        real_run2 = real_run_matrix

        def fake_run_matrix2(runner_args, executor=None):
            calls.append(runner_args)
            selected = f"{runner.MODELS[runner_args.models[0]][0]}_{runner_args.formats[0]}"

            def exec_one(command, *, cwd, stdout, stderr, check):
                self.calls.append(command)
                run_name = value_of(command, "--run-name")
                if run_name == "opt_1.3b_fp8_int4":
                    attempt_count["n"] += 1
                outcome = 60.0 if run_name == "opt_1.3b_fp8_int4" and attempt_count["n"] == 1 else script_ppl[run_name]
                return self.executor(command, cwd=cwd, stdout=stdout, stderr=stderr, check=check,
                                     outcomes={run_name: outcome})
            return real_run2(runner_args, executor=exec_one)

        with mock.patch.object(doubling.runner, "run_matrix", new=fake_run_matrix2), \
             contextlib.redirect_stdout(io.StringIO()):
            code = doubling.run_doubling(base)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 3)  # round 1 both; round 2 only fp8_int4 (60->15)
        rows = list(csv.DictReader((self.root / "doubling" / "doubling_summary.csv").open()))
        by_name = {r["run_name"]: r for r in rows}
        self.assertEqual(by_name["opt_1.3b_fp8_int4"]["final_ppl"], "15.0")
        self.assertEqual(by_name["opt_1.3b_fp8_int4"]["final_linear_outlier_ratio"], "0.0001")
        self.assertEqual(by_name["opt_1.3b_fp8_int4"]["rounds"], "2")
        self.assertEqual(by_name["opt_1.3b_bf16_bf16"]["rounds"], "1")
        self.assertEqual(by_name["opt_1.3b_bf16_bf16"]["rounds"], "1")

    def test_doubling_stops_at_ratio_cap(self):
        from scripts import run_sparsity_ppl_doubling as doubling
        calls = []
        real_run_matrix = runner.run_matrix

        def fake_run_matrix(runner_args, executor=None):
            calls.append(runner_args)
            def exec_one(command, *, cwd, stdout, stderr, check):
                self.calls.append(command)
                return self.executor(command, cwd=cwd, stdout=stdout, stderr=stderr, check=check,
                                     outcomes={command[command.index("--run-name") + 1]: 60.0})
            return real_run_matrix(runner_args, executor=exec_one)

        base = doubling.parse_args([
            "--models", "opt-1.3b", "--formats", "fp8_int4",
            "--opt-1-3b-path", str(self.checkpoint), "--ppl-threshold", "20",
            "--max-ratio", "0.0003", "--output-dir", str(self.root / "doubling_cap")])
        with mock.patch.object(doubling.runner, "run_matrix", new=fake_run_matrix), \
             contextlib.redirect_stdout(io.StringIO()):
            code = doubling.run_doubling(base)
        self.assertEqual(code, 1)  # never meets threshold
        # Round 1 default -> round 2 linear None->1e-4, qk_pv 0->1e-4; round 3 ->2e-4;
        # round 4 linear ->3e-4 (at cap), qk_pv ->4e-4 capped to 3e-4; round 5 exhausted, no call.
        self.assertEqual(len(calls), 4)
        rows = list(csv.DictReader((self.root / "doubling_cap" / "doubling_summary.csv").open()))
        self.assertEqual(rows[0]["exhausted"], "True")
        self.assertEqual(rows[0]["final_linear_outlier_ratio"], "0.0003")
        self.assertEqual(rows[0]["final_qk_pv_outlier_ratio"], "0.0003")  # 2e-4 doubled, capped at 3e-4

    def test_ppl_only_does_not_require_pd_reports(self):
        self.assertEqual(self.run_matrix(self.args("--eval-flow", "ppl")), 0)
        for row in self.summary():
            self.assertEqual(row["prefill_total_bit_zero_ratio"], "")
            self.assertEqual(row["decode_total_bit_zero_ratio"], "")

    def test_corrupt_counts_are_rejected_on_resume(self):
        args = self.args("--formats", "bf16_bf16")
        self.assertEqual(self.run_matrix(args), 0)
        path = Path(self.summary()[0]["report"])
        doc = json.loads(path.read_text())
        doc["bit_sparsity"]["full_forward"]["phase_operands"][0]["zero_bits"] = 101
        path.write_text(json.dumps(doc))
        args.resume = True
        self.assertEqual(self.run_matrix(args), 0)
        self.assertEqual(len(self.calls), 2)

    def test_shell_entry_from_another_directory(self):
        process = subprocess.run(["bash", str(runner.ROOT / "scripts/run_sparsity_ppl_matrix.sh"),
                                  "--dry-run", "--models", "qwen2.5-7b", "--formats", "fp8_int4",
                                  "--output-dir", str(self.output)], cwd=self.root,
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("qwen2.5_7b_fp8_int4: seq=8192", process.stdout)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
