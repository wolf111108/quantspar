"""Independent operation counts and roofline/CLI checks; no Torch dependency."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.estimate_paper_latency import DEFAULT_CONFIG, Workload, estimate, main, validate_config


class PaperLatencyTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT_CONFIG.read_text())

    def test_fig16_names_and_independent_dense_capacity(self):
        result = estimate(self.config, Workload(decode_steps=0))
        rows = result["architectures"]
        self.assertEqual(set(rows), {"sigma", "bitwave", "ebb_cim", "bitpragmatic", "asyn_cim"})
        self.assertEqual([a["speedup"] for a in self.config["architectures"].values()],
                         [1.58, 1.58, 2.71, 3.33, 4.01])
        self.assertAlmostEqual(rows["sigma"]["dense_tops"], 16384*2*500e6/1e12)
        self.assertAlmostEqual(rows["bitwave"]["dense_tops"], 512*2*250e6/1e12)
        self.assertAlmostEqual(rows["ebb_cim"]["dense_tops"], 128*32*2*275e6/1e12)
        self.assertAlmostEqual(rows["asyn_cim"]["dense_tops"], 8.192)
        self.assertIsNone(rows["bitpragmatic"]["dense_tops"])
        self.assertNotEqual(rows["ebb_cim"]["dense_tops"], 3.55)
        self.assertEqual(rows["sigma"]["config"]["external_bandwidth_bytes_per_second"], 1024e9)

    def test_qwen_gqa_workload_and_decode_off_by_one(self):
        result = estimate(self.config, Workload())
        p, d = result["workload_counts"]["prefill"], result["workload_counts"]["decode"]
        # Independent analytic identities for Qwen14B's seven linear matrices.
        parameters = 48*(2*5120**2+2*5120*1024+3*5120*13824)
        self.assertEqual(parameters, 13212057600)
        self.assertEqual(result["workload"]["linear_parameters"], parameters)
        self.assertEqual(p["flops"], 2*parameters*2048+4*48*5120*2048**2)
        self.assertEqual(d["flops"], 2*parameters*256+4*48*5120*(256*2048+256*257//2))
        self.assertEqual(p["weight_bytes"], parameters//2)
        self.assertEqual(d["weight_bytes"], parameters//2*256)
        self.assertEqual(p["kv_read_bytes"], 2*48*1024*2048)
        self.assertEqual(p["kv_write_bytes"], 2*48*1024*2048)
        self.assertEqual(d["kv_read_bytes"], 2*48*1024*(256*2048+256*255//2))
        self.assertEqual(d["kv_write_bytes"], 2*48*1024*256)
        steps = result["architectures"]["asyn_cim"]["cases"]["fig16"]["decode_steps"]
        self.assertEqual((steps[0]["cache_length_before"], steps[-1]["cache_length_after"]), (2048, 2304))

    def test_per_operator_overlap_and_speedup_only_on_compute(self):
        work = Workload(prefill_length=2, decode_steps=2, layers=1, hidden_size=4,
                        intermediate_size=8, query_heads=2, kv_heads=1, head_dim=2)
        cfg = deepcopy(self.config)
        cfg["architectures"] = {"test": dict(name="test", speedup=3, pe_count=1,
                                             products_per_pe_per_cycle=1, dense_cycles_per_mac=1,
                                             frequency_hz=100, external_bandwidth_bytes_per_second=80,
                                             prefill_utilization=1, decode_utilization=0.5)}
        row = estimate(cfg, work)["architectures"]["test"]
        # Explicit independent calls: Q/K/V/O/gate/up/down then QK and PV.
        params = [16, 8, 8, 16, 32, 32, 32]
        for case, speed in (("dense", 1), ("fig16", 3)):
            expected_total = 0
            for step in range(3):
                m, s = (2, 2) if step == 0 else (1, 2+step)
                before = s if step == 0 else s-1
                rate = 200*speed*(1 if step == 0 else 0.5)
                flops = [2*m*n for n in params]+[2*2*m*s*2]*2
                byte_counts = [n//2 for n in params]+[2*before+2*m]*2
                expected_total += sum(max(f/rate, b/80) for f, b in zip(flops, byte_counts))
            self.assertAlmostEqual(row["cases"][case]["gemm_and_io_seconds"]["full_overlap"], expected_total)
        sparse = row["cases"]["fig16"]
        dense = row["cases"]["dense"]
        self.assertEqual(dense["phases"]["decode"]["transfer_seconds"], sparse["phases"]["decode"]["transfer_seconds"])
        self.assertAlmostEqual(dense["phases"]["decode"]["compute_seconds"]/3, sparse["phases"]["decode"]["compute_seconds"])
        self.assertGreater(sparse["gemm_and_io_seconds"]["full_overlap"],
                           max(sum(p["compute_seconds"] for p in sparse["phases"].values()),
                               sum(p["transfer_seconds"] for p in sparse["phases"].values())))

    def test_missing_inputs_are_not_zero_and_no_fake_e2e(self):
        result = estimate(self.config, Workload(decode_steps=1))
        rows = result["architectures"]
        for name in ("bitwave", "ebb_cim"):
            self.assertIsNotNone(rows[name]["cases"]["fig16"]["phases"]["prefill"]["compute_seconds"])
            self.assertIsNone(rows[name]["cases"]["fig16"]["gemm_and_io_seconds"]["full_overlap"])
        self.assertIsNone(rows["sigma"]["cases"]["fig16"]["conditional_e2e_seconds"]["full_overlap"])
        cfg = deepcopy(self.config)
        for a in cfg["architectures"].values():
            a["external_bandwidth_bytes_per_second"] = 1e12
        cfg["architectures"]["bitpragmatic"]["frequency_hz"] = 1e9
        other = dict(schema_version=1, includes_lm_head=True, prefill_seconds=0.2, decode_step_seconds=[0.3],
                     workload=dict(prefill_length=2048, decode_steps=1))
        result = estimate(cfg, Workload(decode_steps=1), other)
        self.assertTrue(result["other_workload_verified"])
        for row in result["architectures"].values():
            case = row["cases"]["fig16"]
            for schedule in ("full_overlap", "no_overlap"):
                self.assertAlmostEqual(case["conditional_e2e_seconds"][schedule], case["gemm_and_io_seconds"][schedule]+0.5)
        other["workload"]["prefill_length"] = 8192
        with self.assertRaisesRegex(ValueError, "does not match"):
            estimate(cfg, Workload(decode_steps=1), other)

    def test_materialized_activation_traffic_and_invalid_inputs(self):
        work = Workload(decode_steps=0)
        a = estimate(self.config, work)
        b = estimate(self.config, work, traffic_mode="materialized_once")
        first = a["workload_counts"]["prefill"]
        self.assertEqual(b["workload_counts"]["prefill"]["modeled_transfer_bytes"],
                         first["modeled_transfer_bytes"]+first["activation_io_bytes"])
        self.assertGreater(b["architectures"]["sigma"]["cases"]["fig16"]["phases"]["prefill"]["transfer_seconds"],
                           a["architectures"]["sigma"]["cases"]["fig16"]["phases"]["prefill"]["transfer_seconds"])
        for field, invalid in (("speedup", True), ("frequency_hz", float("nan")),
                               ("pe_count", 1.5), ("decode_utilization", 1.1),
                               ("external_bandwidth_bytes_per_second", -1)):
            cfg = deepcopy(self.config)
            cfg["architectures"]["sigma"][field] = invalid
            with self.assertRaises(ValueError): validate_config(cfg)
        with self.assertRaises(ValueError): Workload(decode_steps=-1)
        with self.assertRaises(ValueError): Workload(kv_heads=7)
        with self.assertRaises(ValueError): estimate(self.config, work, traffic_mode="unknown")
        for other in ({}, dict(schema_version=1, includes_lm_head=True, prefill_seconds=float("inf"), decode_step_seconds=[]),
                      dict(schema_version=1, includes_lm_head=True, prefill_seconds=0, decode_step_seconds=[0])):
            with self.assertRaises(ValueError): estimate(self.config, work, other)

    def test_cli_standard_library_overrides_provenance_and_safe_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"result"
            process = subprocess.run([sys.executable, "-S", "-m", "scripts.estimate_paper_latency",
                                      "--decode-steps", "2", "--output-dir", str(output),
                                      "--shared-external-bandwidth-gbps", "1000",
                                      "--external-bandwidth-gbps", "sigma=1024",
                                      "--frequency-mhz", "bitpragmatic=1000", "--decode-utilization", "bitwave=0.25"],
                                     cwd=DEFAULT_CONFIG.parents[1], capture_output=True, text=True, check=True)
            result = json.loads((output/"paper_latency_summary.json").read_text())
            self.assertEqual(len(result["architectures"]), 5)
            self.assertEqual(result["architectures"]["sigma"]["config"]["external_bandwidth_bytes_per_second"], 1024e9)
            self.assertEqual(result["architectures"]["bitwave"]["config"]["decode_utilization"], 0.25)
            self.assertAlmostEqual(result["architectures"]["bitpragmatic"]["dense_tops"], 8.192)
            self.assertTrue(result["overrides"])
            self.assertIn("conditional", process.stdout)
            self.assertEqual(len((output/"paper_latency_comparison.csv").read_text().splitlines()), 6)
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(FileExistsError): main(["--output-dir", str(output), "--decode-steps", "0"])
            for extra in (["--frequency-mhz", "bitlet=1000"], ["--decode-utilization", "sigma=2"],
                          ["--external-bandwidth-gbps", "bitwave=nan"]):
                with self.assertRaises(ValueError): main(["--output-dir", str(Path(tmp)/"bad"), *extra])
            self.assertFalse((Path(tmp)/"bad").exists())


if __name__ == "__main__":
    unittest.main()
