"""Independent workload identities, profile coverage and offline CLI checks."""
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.profile_bridge import build_profile
from scripts.estimate_profile_latency import estimate


ROOT = Path(__file__).resolve().parents[1]


def fixture(backend="bitwave", s=2, d=2):
    work = dict(status="complete", completed_decode_steps=d, prefill_length=s, decode_steps=d,
                layers=2, d_model=4, ffn_dim=8, q_heads=2, kv_heads=1, head_dim=2, batch_size=1)
    sizes = dict(q_proj=16, k_proj=8, v_proj=8, o_proj=16, gate_proj=32, up_proj=32, down_proj=32)
    layers = {}
    ebb = backend == "multcim_ebb_bounds"
    for phase in (["prefill", "decode"] if d else ["prefill"]):
        for name in [*sizes, "qk_matmul", "pv_matmul"]:
            linear = name in sizes
            macs = (sizes[name]*(s if phase == "prefill" else d) if linear else
                    4*(s*s if phase == "prefill" else d*s+d*(d+1)//2))
            for layer in range(2):
                # Layer 1 has twice the cost; reductions must preserve per-layer max(compute,IO).
                factor = (layer+1)*(1 if phase == "prefill" else 3)
                cycles = ({"dense": 4*factor*macs, "ideal_balanced": factor*macs,
                           "leading_zero": 2*factor*macs} if ebb else
                          {"dense_tiles": 4*factor*macs, "mapped_tiles": factor*macs})
                row = dict(calls=1 if phase == "prefill" else d, dense_equivalent_operations=2*macs,
                           cycles=cycles, sampled_calls=0)
                layers[f"{phase}:{name}_{layer}"] = dict(counts=row) if ebb else row
    return dict(backend=backend, workload=work, config=dict(frequency_hz=100), layers=layers,
                phases={p: dict(configured_word_coverage_complete=True) for p in (["prefill", "decode"] if d else ["prefill"])})


class ProfileBridgeTests(unittest.TestCase):
    def test_independent_long_sequence_work_and_gqa_traffic(self):
        profile = build_profile(fixture())
        row = estimate(profile, 8, 4, external_bandwidth=80)["cases"]["online_mapped"]
        # Per-layer linear params 144; both attention GEMMs: 2*4*S*S MACs.
        self.assertAlmostEqual(row["phases"]["prefill"]["compute_seconds"], (144*8+8*8**2)*3/100)
        self.assertAlmostEqual(row["phases"]["decode"]["compute_seconds"], (144*4+8*(4*8+4*5//2))*9/100)
        # 2 layers, packed W4; read/write 2 physical KV elements per head, 1 KV head.
        self.assertEqual(row["phases"]["prefill"]["modeled_transfer_bytes"], 144+2*2*2*8*2)
        self.assertEqual(row["phases"]["decode"]["modeled_transfer_bytes"], 144*4+2*2*2*(4*8+4*3//2)+2*2*2*4)

    def test_same_shape_roundtrip_frequency_and_paper_calibration(self):
        summary = fixture()
        profile = build_profile(summary)
        base = estimate(profile, 2, 2, external_bandwidth=10)
        high = estimate(profile, 2, 2, external_bandwidth=10, frequency_hz=200, paper_speedup=2)
        for phase in ("prefill", "decode"):
            expected = sum(r["cycles"]["mapped_tiles"] for k, r in summary["layers"].items() if k.startswith(phase+":"))/100
            self.assertAlmostEqual(base["cases"]["online_mapped"]["phases"][phase]["compute_seconds"], expected)
            new = high["cases"]["online_mapped"]["phases"][phase]
            self.assertAlmostEqual(new["compute_seconds"], expected/2)
            self.assertEqual(new["transfer_seconds"], base["cases"]["online_mapped"]["phases"][phase]["transfer_seconds"])
            self.assertAlmostEqual(high["cases"]["paper_speedup"]["phases"][phase]["compute_seconds"],
                                   high["cases"]["dense"]["phases"][phase]["compute_seconds"]/2)

    def test_per_layer_overlap_and_missing_e2e(self):
        result = estimate(build_profile(fixture()), 2, 2, external_bandwidth=2)
        row = result["cases"]["online_mapped"]
        # q_proj prefill at each layer: compute 0.32/0.64, IO=8/2=4.
        # Every layer/op must be independently merged, not global max(total compute, total IO).
        self.assertGreaterEqual(row["gemm_and_io_seconds"]["full_overlap"],
                                max(sum(p["compute_seconds"] for p in row["phases"].values()),
                                    sum(p["transfer_seconds"] for p in row["phases"].values())))
        self.assertIsNone(row["conditional_e2e_seconds"]["full_overlap"])
        missing = estimate(build_profile(fixture("bitlet")), 2, 2)
        self.assertIsNone(missing["cases"]["online_mapped"]["gemm_and_io_seconds"]["full_overlap"])

    def test_complete_target_other_latency_and_materialized_traffic(self):
        profile = build_profile(fixture())
        other = dict(schema_version=1, includes_lm_head=True, prefill_seconds=.2,
                     decode_step_seconds=[.1]*4, workload=dict(prefill_length=8, decode_steps=4))
        row = estimate(profile, 8, 4, external_bandwidth=80, other=other)["cases"]["online_mapped"]
        self.assertAlmostEqual(row["conditional_e2e_seconds"]["full_overlap"], row["gemm_and_io_seconds"]["full_overlap"]+.6)
        more = estimate(profile, 8, 4, external_bandwidth=80, traffic_mode="materialized_once")["cases"]["online_mapped"]
        self.assertGreater(more["phases"]["prefill"]["modeled_transfer_bytes"], row["phases"]["prefill"]["modeled_transfer_bytes"])
        other["workload"]["prefill_length"] = 2
        with self.assertRaisesRegex(ValueError, "does not match"):
            estimate(profile, 8, 4, other=other)
        other.pop("workload")
        with self.assertRaisesRegex(ValueError, "target workload"):
            estimate(profile, 8, 4, other=other)

    def test_incomplete_profiles_counts_and_invalid_coefficients(self):
        summary = fixture()
        for mutation in (lambda s: s["workload"].update(status="interrupted"),
                         lambda s: s["layers"].pop("prefill:q_proj_0"),
                         lambda s: s["layers"]["decode:q_proj_0"].update(calls=1),
                         lambda s: s["layers"]["prefill:q_proj_0"].update(dense_equivalent_operations=1)):
            bad = deepcopy(summary)
            mutation(bad)
            with self.assertRaises(ValueError): build_profile(bad)
        profile = build_profile(summary)
        profile["coefficients"]["prefill:q:0"]["cycles_per_mac"]["online_mapped"] = float("nan")
        with self.assertRaises(ValueError): estimate(profile)
        profile = build_profile(fixture(d=0))
        with self.assertRaisesRegex(ValueError, "Decode estimation"):
            estimate(profile, decode_steps=1)
        self.assertNotIn("decode", estimate(profile, decode_steps=0)["cases"]["online_mapped"]["phases"])

    def test_ebb_width_coverage_and_two_mappings(self):
        summary = fixture("multcim_ebb_bounds")
        summary["phases"]["prefill"]["configured_word_coverage_complete"] = False
        profile = build_profile(summary)
        with self.assertRaisesRegex(ValueError, "coverage incomplete"): estimate(profile)
        result = estimate(profile, 2, 2, allow_incomplete_word_coverage=True)
        self.assertFalse(result["source_word_coverage_complete"])
        cases = result["cases"]
        self.assertAlmostEqual(cases["online_leading_zero"]["phases"]["decode"]["compute_seconds"],
                               cases["online_mapped"]["phases"]["decode"]["compute_seconds"]*2)

    def test_cli_without_site_packages_and_output_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"bitwave_summary.json").write_text(json.dumps(fixture()))
            cmd = [sys.executable, "-S", "-m", "scripts.estimate_profile_latency", "--profile-dir", str(root),
                   "--prefill-length", "8", "--decode-steps", "4", "--output-dir", str(root/"out"),
                   "--frequency-mhz", "bitwave=250", "--shared-external-bandwidth-gbps", "1000"]
            subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
            doc = json.loads((root/"out/profile_latency_summary.json").read_text())["bitwave"]
            self.assertEqual(doc["frequency_hz"], 250e6)
            self.assertEqual(doc["external_bandwidth_bytes_per_second"], 1e12)
            self.assertEqual(len((root/"out/profile_latency_comparison.csv").read_text().splitlines()), 3)
            second = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Optional model dependencies")
    def test_real_qwen_selected_architecture_profiles_and_roundtrip(self):
        import torch
        import yaml
        from unittest import mock
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
        from scripts.profile_bit_arches import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = yaml.safe_load((ROOT/"config/qwen2_14b_bit_arches_f8i4_256_32.yaml").read_text())
            self.assertEqual(config["prefill_decode_profile"]["prefill_length"], 256)
            self.assertEqual(config["prefill_decode_profile"]["decode_steps"], 32)
            self.assertEqual(config["ebb"]["frequency_hz"], 275e6)
            config["quantization"]["scale_dir"] = str(root/"scales")
            config["calibration"].update(num_samples=1)
            (root/"config.yaml").write_text(yaml.safe_dump(config))
            model = Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=8, intermediate_size=16,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)).eval()
            model.save_pretrained(root/"model")
            PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(16)},
                unk_token="0")), unk_token="0").save_pretrained(root/"model")
            tokens = torch.arange(6).long()
            torch.save(tokens, root/"tokens.pt")
            def loader(*args, **kwargs):
                length = kwargs["seq_length"]
                return [dict(input_ids=tokens[:length][None], attention_mask=torch.ones(1, length, dtype=torch.long))]
            for name in ("ebb", "bitwave"):
                with mock.patch("others.data.CalibrationDataLoader", side_effect=loader), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    main(["--config", str(root/"config.yaml"), "--model-path", str(root/"model"),
                          "--token-file", str(root/"tokens.pt"), "--output-dir", str(root/name), "--device", "cpu",
                          "--prefill-length", "4", "--decode-steps", "2", "--architectures", name, "--exact"])
                self.assertFalse((root/f"{name}/slimllama_summary.json").exists())
                profile = json.loads((root/f"{name}/{name}_profile.json").read_text())
                result = estimate(profile, 4, 2, allow_incomplete_word_coverage=True)
                summary = json.loads((root/f"{name}/{name}_summary.json").read_text())
                for phase in ("prefill", "decode"):
                    expected = (summary["phases"][phase]["counts"]["cycles"]["ideal_balanced"] if name == "ebb" else
                                summary["phases"][phase]["cycles"]["mapped_tiles"])/summary["config"]["frequency_hz"]
                    self.assertAlmostEqual(result["cases"]["online_mapped"]["phases"][phase]["compute_seconds"], expected)


if __name__ == "__main__":
    unittest.main()
