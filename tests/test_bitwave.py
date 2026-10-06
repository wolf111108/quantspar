import importlib.util
import io
import json
import math
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
import yaml
from quant.bitlet import BitletConfig, group_column_work
from quant.bitwave import (BitWaveConfig, DATAFLOWS, column_work, fixed_point_groups,
                           measure_bitwave_mapping, traffic_and_latency)
from quant.quant_spec import parse_quant_spec
from quant.stat_manager import QuantStatManager


def scalar_fixed(values, spec):
    fractional = {"e4m3": 9, "e5m2": 16}.get(spec, 0)
    integers = [int(abs(value)*(1 << fractional)) for value in values]
    common = (min((value & -value).bit_length()-1 for value in integers if value)
              if fractional and any(integers) else 0)
    return [value >> common for value in integers], common-fractional if any(integers) else 0


def scalar_schedule(x, weights, spec_a, spec_b, su, setup=0):
    group, mu, nu, _, _ = DATAFLOWS[su]
    m, k, n = len(x[0]), len(x[0][0]), len(weights[0])
    limit = spec_b if isinstance(spec_b, int) else {"e4m3": 18, "e5m2": 32}[spec_b]
    mapped = dense = 0
    for a, b in zip(x, weights):
        for mr in range(0, m, mu):
            for nr in range(0, n, nu):
                for kr in range(0, k, group):
                    a_passes, columns = [], []
                    for row in a[mr:mr+mu]:
                        values = row[kr:kr+group]
                        magnitude, _ = scalar_fixed(values, spec_a)
                        positive = max((v for v, value in zip(magnitude, values) if value >= 0), default=0)
                        negative = max((v for v, value in zip(magnitude, values) if value < 0), default=0)
                        width = max(positive.bit_length(), max(0, negative-1).bit_length())+1
                        a_passes.append(1 if width <= 8 else math.ceil(max(magnitude).bit_length()/7))
                    for row in b[nr:nr+nu]:
                        magnitude, _ = scalar_fixed(row[kr:kr+group], spec_b)
                        union = 0
                        for value in magnitude:
                            union |= value
                        columns.append(union.bit_count())
                    mapped += max(a_passes)*max(columns)+setup
                    dense += max(a_passes)*limit+setup
    return mapped, dense


class BitWaveTests(unittest.TestCase):
    def test_paper_parameters_and_shared_quant_config(self):
        config = BitWaveConfig()
        self.assertEqual((config.bce_count, config.frequency_hz), (512, 250e6))
        self.assertEqual(config.activation_sram_bytes, 262144)
        self.assertEqual(config.weight_sram_bytes, 262144)
        self.assertIsNone(config.dram_bytes_per_second)
        self.assertFalse(config.bit_flip)
        self.assertEqual(DATAFLOWS["SU4"], (8, 1, 128, 1024, 64))
        root = Path(__file__).resolve().parents[1]
        old = yaml.safe_load((root/"config/qwen2_14b_bitlet_f8i4_2048_256.yaml").read_text())
        joint = yaml.safe_load((root/"config/qwen2_14b_bitlet_bitwave_f8i4_2048_256.yaml").read_text())
        for section in old:
            self.assertEqual(joint[section], old[section])
        self.assertEqual(BitWaveConfig.from_dict(joint["bitwave"]), config)
        for values in (dict(bit_flip=True), dict(dataflow="SU7"), dict(bce_count=32),
                       dict(dram_bytes_per_second=-1), dict(chunk_waves=0)):
            with self.assertRaises(ValueError): BitWaveConfig(**values)

    def test_column_rule_sign_and_int4_minimum_are_distinct_from_bitlet(self):
        a, b = torch.ones(1, 1, 8), torch.ones(1, 1, 8)
        valid = torch.ones_like(a, dtype=torch.bool)
        result = column_work(a, b, 8, 4, valid, valid)
        self.assertEqual(result["cycles"].item(), 1)
        bitlet, _, _ = group_column_work(a, b, 8, 4, BitletConfig(group_size=8))
        self.assertEqual(bitlet.item(), 8)
        self.assertEqual(result["compressed_B_bytes"].item(), 2)  # 8-bit index + data column
        result = column_work(a, -b, 8, 4, valid, valid)
        self.assertEqual(result["cycles"].item(), 1)
        self.assertEqual(result["compressed_B_bytes"].item(), 3)  # sign column is stored
        result = column_work(a, b*(-8), 8, 4, valid, valid)
        self.assertEqual(result["observed"]["B_magnitude_width_histogram"][4], 1)
        self.assertEqual(result["cycles"].item(), 1)
        zero_a = column_work(a*0, b, 8, 4, valid, valid)
        self.assertEqual(zero_a["cycles"].item(), 1)
        zero_b = column_work(a, b*0, 8, 4, valid, valid)
        self.assertEqual(zero_b["cycles"].item(), 0)
        self.assertEqual(zero_b["compressed_B_bytes"].item(), 1)

    def test_lossless_fp8_alignment_hidden_one_and_int8_slices(self):
        values = torch.tensor([[[2**-9, -448., 1., 0., -0., 2., -4., 0.]]])
        valid = torch.ones_like(values, dtype=torch.bool)
        magnitude, negative, exponent, _, _, _ = fixed_point_groups(values, "e4m3", valid)
        restored = magnitude.double()*torch.exp2(exponent.double()).unsqueeze(-1)
        restored = torch.where(negative, -restored, restored)
        self.assertTrue(torch.equal(restored, values.double()))
        result = column_work(values, torch.ones_like(values), "e4m3", "e4m3", valid, valid)
        self.assertEqual(result["cycles"].item(), 3)
        self.assertEqual(result["observed"]["nonzero_columns_histogram"][1], 1)
        self.assertEqual(result["observed"]["observed_A_groups_exceeding_int8"], 1)
        for source in (torch.tensor([[[-128., 1.]]]), torch.tensor([[[-129., 1.]]])):
            spec = 8 if source[0, 0, 0] == -128 else "e5m2"
            # -129 is off the E5M2 grid: reject it rather than rounding silently.
            if spec == "e5m2":
                with self.assertRaises(ValueError):
                    fixed_point_groups(source, spec, torch.ones_like(source, dtype=torch.bool))
            else:
                got = column_work(source, torch.ones_like(source), spec, 4,
                                  torch.ones_like(source, dtype=torch.bool),
                                  torch.ones_like(source, dtype=torch.bool))
                self.assertEqual(got["cycles"].item(), 1)
        with self.assertRaises(ValueError):
            fixed_point_groups(values*1.1, "e4m3", valid)

    def test_all_six_exact_dataflows_match_independent_scalar_with_tails(self):
        x = ((torch.arange(3*17).reshape(3, 17) % 7)-3).float()
        w = ((torch.arange(35*17).reshape(35, 17) % 15)-8).float()
        for su in DATAFLOWS:
            config = BitWaveConfig(dataflow=su, prefill_sample_waves=0, chunk_waves=1)
            result = measure_bitwave_mapping(x, w, "e4m3", 4, 17, 35, config)
            expected = scalar_schedule([x.tolist()], [w.tolist()], "e4m3", 4, su)
            self.assertEqual(result["cycles"]["mapped_tiles"], expected[0])
            self.assertEqual(result["cycles"]["dense_tiles"], expected[1])
            second = measure_bitwave_mapping(x, w, "e4m3", 4, 17, 35,
                                             replace(config, chunk_waves=7))
            self.assertEqual(result, second)
            self.assertLessEqual(result["cycles"]["ideal_pe_balance"], result["cycles"]["mapped_tiles"])
        attention_a = x.unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1)
        attention_b = w.t().unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1)
        result = measure_bitwave_mapping(attention_a, attention_b, "e4m3", 4, 17, 35,
                                         BitWaveConfig(dataflow="SU2", prefill_sample_waves=0))
        expected = scalar_schedule([x.tolist()]*2, [w.tolist()]*2, "e4m3", 4, "SU2")
        self.assertEqual(result["cycles"]["mapped_tiles"], expected[0])
        floating_w = w.clone()
        floating_w[:, 0] = 2**-9
        floating_w[:, 2] = 448.
        floating_b = floating_w.t().unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1)
        result = measure_bitwave_mapping(attention_a, floating_b, "e4m3", "e4m3", 17, 35,
                                         BitWaveConfig(dataflow="SU2", prefill_sample_waves=0))
        expected = scalar_schedule([x.tolist()]*2, [floating_w.tolist()]*2, "e4m3", "e4m3", "SU2")
        self.assertEqual(result["cycles"]["mapped_tiles"], expected[0])
        self.assertGreater(result["observed"]["observed_B_groups_exceeding_native_magnitude"], 0)
        with self.assertRaises(ValueError):
            measure_bitwave_mapping(x, w, "e4m3", 4, 17, 35,
                BitWaveConfig(activation_sram_bytes=1, weight_sram_bytes=1))

    def test_sampling_reproducibility_selection_and_transport_missing_bandwidth(self):
        torch.manual_seed(5)
        x = torch.randint(-6, 7, (24, 129)).float()
        w = torch.randint(-8, 8, (64, 129)).float()
        config = BitWaveConfig(prefill_sample_waves=10, chunk_waves=1)
        result = measure_bitwave_mapping(x, w, "e4m3", 4, 129, 64, config, seed=31)
        other = measure_bitwave_mapping(x, w, "e4m3", 4, 129, 64, replace(config, chunk_waves=4), seed=31)
        self.assertEqual(result, other)
        selected = result["layout"]["dataflow"]
        self.assertEqual(result["selection_compute_plus_local_streaming_seconds"],
            min(row["selection_compute_plus_local_streaming_seconds"] for row in result["candidates"].values()))
        fixed = measure_bitwave_mapping(x, w, "e4m3", 4, 129, 64, replace(config, dataflow=selected), seed=31)
        self.assertEqual(result["cycles"], fixed["cycles"])
        traffic, latency = traffic_and_latency(result, x, w, "e4m3", 4, "q_proj", {}, config)
        self.assertEqual(traffic["packed_weight_read_bytes_minimum"], 4128)
        self.assertNotIn("paper_equation5_resident_seconds", latency)
        with_bw = replace(config, dram_bytes_per_second=12.8e9)
        traffic, latency = traffic_and_latency(result, x, w, "e4m3", 4, "q_proj", {}, with_bw)
        self.assertAlmostEqual(latency["dram_seconds"], (
            traffic["activation_read_bytes_minimum"]+traffic["B_read_bytes_minimum"]+
            traffic["output_write_bytes"])/12.8e9)
        self.assertGreaterEqual(latency["streaming_no_overlap_seconds"],
                                latency["paper_equation5_resident_seconds"])

    def test_manager_fans_out_same_tensors_gqa_and_standalone_equivalence(self):
        spec = parse_quant_spec("e4m3")
        with tempfile.TemporaryDirectory() as tmp:
            configs = dict(bitlet_config=dict(pe_count=2, group_size=8,
                decode_sample_waves=0), bitwave_config=dict(decode_sample_waves=0, dataflow="SU2"))
            both = QuantStatManager(tmp, **configs)
            single = QuantStatManager(tmp, bitwave_config=configs["bitwave_config"])
            a, b = torch.ones(1, 4, 1, 8), torch.ones(1, 4, 8, 9)
            for manager in (both, single):
                manager.set_phase("decode"); manager.set_step(0, cache_length_before=8, query_length=1)
                manager.set_attention_context(0, q_heads=4, kv_heads=2, head_dim=8,
                    shared_kv_gqa=True, cache_length_before=8, query_length=1)
            seen = []
            def observe(original):
                def callback(*args):
                    seen.append((args[2], args[3]))
                    return original(*args)
                return callback
            with mock.patch.object(both.bitlet_stats, "collect", side_effect=observe(both.bitlet_stats.collect)), \
                 mock.patch.object(both.bitwave_stats, "collect", side_effect=observe(both.bitwave_stats.collect)):
                records = both.collect_quant_activation("qk_matmul", 0, a, a, b, spec, spec, 8, 4, 8, 9)
            standalone = single.collect_quant_activation("qk_matmul", 0, a, a, b, spec, spec, 8, 4, 8, 9)
            self.assertIs(seen[0][0], seen[1][0])
            self.assertIs(seen[0][1], seen[1][1])
            self.assertEqual(records["bitwave"], standalone)
            self.assertEqual(records["bitwave"]["traffic"]["fp8_kv_read_bytes_minimum"], 128)
            workload = dict(status="complete", decode_steps=1, completed_decode_steps=1)
            doc = both.export_bitwave_stats(Path(tmp)/"summary.json", workload)
            self.assertIsNone(doc["latency"]["gemm_and_io_seconds"])
            from scripts.estimate_bitlet_latency import estimate
            rest = dict(schema_version=1, includes_lm_head=True, prefill_seconds=0., decode_step_seconds=[.1])
            with self.assertRaises(ValueError): estimate(doc, rest)
            with self.assertRaises(ValueError): single.export_cim_stats(Path(tmp)/"bad.json")
            both.close(); single.close()

    @unittest.skipUnless(importlib.util.find_spec("transformers") and importlib.util.find_spec("datasets"),
                         "Install requirements-model.txt")
    def test_joint_qwen_one_calibration_one_inference_and_cli_both_decode_modes(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from scripts.profile_bit_arches import main
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = yaml.safe_load((root/"config/qwen2_14b_bitlet_bitwave_f8i4_2048_256.yaml").read_text())
            config["quantization"]["scale_dir"] = str(tmp/"scales")
            config["quantization"]["calibration_policy"] = {"default": "recalibrate"}
            config["calibration"].update(seq_length=8192, min_text_tokens=8192, num_samples=1)
            model_dir = tmp/"model"
            torch.manual_seed(17)
            model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)).eval()
            model.save_pretrained(model_dir)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
                WordLevel({str(i): i for i in range(64)}, unk_token="0")), unk_token="0")
            tokenizer.save_pretrained(model_dir)
            tokens = torch.arange(16).long()
            torch.save(tokens, tmp/"tokens.pt")
            config_file = tmp/"config.yaml"
            config_file.write_text(yaml.safe_dump(config))
            loader_calls = []
            def loader(*args, **kwargs):
                loader_calls.append(kwargs)
                length = kwargs["seq_length"]
                return [{"input_ids": tokens[:length].unsqueeze(0),
                         "attention_mask": torch.ones(1, length, dtype=torch.long)}]
            forwards = []
            from quant.qwen_wrapper import wrap_qwen_model
            def instrument(model, *args, **kwargs):
                model = wrap_qwen_model(model, *args, **kwargs)
                model.model.register_forward_hook(lambda *args: forwards.append(1))
                return model
            args = ["--config", str(config_file), "--model-path", str(model_dir),
                    "--token-file", str(tmp/"tokens.pt"), "--device", "cpu",
                    "--prefill-length", "8", "--decode-steps", "2", "--exact"]
            with mock.patch("others.data.CalibrationDataLoader", side_effect=loader), \
                 mock.patch("quant.model_wrapper.wrap_model_by_family", side_effect=instrument), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                main([*args, "--output-dir", str(tmp/"joint")])
            self.assertEqual(len(loader_calls), 1)
            self.assertEqual(loader_calls[0]["seq_length"], 8)
            self.assertEqual(len(forwards), 4)  # calibration + prefill + 2 decode, not duplicated
            joint = {}
            for name in ("bitlet", "bitwave"):
                doc = json.loads((tmp/f"joint/{name}_summary.json").read_text())
                joint[name] = doc
                self.assertEqual(doc["phases"]["prefill"]["calls"], 9)
                self.assertEqual(doc["phases"]["decode"]["calls"], 18)
                self.assertEqual(doc["workload"]["status"], "complete")
                self.assertEqual(len((tmp/f"joint/{name}_trace.jsonl").read_text().splitlines()), 27)
            comparison = json.loads((tmp/"joint/bit_arch_comparison.json").read_text())
            self.assertTrue(comparison["shared_quantized_operands"])
            self.assertEqual(comparison["workload"]["profile_architectures"], ["bitlet", "bitwave"])
            self.assertEqual(joint["bitlet"]["workload"], joint["bitwave"]["workload"])
            for name in ("bitlet", "bitwave"):
                command = [sys.executable, "-m", f"scripts.profile_{name}", *args,
                           "--skip-calibration", "--output-dir", str(tmp/name)]
                process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
                self.assertEqual(process.returncode, 0, process.stderr)
                single = json.loads((tmp/f"{name}/{name}_summary.json").read_text())
                self.assertEqual(single["phases"], joint[name]["phases"])
            rest = dict(schema_version=1, includes_lm_head=True, prefill_seconds=.01,
                        decode_step_seconds=[.001, .002])
            (tmp/"rest.json").write_text(json.dumps(rest))
            command = [sys.executable, "-m", "scripts.profile_bit_arches", *args, "--skip-calibration",
                       "--greedy-decode", "--bitwave-dram-bandwidth-gbps", "12.8",
                       "--other-latency-json", str(tmp/"rest.json"), "--output-dir", str(tmp/"greedy")]
            process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            from scripts.estimate_bitlet_latency import estimate
            for name in ("bitlet", "bitwave"):
                doc = json.loads((tmp/f"greedy/{name}_summary.json").read_text())
                self.assertEqual(doc["workload"]["decode_mode"], "greedy")
                self.assertEqual(estimate(doc, rest)["conditional_e2e_seconds"],
                                 doc["latency"]["conditional_e2e_seconds"])
            managers = []
            def manager_factory(*args, **kwargs):
                manager = QuantStatManager(*args, **kwargs)
                managers.append(manager)
                return manager
            with mock.patch("scripts.profile_ebb.QuantStatManager", side_effect=manager_factory), \
                 mock.patch("quant.bitwave.BitWaveStats.collect", side_effect=RuntimeError("forced collector failure")), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "forced collector failure"):
                    main([*args, "--skip-calibration", "--output-dir", str(tmp/"interrupted")])
            for name in ("bitlet", "bitwave"):
                doc = json.loads((tmp/f"interrupted/{name}_summary.json").read_text())
                self.assertEqual(doc["workload"]["status"], "interrupted")
                self.assertIsNone(doc["latency"]["conditional_e2e_seconds"])
                self.assertIsNone(getattr(managers[-1], f"{name}_stats").trace)


if __name__ == "__main__":
    unittest.main()
