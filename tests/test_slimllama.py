"""Slim-Llama numerical decomposition and independent scalar scheduling checks."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
import yaml
from quant.slimllama import (SlimLlamaConfig, SlimLlamaStats, prepare_weight_plan,
    validate_weight_plan, slut_group_work, measure_slimllama_mapping, traffic_and_latency)
from quant.bitwave import fixed_point_groups
from quant.quant_spec import parse_quant_spec
from quant.stat_manager import QuantStatManager


def scalar_fixed(values, spec):
    fractional = {"e4m3": 9, "e5m2": 16}.get(spec, 0)
    mag = [int(abs(value)*(1 << fractional)) for value in values]
    common = min(((value & -value).bit_length()-1 for value in mag if value), default=0) if fractional else 0
    return [value >> common for value in mag]


def scalar_stage(a, b, spec_a, spec_b, config, mode):
    """Pure Python: explicit planes/barriers, without importing mapping helpers."""
    m, n, k = len(a), len(b), len(a[0])
    group = 7 if mode == "mixed" else 8
    mu = min(m, max(1, config.sbcs//math.ceil(n/config.columns_per_sbc)))
    nu = min(n, config.sbcs//mu*config.columns_per_sbc)
    span = group*config.sluts_per_column
    limit = spec_b if isinstance(spec_b, int) else {"e4m3": 18, "e5m2": 32}[spec_b]
    mapped = dense = 0
    for mr in range(0, m, mu):
        for nr in range(0, n, nu):
            for kr in range(0, k, span):
                per_plane = [0]*limit
                setup_max = dense_max = 0
                for row in a[mr:mr+mu]:
                    for column in b[nr:nr+nu]:
                        for offset in range(0, span, group):
                            start = kr+offset
                            av, bv = row[start:start+group], column[start:start+group]
                            if not av:
                                continue
                            am, bm = scalar_fixed(av, spec_a), scalar_fixed(bv, spec_b)
                            pos = max((v for v, original in zip(am, av) if original >= 0), default=0)
                            neg = max((v for v, original in zip(am, av) if original < 0), default=0)
                            width = max(pos.bit_length(), max(0, neg-1).bit_length())+1
                            passes = 1 if width <= 4 else math.ceil(max(am).bit_length()/3)
                            any_lut = False
                            for bit in range(limit):
                                bits = [(value >> bit) & 1 for value in bm]
                                use_lut = mode == "mixed" and len(bits) >= 3 and all(bits[:3])
                                nnz = sum(bits)
                                cost = (1+math.ceil(sum(bits[3:])/2) if use_lut else math.ceil(nnz/2))
                                if mode == "mixed" and not use_lut and nnz:
                                    cost += config.mode_switch_cycles
                                any_lut |= use_lut
                                per_plane[bit] = max(per_plane[bit], passes*cost)
                            setup_max = max(setup_max, passes*(config.group_setup_cycles+
                                                       any_lut*config.lut_setup_cycles))
                            full_prefix = mode == "mixed" and len(bv) >= 3
                            dc = (1+math.ceil((len(bv)-3)/2) if full_prefix else math.ceil(len(bv)/2))
                            dense_max = max(dense_max, passes*(dc*limit+config.group_setup_cycles+
                                                              full_prefix*config.lut_setup_cycles))
                mapped += sum(per_plane)+setup_max
                dense += dense_max
    return mapped, dense


class SlimLlamaTests(unittest.TestCase):
    def test_clock_rescaling_missing_dram_and_selected_point_export(self):
        a, b = torch.ones(1, 8), torch.ones(2, 8)
        low = SlimLlamaConfig(output_reuse=False, prefill_sample_waves=0)
        high = replace(low, frequency_hz=200e6)
        first = measure_slimllama_mapping(a, b, 4, 4, 8, 2, low)
        second = measure_slimllama_mapping(a, b, 4, 4, 8, 2, high)
        self.assertEqual(first["cycles"], second["cycles"])
        self.assertAlmostEqual(first["compute_seconds"]["mapped_tiles"],
                               4*second["compute_seconds"]["mapped_tiles"])
        traffic, latency = traffic_and_latency(first, a, b, 4, 4, "q_proj", {}, low)
        self.assertGreater(traffic["dram_bytes_minimum"], 0)
        self.assertNotIn("minimum_dram_seconds", latency)
        self.assertNotIn("minimum_traffic_full_overlap_seconds", latency)
        work = dict(status="complete", decode_steps=0, completed_decode_steps=0)
        rest = dict(schema_version=1, includes_lm_head=True, prefill_seconds=.01, decode_step_seconds=[])
        with tempfile.TemporaryDirectory() as tmp:
            for config in (low, replace(high, dram_bytes_per_second=1.6e9)):
                stats = SlimLlamaStats(config)
                for _ in range(2):
                    stats.collect("q_proj", 0, a, b, 4, 4, 8, 2, "prefill", {})
                doc = stats.export(Path(tmp)/"summary.json", work, rest)
                self.assertEqual(doc["phases"]["prefill"]["calls"], 2)
                self.assertEqual(doc["paper_parameters"]["selected_frequency_hz"], config.frequency_hz)
                from scripts.estimate_bitlet_latency import estimate
                if config.dram_bytes_per_second is None:
                    self.assertIsNone(doc["latency"]["per_phase_seconds"])
                    self.assertIsNone(doc["latency"]["gemm_and_io_seconds"])
                    self.assertIsNone(doc["latency"]["conditional_e2e_seconds"])
                    self.assertIn("external DRAM bandwidth at selected operating point", doc["latency"]["missing_costs"])
                    with self.assertRaisesRegex(ValueError, "DRAM bandwidth"):
                        estimate(doc, rest)
                else:
                    self.assertAlmostEqual(doc["phases"]["prefill"]["latency"]["minimum_dram_seconds"],
                                           2*traffic["dram_bytes_minimum"]/1.6e9)
                    self.assertEqual(estimate(doc, rest)["conditional_e2e_seconds"], doc["latency"]["conditional_e2e_seconds"])
                stats.close()

    def test_paper_defaults_shared_quantization_and_invalid_parameters(self):
        config = SlimLlamaConfig()
        self.assertEqual((config.sbcs, config.columns, config.sluts_per_column), (64, 512, 8))
        self.assertEqual((config.frequency_hz, config.sram_bytes, config.dram_bytes_per_second),
                         (50e6, 512000, None))
        self.assertIsNone(config.sram_bytes_per_second)
        root = Path(__file__).resolve().parents[1]
        old = yaml.safe_load((root/"config/qwen2_14b_bitlet_bitwave_f8i4_2048_256.yaml").read_text())
        new = yaml.safe_load((root/"config/qwen2_14b_bit_arches_f8i4_2048_256.yaml").read_text())
        for section, value in old.items():
            self.assertEqual(new[section], value)
        self.assertEqual(SlimLlamaConfig.from_dict(new["slimllama"]), config)
        for fields in (dict(sram_bytes=0), dict(weight_clusters=0), dict(sparse_threshold=1.1),
                       dict(frequency_hz=None), dict(dram_bytes_per_second=float('nan')),
                       dict(lut_setup_cycles=-1), dict(output_reuse=1)):
            with self.assertRaises(ValueError): SlimLlamaConfig(**fields)
        from scripts.profile_ebb import validate_config
        new["quantization"]["weight_scale_granularity"] = "output_channel"
        with self.assertRaisesRegex(ValueError, "scalar"):
            validate_config(new, "slimllama")

    def test_actual_centers_exact_delta_and_int4_extrema(self):
        w = torch.tensor([[-8., 7., -8., 7.], [7., -8., 7., -8.], [0., 0., 0., 0.]])
        matrix = w.t()
        config = SlimLlamaConfig(weight_clusters=1, clustering_features=3)
        plan = prepare_weight_plan(matrix, 4, config, seed=19)
        centers = matrix[:, plan["centers"]]
        residual = matrix-centers[:, plan["assignments"]]
        self.assertEqual(float(residual.abs().max()), 15)
        self.assertTrue(torch.equal(centers[:, plan["assignments"]]+residual, matrix))
        a = torch.tensor([[1., -2., 3., -4.]])
        torch.testing.assert_close(a@matrix, (a@centers)[:, plan["assignments"]]+a@residual)
        self.assertEqual(plan["metadata"]["residual_signed_bits"], 5)
        second = prepare_weight_plan(matrix, 4, config, seed=19)
        self.assertEqual(plan["metadata"]["assignment_sha256"], second["metadata"]["assignment_sha256"])
        mutation = matrix.clone()
        mutation[plan["probe_k"][0], plan["probe_n"][0]] += 1
        with self.assertRaisesRegex(ValueError, "probe changed"): validate_weight_plan(plan, mutation, 4)
        wide = torch.tensor([[-32768., 32752., -32768., 32752.],
                             [32752., -32768., 32752., -32768.]], dtype=torch.float16)
        # Half storage is exact for these codes, but half subtraction overflows.
        got = measure_slimllama_mapping(torch.ones(1, 4), wide, 4, 16, 4, 2,
            replace(config, prefill_sample_waves=0))
        self.assertTrue(math.isfinite(got["cycles"]["mapped_tiles"]))

    def test_buffer_two_nonzeros_and_mixed_register_partition(self):
        config = SlimLlamaConfig()
        a, b = torch.ones(1, 1, 1, 8), torch.ones(1, 1, 1, 8)
        valid = torch.ones_like(a, dtype=torch.bool)
        work = slut_group_work(a, b, 4, 4, valid, valid, config, "buffer")
        self.assertEqual(work["mapped"].item(), 4)
        b[..., 2:] = 0
        work = slut_group_work(a*0, b, 4, 4, valid, valid, config, "buffer")
        self.assertEqual(work["mapped"].item(), 1)  # 75% B zero, no free A-zero skip
        work = slut_group_work(a, b*0, 4, 4, valid, valid, config, "buffer")
        self.assertEqual(work["mapped"].item(), 0)
        a, b, valid = a[..., :7], torch.ones(1, 1, 1, 7), valid[..., :7]
        work = slut_group_work(a, b, 4, 4, valid, valid, config, "mixed")
        self.assertEqual(work["mapped"].item(), 3)  # 3 LUT + 4 Buffer operands
        self.assertEqual(work["observed"]["observed_lut_reads"], 1)
        cold = slut_group_work(a, b, 4, 4, valid, valid, replace(config, lut_setup_cycles=6), "mixed")
        self.assertEqual(cold["mapped"].item(), 9)
        b[..., 0] = 0
        switched = slut_group_work(a, b, 4, 4, valid, valid, replace(config, mode_switch_cycles=2), "mixed")
        self.assertEqual(switched["mapped"].item(), 5)

    def test_fp8_lossless_alignment_wide_activation_and_negative_eight(self):
        a = torch.tensor([[[[2**-9, -448., 1., 0., -0., 2., -4., 0.]]]])
        valid = torch.ones_like(a, dtype=torch.bool)
        magnitude, negative, exponent, _, _, _ = fixed_point_groups(a, "e4m3", valid)
        restored = magnitude.double()*torch.exp2(exponent.double())[..., None]
        restored = torch.where(negative, -restored, restored)
        self.assertTrue(torch.equal(restored, a.double()))
        work = slut_group_work(a, torch.ones_like(a), "e4m3", 4, valid, valid, SlimLlamaConfig(), "buffer")
        self.assertEqual(work["mapped"].item(), 24)  # six exact signed-INT4 passes × four reads
        self.assertEqual(work["observed"]["observed_A_groups_exceeding_int4"], 1)
        a = torch.full_like(a, -8)
        work = slut_group_work(a, torch.ones_like(a), 4, 4, valid, valid, SlimLlamaConfig(), "buffer")
        self.assertEqual(work["mapped"].item(), 4)
        with self.assertRaises(ValueError):
            slut_group_work(a*.1, torch.ones_like(a), "e4m3", 4, valid, valid, SlimLlamaConfig(), "buffer")

    def test_direct_tails_and_dynamic_fp8_match_independent_scalar(self):
        config = SlimLlamaConfig(sbc_clusters=1, sbcs_per_cluster=2, columns_per_sbc=2,
            sluts_per_column=2, output_reuse=False, prefill_sample_waves=0,
            lut_setup_cycles=2, group_setup_cycles=1, mode_switch_cycles=1)
        a = ((torch.arange(3*19).reshape(3, 19)%9)-4).float()
        b = ((torch.arange(7*19).reshape(7, 19)%15)-8).float()
        result = measure_slimllama_mapping(a, b, "e4m3", 4, 19, 7, config)
        mapped, dense = scalar_stage(a.tolist(), b.tolist(), "e4m3", 4, config, "mixed")
        self.assertEqual(result["cycles"]["mapped_tiles"], mapped)
        self.assertEqual(result["cycles"]["dense_tiles"], dense)
        b[:, 0], b[:, 2] = 2**-9, -448.
        aq = a[None, None].expand(1, 2, -1, -1)
        kv = b.t()[None, None].expand(1, 2, -1, -1)
        result = measure_slimllama_mapping(aq, kv, "e4m3", "e4m3", 19, 7, config)
        mapped, dense = scalar_stage(a.tolist(), b.tolist(), "e4m3", "e4m3", config, "buffer")
        self.assertEqual(result["cycles"]["mapped_tiles"], 2*mapped)
        self.assertEqual(result["cycles"]["dense_tiles"], 2*dense)
        self.assertFalse(result["weight_reuse"]["enabled"])

    def test_center_residual_stages_match_scalar_and_can_cost_more_than_reference(self):
        config = SlimLlamaConfig(sbc_clusters=1, sbcs_per_cluster=2, columns_per_sbc=2,
            sluts_per_column=2, weight_clusters=2, clustering_features=17,
            prefill_sample_waves=0, reuse_outputs_per_cycle=4)
        a = ((torch.arange(3*17).reshape(3, 17)%7)-3).float()
        w = ((torch.arange(7*17).reshape(7, 17)%15)-8).float()
        plan = prepare_weight_plan(w.t(), 4, config, seed=7)
        result = measure_slimllama_mapping(a, w, "e4m3", 4, 17, 7, config, weight_plan=plan)
        expected = 0
        for name, stage in result["stages"].items():
            indices = plan["centers"] if name == "centers" else plan["mixed"] if name == "mixed_residuals" else plan["buffer"]
            values = w[indices]
            if stage["residual"]:
                values = values-w[plan["centers"][plan["assignments"][indices]]]
            cycles, _ = scalar_stage(a.tolist(), values.tolist(), "e4m3", 5 if stage["residual"] else 4,
                                     config, stage["mode"])
            self.assertEqual(stage["cycles"], cycles)
            expected += cycles
        expected += math.ceil(3*2/4)+math.ceil(3*5/4)
        self.assertEqual(result["cycles"]["mapped_tiles"], expected)
        self.assertEqual(result["cycles"]["dense_tiles"], scalar_stage(
            a.tolist(), w.tolist(), "e4m3", 4, config, "mixed")[1])
        # Stage barriers, residual width and center reuse can outweigh savings.
        costly = replace(config, mode_switch_cycles=100)
        expensive = measure_slimllama_mapping(a, w, "e4m3", 4, 17, 7, costly, weight_plan=plan)
        self.assertLess(expensive["mapped_compute_speedup"], 1)

    def test_identical_vectors_reuse_and_center_cache_only_once(self):
        config = SlimLlamaConfig(weight_clusters=1, prefill_sample_waves=0)
        a, w = torch.ones(16, 13), torch.ones(64, 13)
        result = measure_slimllama_mapping(a, w, "e4m3", 4, 13, 64, config)
        self.assertEqual(result["stages"]["buffer_residuals"]["cycles"], 0)
        self.assertGreater(result["mapped_compute_speedup"], 1)
        stats = SlimLlamaStats(config)
        with mock.patch("quant.slimllama.prepare_weight_plan", wraps=prepare_weight_plan) as prepared:
            for phase in ("prefill", "decode", "decode"):
                stats.collect("q_proj", 0, a, w, "e4m3", 4, 13, 64, phase, {})
            self.assertEqual(prepared.call_count, 1)
        self.assertEqual(len(stats.weight_plans), 1)
        stats.close()

    def test_sampling_chunk_invariance_capacity_and_transport_overhead(self):
        torch.manual_seed(41)
        a = torch.randint(-7, 8, (9, 81)).float()
        w = torch.randint(-8, 8, (19, 81)).float()
        config = SlimLlamaConfig(sbc_clusters=1, sbcs_per_cluster=2, columns_per_sbc=2,
            sluts_per_column=2, weight_clusters=2, prefill_sample_waves=7, chunk_waves=1,
            sram_bytes=2048, sram_bytes_per_second=3.2e9)
        config = replace(config, dram_bytes_per_second=1.6e9)
        plan = prepare_weight_plan(w.t(), 4, config, seed=3)
        one = measure_slimllama_mapping(a, w, "e4m3", 4, 81, 19, config, seed=31, weight_plan=plan)
        many = measure_slimllama_mapping(a, w, "e4m3", 4, 81, 19, replace(config, chunk_waves=3), seed=31, weight_plan=plan)
        self.assertEqual(one, many)
        self.assertFalse(one["sampling"]["exact"])
        used = (one["memory"]["minimum_tile_bytes"]+
                one["memory"]["rows_per_weight_window"]*one["memory"]["accumulator_bytes_per_window_row"])
        self.assertLessEqual(used, config.sram_bytes)
        traffic, latency = traffic_and_latency(one, a, w, "e4m3", 4, "q_proj", {}, config)
        self.assertEqual(traffic["packed_weight_read_bytes_minimum"], math.ceil(19*81/2))
        self.assertEqual(traffic["center_coefficients_extra_bytes_per_window"], 81)
        self.assertEqual(traffic["cluster_id_bytes_per_window"], 3)
        self.assertGreaterEqual(traffic["dram_bytes_window_schedule"], traffic["dram_bytes_minimum"])
        self.assertAlmostEqual(latency["capacity_window_no_overlap_seconds"],
            latency["mapped_compute_seconds"]+traffic["dram_bytes_window_schedule"]/1.6e9+
            latency["modeled_local_sram_seconds"])
        with self.assertRaisesRegex(ValueError, "SRAM cannot fit"):
            measure_slimllama_mapping(a, w, "e4m3", 4, 81, 19, replace(config, sram_bytes=1), weight_plan=plan)

    def test_three_collectors_share_operands_gqa_and_e2e_schema(self):
        spec = parse_quant_spec("e4m3")
        with tempfile.TemporaryDirectory() as tmp:
            configs = dict(bitlet_config=dict(pe_count=2, group_size=8, decode_sample_waves=0),
                bitwave_config=dict(decode_sample_waves=0, dataflow="SU2"),
                slimllama_config=dict(decode_sample_waves=0, dram_bytes_per_second=1.6e9))
            joint = QuantStatManager(tmp, **configs)
            single = QuantStatManager(tmp, slimllama_config=configs["slimllama_config"])
            a, b = torch.ones(1, 4, 1, 8), torch.ones(1, 4, 8, 9)
            seen = []
            def observe(original):
                def call(*args):
                    seen.append((args[2], args[3]))
                    return original(*args)
                return call
            for manager in (joint, single):
                manager.set_phase("decode"); manager.set_step(0, cache_length_before=8, query_length=1)
                manager.set_attention_context(0, q_heads=4, kv_heads=2, head_dim=8,
                    shared_kv_gqa=True, cache_length_before=8, query_length=1)
            with mock.patch.object(joint.bitlet_stats, "collect", side_effect=observe(joint.bitlet_stats.collect)), \
                 mock.patch.object(joint.bitwave_stats, "collect", side_effect=observe(joint.bitwave_stats.collect)), \
                 mock.patch.object(joint.slimllama_stats, "collect", side_effect=observe(joint.slimllama_stats.collect)):
                records = joint.collect_quant_activation("qk_matmul", 0, a, a, b, spec, spec, 8, 4, 8, 9)
            only = single.collect_quant_activation("qk_matmul", 0, a, a, b, spec, spec, 8, 4, 8, 9)
            for tensors in seen:
                self.assertIs(tensors[0], a); self.assertIs(tensors[1], b)
            self.assertEqual(records["slimllama"], only)
            self.assertFalse(only["weight_reuse"]["enabled"])
            self.assertEqual(only["traffic"]["fp8_kv_read_bytes_minimum"], 128)
            self.assertEqual(only["traffic"]["fp8_kv_append_bytes"], 16)
            workload = dict(status="complete", decode_steps=1, completed_decode_steps=1)
            rest = dict(schema_version=1, includes_lm_head=True, prefill_seconds=.01, decode_step_seconds=[.001])
            doc = joint.export_slimllama_stats(Path(tmp)/"summary.json", workload, rest)
            from scripts.estimate_bitlet_latency import estimate
            self.assertEqual(estimate(doc, rest)["conditional_e2e_seconds"], doc["latency"]["conditional_e2e_seconds"])
            self.assertIn("internal SRAM bandwidth", doc["latency"]["missing_costs"])
            interrupted = dict(workload, status="interrupted")
            self.assertIsNone(joint.export_slimllama_stats(Path(tmp)/"partial.json", interrupted, rest)["latency"]["conditional_e2e_seconds"])
            with self.assertRaises(ValueError): single.export_cim_stats(Path(tmp)/"bad.json")
            joint.close(); single.close()

    @unittest.skipUnless(importlib.util.find_spec("transformers") and importlib.util.find_spec("datasets"),
                         "Install requirements-model.txt")
    def test_real_three_way_qwen_cli_single_pass_and_interrupted_cleanup(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from scripts.profile_bit_arches import main
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            config = yaml.safe_load((root/"config/qwen2_14b_bit_arches_f8i4_2048_256.yaml").read_text())
            config["quantization"]["scale_dir"] = str(tmp/"scales")
            config["quantization"]["calibration_policy"] = dict(default="recalibrate")
            config["calibration"].update(seq_length=8192, min_text_tokens=8192, num_samples=1)
            config["slimllama"]["weight_clusters"] = 2
            torch.manual_seed(29)
            model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)).eval()
            model_dir = tmp/"model"
            model.save_pretrained(model_dir)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
                WordLevel({str(i): i for i in range(64)}, unk_token="0")), unk_token="0")
            tokenizer.save_pretrained(model_dir)
            tokens = torch.arange(16).long()
            torch.save(tokens, tmp/"tokens.pt")
            file = tmp/"config.yaml"
            file.write_text(yaml.safe_dump(config))
            loader_calls, forwards = [], []
            def loader(*args, **kwargs):
                loader_calls.append(kwargs)
                length = kwargs["seq_length"]
                return [dict(input_ids=tokens[:length][None], attention_mask=torch.ones(1, length, dtype=torch.long))]
            from quant.qwen_wrapper import wrap_qwen_model
            def instrument(model, *args, **kwargs):
                model = wrap_qwen_model(model, *args, **kwargs)
                model.model.register_forward_hook(lambda *args: forwards.append(1))
                return model
            args = ["--config", str(file), "--model-path", str(model_dir), "--token-file", str(tmp/"tokens.pt"),
                    "--device", "cpu", "--prefill-length", "8", "--decode-steps", "2", "--exact"]
            with mock.patch("others.data.CalibrationDataLoader", side_effect=loader), \
                 mock.patch("quant.model_wrapper.wrap_model_by_family", side_effect=instrument), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                main([*args, "--output-dir", str(tmp/"joint")])
            self.assertEqual(len(loader_calls), 1)
            self.assertEqual(loader_calls[0]["seq_length"], 8)
            self.assertEqual(len(forwards), 4)
            names = ("bitlet", "bitwave", "slimllama")
            joint_docs = {}
            for name in names:
                doc = json.loads((tmp/f"joint/{name}_summary.json").read_text())
                joint_docs[name] = doc
                self.assertEqual(doc["phases"]["prefill"]["calls"], 9)
                self.assertEqual(doc["phases"]["decode"]["calls"], 18)
                self.assertEqual(doc["workload"]["status"], "complete")
                self.assertEqual(len((tmp/f"joint/{name}_trace.jsonl").read_text().splitlines()), 27)
            comparison = json.loads((tmp/"joint/bit_arch_comparison.json").read_text())
            self.assertEqual(comparison["workload"]["profile_architectures"], list(names))
            slim = joint_docs["slimllama"]
            self.assertEqual(slim["config"]["frequency_hz"], 50e6)
            self.assertEqual(slim["paper_parameters"]["selected_frequency_hz"], 50e6)
            self.assertIsNone(slim["latency"]["gemm_and_io_seconds"])
            self.assertEqual(len(slim["weight_preprocessing"]), 7)
            self.assertEqual(slim["phases"]["prefill"]["output_reuse_calls"], 7)
            for name in names:
                command = [sys.executable, "-m", f"scripts.profile_{name}", *args, "--skip-calibration",
                           "--output-dir", str(tmp/name)]
                process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
                self.assertEqual(process.returncode, 0, process.stderr)
                solo = json.loads((tmp/f"{name}/{name}_summary.json").read_text())
                self.assertEqual(solo["phases"], joint_docs[name]["phases"])
            rest = dict(schema_version=1, includes_lm_head=True, prefill_seconds=.01, decode_step_seconds=[.001, .002])
            (tmp/"rest.json").write_text(json.dumps(rest))
            command = [sys.executable, "-m", "scripts.profile_bit_arches", *args, "--skip-calibration", "--greedy-decode",
                       "--bitwave-dram-bandwidth-gbps", "12.8", "--slimllama-dram-bandwidth-gbps", "1.6",
                       "--other-latency-json", str(tmp/"rest.json"),
                       "--output-dir", str(tmp/"greedy")]
            process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            from scripts.estimate_bitlet_latency import estimate
            for name in names:
                doc = json.loads((tmp/f"greedy/{name}_summary.json").read_text())
                self.assertEqual(doc["workload"]["decode_mode"], "greedy")
                self.assertEqual(estimate(doc, rest)["conditional_e2e_seconds"], doc["latency"]["conditional_e2e_seconds"])
            managers = []
            def create_manager(*args, **kwargs):
                manager = QuantStatManager(*args, **kwargs)
                managers.append(manager)
                return manager
            with mock.patch("scripts.profile_ebb.QuantStatManager", side_effect=create_manager), \
                 mock.patch("quant.slimllama.SlimLlamaStats.collect", side_effect=RuntimeError("forced Slim-Llama failure")), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "forced Slim-Llama failure"):
                    main([*args, "--skip-calibration", "--output-dir", str(tmp/"interrupted")])
            for name in names:
                doc = json.loads((tmp/f"interrupted/{name}_summary.json").read_text())
                self.assertEqual(doc["workload"]["status"], "interrupted")
                self.assertIsNone(getattr(managers[-1], f"{name}_stats").trace)
            # Explicit subset keeps the former two-collector workflow available.
            command = [sys.executable, "-m", "scripts.profile_bit_arches", *args, "--skip-calibration",
                       "--architectures", "bitlet", "bitwave", "--output-dir", str(tmp/"pair")]
            process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertFalse((tmp/"pair/slimllama_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
