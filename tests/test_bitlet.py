"""Independent Bitlet BCE reference, transport checks and real tiny-Qwen CLI."""
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
from quant import QuantStatManager
from quant.bitlet import (BitletConfig, BitletStats, OPERATORS, group_column_work,
                         measure_bitlet_mapping, traffic_and_latency,
                         validate_other_latency)
from quant.quant_spec import parse_quant_spec


def scalar_group(a, b, group_size, integer=False):
    """Python reference independently normalizes each product exponent."""
    pairs = [(av, bv) for av, bv in zip(a, b)]
    dense = len(pairs)
    if integer:
        aligned = [abs(int(bv)) for _, bv in pairs]
    elif pairs:
        exponents = [(math.frexp(abs(av))[1]-1 if av else -126) +
                     (math.frexp(abs(bv))[1]-1 if bv else -126) for av, bv in pairs]
        high = max(exponents)
        aligned = [int(math.frexp(abs(bv))[0]*2**24) >> (high-exponent)
                   for (_, bv), exponent in zip(pairs, exponents)]
    else:
        aligned = []
    cycles = max((sum((value >> lane) & 1 for value in aligned)
                  for lane in range(24)), default=0)
    return cycles, dense


def scalar_schedule(a, b, pe, group, integer=False):
    """A [operands,M,K], B [operands,N,K]; synchronized output-column tiles."""
    sparse = dense = 0
    for x, weights in zip(a, b):
        for row in x:
            for start in range(0, len(weights), pe):
                for kg in range(0, len(row), group):
                    costs = [scalar_group(row[kg:kg+group], w[kg:kg+group], group, integer)
                             for w in weights[start:start+pe]]
                    sparse += max(v[0] for v in costs)
                    dense += max(v[1] for v in costs)
    return sparse, dense


class BitletTests(unittest.TestCase):
    def test_paper_defaults_do_not_invent_buffer_capacity_or_peak_tops(self):
        config = BitletConfig()
        self.assertEqual((config.pe_count, config.group_size, config.frequency_hz),
                         (32, 64, 1e9))
        self.assertEqual(config.mantissa_bits, 24)
        self.assertEqual(config.activation_dma_bytes_per_second, 12.8e9)
        self.assertEqual(config.weight_dma_bytes_per_second, 12.8e9)
        self.assertEqual(config.local_buffer_bytes_per_second, 25.6e9)
        self.assertIsNone(config.local_buffer_bytes)
        with self.assertRaises(ValueError): BitletConfig(pe_count=True)
        with self.assertRaises(ValueError): BitletConfig(prefill_sample_waves=-1)
        with self.assertRaises(ValueError): BitletConfig(frequency_hz=float("nan"))
        with self.assertRaises(ValueError): BitletConfig(mantissa_bits=8)
        root = Path(__file__).resolve().parents[1]
        import yaml
        from scripts.profile_ebb import validate_config
        doc = yaml.safe_load((root/"config/qwen2_14b_bitlet_f8i4_2048_256.yaml").read_text())
        validate_config(doc, "bitlet")
        self.assertEqual(BitletConfig.from_dict(doc["bitlet"]), config)
        self.assertEqual(doc["model"]["num_layers"], 48)
        self.assertEqual(doc["prefill_decode_profile"]["decode_steps"], 256)

    def test_hidden_one_sign_and_negative_eight_are_not_skipped(self):
        config = BitletConfig(group_size=4)
        ones = torch.ones(1, 4)
        cycles, dense, counts = group_column_work(ones, ones, "e4m3", "e4m3", config)
        self.assertEqual(cycles.item(), 4)
        self.assertEqual(dense.item(), 4)
        self.assertEqual(counts["column_population_histogram"][4], 1)
        self.assertEqual(counts["observed_source_B_one_bits"], 4*3)
        b = torch.tensor([[-8., -1., 0., 1.]])
        got, _, counts = group_column_work(ones, b, "e4m3", 4, config)
        expected, _ = scalar_group(ones[0].tolist(), b[0].tolist(), 4)
        self.assertEqual(got.item(), expected)
        self.assertEqual(counts["observed_source_B_one_bits"], 6)
        self.assertEqual(counts["observed_negative_products"], 2)
        self.assertEqual(counts["observed_zero_products"], 1)
        negative, _, _ = group_column_work(ones, -ones, "e4m3", "e4m3", config)
        self.assertEqual(negative.item(), cycles.item())

    def test_product_exponent_matching_truncation_and_subnormals(self):
        config = BitletConfig(group_size=4)
        a = torch.tensor([[448., 2**-9, 1., 0.]])
        b = torch.tensor([[448., 2**-9, 2., -1.]])
        got, _, counts = group_column_work(a, b, "e4m3", "e4m3", config)
        self.assertEqual(got.item(), scalar_group(a[0].tolist(), b[0].tolist(), 4)[0])
        self.assertEqual(counts["observed_truncated_products"], 1)
        self.assertEqual(counts["observed_truncated_groups"], 1)
        self.assertEqual(counts["observed_alignment_span_sum"], 34)
        # Only A's exponents change, so looking at B sparsity alone is wrong.
        uniform = torch.ones_like(a)
        changed, _, _ = group_column_work(a, uniform, "e4m3", "e4m3", config)
        same, _, _ = group_column_work(uniform, uniform, "e4m3", "e4m3", config)
        self.assertLess(changed.item(), same.item())

    def test_exact_streaming_mapping_matches_scalar_and_masks_both_tails(self):
        config = BitletConfig(pe_count=3, group_size=4, prefill_sample_waves=0, chunk_waves=1)
        x = ((torch.arange(3*9).reshape(3, 9) % 7)-3).float()
        w = ((torch.arange(7*9).reshape(7, 9) % 15)-8).float()
        result = measure_bitlet_mapping(x, w, "e4m3", 4, 9, 7, config)
        sparse, dense = scalar_schedule([x.tolist()], [w.tolist()], 3, 4)
        self.assertEqual(result["cycles"]["mapped_tiles"], sparse)
        self.assertEqual(result["cycles"]["dense_tiles"], dense)
        self.assertEqual(result["observed"]["observed_elements"], 3*7*9)
        self.assertLessEqual(result["cycles"]["ideal_pe_balance"], sparse)
        many = measure_bitlet_mapping(x, w, "e4m3", 4, 9, 7, replace(config, chunk_waves=19))
        self.assertEqual(many, result)
        with self.assertRaises(ValueError):
            measure_bitlet_mapping(x, w, "e4m3", 4, 9, 7,
                                   replace(config, local_buffer_bytes=1))

    def test_independent_attention_operands_match_scalar_schedule(self):
        config = BitletConfig(pe_count=3, group_size=4, prefill_sample_waves=0)
        x = ((torch.arange(2*2*9).reshape(1, 2, 2, 9) % 7)-3).float()
        w = ((torch.arange(2*9*7).reshape(1, 2, 9, 7) % 5)-2).float()
        result = measure_bitlet_mapping(x, w, "e4m3", "e4m3", 9, 7, config)
        expected = scalar_schedule(x[0].tolist(), w[0].transpose(-2, -1).tolist(), 3, 4)
        self.assertEqual(result["cycles"]["mapped_tiles"], expected[0])
        self.assertEqual(result["cycles"]["dense_tiles"], expected[1])
        self.assertEqual(result["observed"]["observed_elements"], x.shape[1]*2*9*7)

    def test_integer_mode_and_zero_groups_match_reference_with_setup(self):
        config = BitletConfig(pe_count=2, group_size=4, prefill_sample_waves=0)
        x = torch.tensor([[1., 0., -2., 3., 0.]])
        w = torch.tensor([[-8., -1., 0., 1., 7.], [0., 0., 0., 0., 0.], [7., 2., -4., -3., 0.]])
        result = measure_bitlet_mapping(x, w, 8, 4, 5, 3, config)
        expected = scalar_schedule([x.tolist()], [w.tolist()], 2, 4, integer=True)
        self.assertEqual(result["cycles"]["mapped_tiles"], expected[0])
        zero = torch.zeros_like(w)
        result = measure_bitlet_mapping(x, zero, 8, 4, 5, 3,
                                        replace(config, group_setup_cycles=2))
        self.assertEqual(result["cycles"]["mapped_tiles"], result["sampling"]["total_waves"]*2)
        # A zeros still consume W-interleaved cycles; this is not Asyn-CIM's
        # A-value/bit skipping, and causal PV groups must not receive it.
        for aspec, bspec in ((8, 4), ("e4m3", "e4m3")):
            zeros = torch.zeros(1, 4)
            ones = torch.ones(1, 4)
            work, _, _ = group_column_work(zeros, ones, aspec, bspec, config)
            self.assertEqual(work.item(), 4)

    def test_sampling_is_reproducible_chunk_invariant_and_reports_uncertainty(self):
        config = BitletConfig(pe_count=4, group_size=8, prefill_sample_waves=200, chunk_waves=7)
        torch.manual_seed(12)
        x = torch.randint(-7, 8, (120, 32)).float()
        w = torch.randint(-8, 8, (32, 32)).float()
        first = measure_bitlet_mapping(x, w, "e4m3", 4, 32, 32, config, seed=91)
        second = measure_bitlet_mapping(x, w, "e4m3", 4, 32, 32,
                                        replace(config, chunk_waves=1), seed=91)
        self.assertEqual(first, second)
        self.assertFalse(first["sampling"]["exact"])
        self.assertEqual(first["sampling"]["sample_waves"], 200)
        self.assertGreater(first["sampling"]["mapped_cycles_standard_error"], 0)
        exact = measure_bitlet_mapping(x, w, "e4m3", 4, 32, 32,
                                       replace(config, prefill_sample_waves=0))
        self.assertLess(abs(first["cycles"]["mapped_tiles"]-exact["cycles"]["mapped_tiles"]),
                        4*first["sampling"]["mapped_cycles_standard_error"])
        single = measure_bitlet_mapping(x, w, "e4m3", 4, 32, 32,
                                        replace(config, prefill_sample_waves=1))
        self.assertIsNone(single["sampling"]["mapped_cycles_approximate_95pct_interval"])
        # A sampled full K group must not overestimate the analytically known
        # dense work when the final K group is short.
        tail_config = BitletConfig(pe_count=3, group_size=4, prefill_sample_waves=2)
        tail = measure_bitlet_mapping(torch.ones(7, 9), torch.ones(7, 9),
                                      "e4m3", "e4m3", 9, 7, tail_config)
        self.assertEqual(tail["cycles"]["mapped_tiles"], 7*3*9)
        self.assertEqual(tail["mapped_compute_speedup"], 1.)
        self.assertEqual(tail["sampling"]["mapped_cycles_approximate_95pct_interval"], [189., 189.])

    def test_transport_uses_packed_w4_paper_buses_and_no_global_overlap(self):
        config = BitletConfig(group_size=4, pe_count=2, prefill_sample_waves=0)
        x, w = torch.ones(3, 8), torch.ones(4, 8)
        result = measure_bitlet_mapping(x, w, "e4m3", 4, 8, 4, config)
        traffic, latency = traffic_and_latency(result, x, w, "e4m3", 4, "q_proj", {}, config)
        self.assertEqual(traffic["packed_weight_read_bytes_minimum"], 16)
        self.assertEqual(traffic["activation_read_bytes_minimum"], 24)
        self.assertEqual(traffic["local_B_read_bytes_streaming"], 48)
        self.assertAlmostEqual(latency["weight_dma_seconds"], 16/12.8e9)
        self.assertAlmostEqual(latency["resident_full_overlap_seconds"],
            max(latency["mapped_compute_seconds"], latency["activation_dma_seconds"],
                latency["weight_dma_seconds"], latency["local_resident_seconds"]))
        self.assertGreater(latency["streaming_no_overlap_seconds"],
                           latency["resident_full_overlap_seconds"])

    def test_collector_gqa_trace_export_isolation_and_remaining_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(group_size=4, pe_count=2, prefill_sample_waves=0,
                          decode_sample_waves=0, trace_path=str(Path(tmp)/"trace.jsonl"))
            sm = QuantStatManager(tmp, bitlet_config=config)
            sm.set_phase("decode"); sm.set_step(0, cache_length_before=8, query_length=1)
            sm.set_attention_context(0, q_heads=4, kv_heads=2, head_dim=4,
                                     shared_kv_gqa=True, cache_length_before=8, query_length=1)
            a, b = torch.ones(1, 4, 1, 4), torch.ones(1, 4, 4, 9)
            spec = parse_quant_spec("e4m3")
            sm.collect_quant_activation("qk_matmul", 0, a, a, b, spec, spec, 8, 4, 4, 9)
            workload = dict(status="complete", decode_steps=1, completed_decode_steps=1)
            doc = sm.export_bitlet_stats(Path(tmp)/"summary.json", workload)
            self.assertIsNone(doc["latency"]["conditional_e2e_seconds"])
            with self.assertRaises(ValueError): sm.export_cim_stats(Path(tmp)/"wrong.json")
            with self.assertRaises(ValueError): sm.export_llmcompass_manifest("", {}, "dummy")
            with self.assertRaises(ValueError): sm.reset_sparsity()
            with self.assertRaises(ValueError): QuantStatManager(tmp, ebb_config={}, bitlet_config={})
            rest = dict(schema_version=1, includes_lm_head=True, prefill_seconds=0.,
                        decode_step_seconds=[0.1])
            doc = sm.export_bitlet_stats(Path(tmp)/"summary.json", workload, rest)
            sm.close()
            record = json.loads((Path(tmp)/"trace.jsonl").read_text())
            self.assertEqual(record["operand_shape"], [2, 2, 4])
            self.assertEqual(record["traffic"]["fp8_kv_read_bytes_minimum"], 64)
            self.assertEqual(record["traffic"]["fp8_kv_append_bytes"], 8)
            self.assertAlmostEqual(doc["latency"]["conditional_e2e_seconds"]["resident_full_overlap_seconds"],
                doc["latency"]["gemm_and_io_seconds"]["resident_full_overlap_seconds"]+0.1)
            from scripts.estimate_bitlet_latency import estimate
            combined = estimate(doc, rest)
            self.assertEqual(combined["conditional_e2e_seconds"],
                             doc["latency"]["conditional_e2e_seconds"])
            with self.assertRaises(ValueError):
                estimate(dict(doc, backend="ebb"), rest)
            with self.assertRaises(ValueError):
                estimate(dict(doc, workload=dict(workload, status="interrupted")), rest)
            for bad in (dict(rest, includes_lm_head=False), dict(rest, decode_step_seconds=[]),
                        dict(rest, prefill_seconds=float("nan"))):
                with self.assertRaises(ValueError): validate_other_latency(bad, 1)
            with self.assertRaises(ValueError): sm.bitlet_stats.validate_coverage(1, 1)

    def test_invalid_codes_are_rejected_before_mapping(self):
        config = BitletConfig(group_size=4)
        ones = torch.ones(1, 4)
        for a, b, aspec, bspec in (
            (ones*1.1, ones, "e4m3", 4),
            (ones, ones*8, "e4m3", 4),
            (ones*float("inf"), ones, "e4m3", 4),
        ):
            with self.assertRaises(ValueError): group_column_work(a, b, aspec, bspec, config)

    @unittest.skipUnless(importlib.util.find_spec("transformers") and importlib.util.find_spec("datasets"),
                         "Install requirements-model.txt")
    def test_local_qwen_cli_both_decode_modes_short_calibration_and_exact_coverage(self):
        import yaml
        from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from quant.qwen_wrapper import wrap_qwen_model
        from scripts.profile_bitlet import main
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = yaml.safe_load((root/"config/qwen2_14b_bitlet_f8i4_2048_256.yaml").read_text())
            config["quantization"]["scale_dir"] = str(tmp/"scales")
            model_dir = tmp/"model"
            model = Qwen2ForCausalLM(Qwen2Config(
                vocab_size=64, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)).eval()
            model.save_pretrained(model_dir)
            vocab = {str(index): index for index in range(64)}
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
                WordLevel(vocab, unk_token="0")), unk_token="0")
            tokenizer.save_pretrained(model_dir)
            tokens = torch.arange(16).long()
            torch.save(tokens, tmp/"tokens.pt")
            config["calibration"].update(seq_length=8192, min_text_tokens=8192, num_samples=1)
            config["quantization"]["calibration_policy"] = {"default": "recalibrate"}
            config_file = tmp/"config.yaml"
            config_file.write_text(yaml.safe_dump(config))
            calls = []
            def loader(*args, **kwargs):
                calls.append(kwargs)
                length = kwargs["seq_length"]
                return [{"input_ids": tokens[:length].unsqueeze(0),
                         "attention_mask": torch.ones(1, length, dtype=torch.long)}]
            with mock.patch("others.data.CalibrationDataLoader", side_effect=loader), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                main(["--config", str(config_file), "--model-path", str(model_dir),
                      "--output-dir", str(tmp/"calibration"), "--token-file", str(tmp/"tokens.pt"),
                      "--device", "cpu", "--prefill-length", "8", "--decode-steps", "2", "--exact"])
            self.assertEqual(calls[0]["seq_length"], 8)
            doc = json.loads((tmp/"calibration/bitlet_summary.json").read_text())
            self.assertEqual(doc["workload"]["calibration"]["completed_batches"], 1)
            self.assertEqual(doc["phases"]["prefill"]["calls"], 9)
            self.assertEqual(doc["phases"]["decode"]["calls"], 18)
            self.assertEqual(doc["phases"]["decode"]["sampled_calls"], 0)
            self.assertIsNone(doc["latency"]["conditional_e2e_seconds"])
            for phase in ("prefill", "decode"):
                self.assertEqual({key.split(":")[1].rsplit("_", 1)[0]
                                  for key in doc["layers"] if key.startswith(phase+":")}, set(OPERATORS))
            contexts = [step["context"]["cache_length_before"] for step in doc["steps"]
                        if step["phase"] == "decode"]
            self.assertEqual(contexts, [8, 9])
            rest = dict(schema_version=1, includes_lm_head=True, prefill_seconds=.01,
                        decode_step_seconds=[.001, .002])
            (tmp/"rest.json").write_text(json.dumps(rest))
            base_args = ["--config", str(config_file), "--model-path", str(model_dir),
                         "--token-file", str(tmp/"tokens.pt"), "--device", "cpu",
                         "--prefill-length", "8", "--decode-steps", "2", "--skip-calibration",
                         "--other-latency-json", str(tmp/"rest.json"), "--exact"]
            for mode in ("teacher", "greedy"):
                out = tmp/mode
                command = [sys.executable, "-m", "scripts.profile_bitlet", *base_args,
                           "--output-dir", str(out)]
                if mode == "greedy":
                    command.append("--greedy-decode")
                result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                doc = json.loads((out/"bitlet_summary.json").read_text())
                self.assertEqual(doc["workload"]["status"], "complete")
                self.assertTrue(doc["latency"]["collection_complete"])
                self.assertIsNotNone(doc["latency"]["conditional_e2e_seconds"])
                self.assertEqual(doc["phases"]["decode"]["calls"], 18)
                self.assertEqual(len((out/"bitlet_trace.jsonl").read_text().splitlines()), 27)
                estimate = doc["latency"]["conditional_e2e_seconds"]["resident_full_overlap_seconds"]
                baseline = doc["latency"]["gemm_and_io_seconds"]["resident_full_overlap_seconds"]
                self.assertAlmostEqual(estimate-baseline, .013)


if __name__ == "__main__":
    unittest.main()
