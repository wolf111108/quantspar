"""EBB bounds, code representation, traffic and real tiny-Qwen integration."""
from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest import mock

import torch
from quant import QuantStatManager
from quant.ebb import EBBConfig, EBBStats, effective_bit_groups, measure_ebb_mapping, operand_counts
from quant.quant_spec import parse_quant_spec


class EBBTests(unittest.TestCase):
    def test_transformers_version_window_matches_qwen2_api(self):
        from scripts.profile_ebb import (TRANSFORMERS_MAX_EXCLUSIVE, TRANSFORMERS_MIN,
                                         transformers_version_tuple)
        def accepted(raw):
            return TRANSFORMERS_MIN <= transformers_version_tuple(raw) < TRANSFORMERS_MAX_EXCLUSIVE
        # Validated pin and the exercised lower bound stay supported.
        self.assertTrue(accepted('4.43.1'))
        self.assertTrue(accepted('4.40.0'))
        self.assertTrue(accepted('4.44.2'))
        # 4.45 moved RoPE to position_ids and dropped Cache.get_usable_length.
        self.assertFalse(accepted('4.45.0'))
        self.assertFalse(accepted('4.45.0rc1'))
        self.assertFalse(accepted('5.0.0'))
        self.assertFalse(accepted('4.39.0'))

    def test_paper_figure13_replicated_to_eight_lanes(self):
        config = EBBConfig(input_bits=8)
        # Two replicas of Fig.13's EB=[4,1,4,7], now on eight detector lanes.
        x = torch.tensor([[8., 1., 8., 64.] * 2])
        costs, counts = effective_bit_groups(x, 8, config)
        self.assertEqual(costs['dense'].item(), 8)
        self.assertEqual(costs['leading_zero'].item(), 7)
        self.assertEqual(costs['ideal_balanced'].item(), 4)
        self.assertEqual(counts['effective_bits_histogram'][4], 4)

    def test_hidden_one_is_counted_and_exponent_span_is_preserved(self):
        config = EBBConfig(input_bits=8)
        costs, counts = effective_bit_groups(torch.ones(1, 8), 'e4m3', config)
        self.assertEqual(costs['leading_zero'].item(), 4)
        self.assertEqual(counts['effective_bits_histogram'][4], 8)
        x = torch.tensor([[1., 2**-9, 0., 0., 0., 0., 0., 0.]])
        _, counts = effective_bit_groups(x, 'e4m3', config)
        self.assertEqual(counts['overflow_int8_groups'], 1)
        self.assertEqual(counts['overflow_int16_groups'], 0)
        self.assertEqual(counts['group_max_bits_histogram'][10], 1)

    def test_overflow_is_reported_without_clipping(self):
        x = torch.tensor([[448., 2**-9, 0., 0., 0., 0., 0., 0.]])
        costs, counts = effective_bit_groups(x, 'e4m3', EBBConfig())
        self.assertEqual(counts['overflow_int16_groups'], 1)
        self.assertEqual(costs['dense'].item(), 19)
        self.assertEqual(costs['leading_zero'].item(), 18)
        with self.assertRaises(ValueError):
            effective_bit_groups(torch.full((1, 8), 1.1), 'e4m3', EBBConfig())

    def test_signed_width_and_tail_padding(self):
        costs, counts = effective_bit_groups(torch.tensor([[-128., -1., 0., 1.]]), 8,
                                             EBBConfig(input_bits=8))
        self.assertEqual(counts['elements'], 4)
        self.assertEqual(counts['negative_elements'], 2)
        self.assertEqual(counts['overflow_int8_groups'], 0)
        self.assertEqual(counts['zero_elements'], 1)
        self.assertEqual(costs['leading_zero'].item(), 8)
        self.assertEqual(costs['ideal_balanced'].item(), 3)

    def test_chunking_invariance_and_bounds(self):
        config = EBBConfig(macros=8, arrays_per_macro=2, input_bits=8)
        x = (torch.arange(2*5*33).reshape(2, 5, 33) % 65).float()
        one = measure_ebb_mapping(x, 8, 33, 7, replace(config, chunk_rows=1))
        many = measure_ebb_mapping(x, 8, 33, 7, replace(config, chunk_rows=512))
        self.assertEqual(one, many)
        cycles = one['cycles']
        self.assertLessEqual(cycles['ideal_balanced'], cycles['leading_zero'])
        self.assertLessEqual(cycles['leading_zero'], cycles['dense'])

    def test_mapping_matches_independent_scalar_schedule(self):
        config = EBBConfig(macros=8, arrays_per_macro=2, input_bits=8)
        x = (torch.arange(2*5*33).reshape(2, 5, 33) % 65).float()
        result = measure_ebb_mapping(x, 8, 33, 7, config)
        layout = result['layout']
        for mode in ['dense', 'leading_zero', 'ideal_balanced']:
            expected = 0
            for operand in x.tolist():
                for start in range(0, len(operand), layout['token_parallel']):
                    wave = []
                    for row in operand[start:start + layout['token_parallel']]:
                        grouped = []
                        for group_start in range(0, len(row), 8):
                            eb = [int(v).bit_length() for v in row[group_start:group_start+8]]
                            grouped.append(8 if mode == 'dense' else
                                           max(eb) if mode == 'leading_zero' else
                                           (sum(eb)+7)//8)
                        size = layout['groups_per_lane']
                        for lane in range(layout['k_parallel']):
                            wave.append(sum(grouped[lane*size:(lane+1)*size]))
                    expected += max(wave) * layout['n_rounds']
            self.assertEqual(result['cycles'][mode], expected)

    def test_streamed_trace_gqa_traffic_and_schema_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            default = QuantStatManager(tmp, ebb_config={})
            self.assertIsNotNone(default.ebb_stats)
            default.close()
            sm = QuantStatManager(tmp, ebb_config=dict(trace_path=str(Path(tmp)/'trace.jsonl')))
            sm.set_phase('decode');sm.set_step(0, cache_length_before=8)
            sm.set_attention_context(0, q_heads=4, kv_heads=2, head_dim=4,
                                     shared_kv_gqa=True, cache_length_before=8, query_length=1)
            spec = parse_quant_spec('e4m3')
            a = torch.ones(1, 4, 1, 4)
            b = torch.ones(1, 4, 4, 9)
            sm.collect_quant_activation('qk_matmul', 0, a, a, b, spec, spec, 8, 4, 4, 9)
            doc = sm.export_ebb_stats(Path(tmp)/'summary.json')
            sm.close()
            record = json.loads((Path(tmp)/'trace.jsonl').read_text())
            self.assertEqual(record['operand_shape'], [2, 2, 4])
            self.assertEqual(record['traffic']['fp8_kv_read_bytes_minimum'], 64)
            self.assertEqual(record['traffic']['fp8_kv_append_bytes'], 8)
            self.assertEqual(doc['layers']['decode:qk_matmul_0']['weight']['elements'], 2*4*9)
            with self.assertRaises(ValueError):sm.export_cim_stats(Path(tmp)/'wrong.json')
            with self.assertRaises(ValueError):sm.export_llmcompass_manifest('', {}, 'dummy')

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'Install requirements-model.txt')
    def test_real_qwen_prefill_decode_and_fp8_cache(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from transformers.cache_utils import DynamicCache
        from quant.qwen_wrapper import wrap_qwen_model, switch_quantization_mode_all
        from scripts.profile_ebb import bind_manager
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as tmp:
            config = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=2, num_attention_heads=4,
                                num_key_value_heads=2, max_position_embeddings=64)
            model = Qwen2ForCausalLM(config).eval()
            quant = dict(scale_dir=tmp, model_family='qwen2', quantize_matmul=True,
                         calibration_policy={'default':'recalibrate'}, shared_kv_gqa=True,
                         kv_cache={'fp8_static':True})
            for name in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
                quant[name] = dict(a_bit='e4m3', w_bit=4, o_bit='e4m3', outlier_ratio=0)
            for name in ['qk_matmul','pv_matmul']:
                quant[name] = dict(A_bit='e4m3', B_bit='e4m3', O_bit='e4m3', outlier_ratio=0)
            calibration = QuantStatManager(tmp)
            wrap_qwen_model(model, quant, stat_manager=calibration)
            tokens = torch.randint(0, 64, (1, 12))
            with torch.no_grad():model.model(tokens, use_cache=False)
            calibration.save_all_scales()
            switch_quantization_mode_all(model, 'quant_forward')
            sm = QuantStatManager(tmp, ebb_config={"enabled":True})
            bind_manager(model, sm)
            with torch.no_grad():
                sm.set_phase('prefill');sm.set_step(0, query_length=8)
                out = model.model(tokens[:, :8], past_key_values=DynamicCache(), use_cache=True)
                self.assertEqual(out.past_key_values.get_seq_length(), 8)
                sm.set_phase('decode')
                for step in range(3):
                    sm.set_step(step, cache_length_before=8+step, query_length=1)
                    out = model.model(tokens[:, 8+step:9+step], past_key_values=out.past_key_values,
                                      use_cache=True)
                    self.assertEqual(out.past_key_values.get_seq_length(), 9+step)
            doc = sm.export_ebb_stats(Path(tmp)/'summary.json')
            self.assertEqual(doc['phases']['prefill']['counts']['calls'], 18)
            self.assertEqual(doc['phases']['decode']['counts']['calls'], 54)
            self.assertEqual([s['context']['cache_length_before'] for s in doc['steps']
                              if s['phase']=='decode'], [8, 9, 10])
            self.assertEqual(doc['phases']['prefill']['counts']['traffic']['fp8_kv_append_bytes'], 256)
            for index, layer in enumerate(model.model.layers):
                scales = layer.self_attn.qk_matmul.B_interval
                cached = out.past_key_values.key_cache[index].float() / scales
                torch.testing.assert_close(cached, cached.to(torch.float8_e4m3fn).float())
            sm.close()

    def test_incremental_cache_statistics_match_complete_recount(self):
        config = EBBConfig(chunk_rows=3)
        stats = EBBStats(config)
        spec = parse_quant_spec('e4m3')
        qk = (torch.arange(2*8*12).reshape(1, 2, 8, 12) % 7).float()
        pv = qk.transpose(-2, -1).contiguous()
        for length in (4, 5, 8, 9, 12):
            meta = dict(kv_cache_encoding='fp8_static_dequantized_emulation',
                        cache_length_before=0 if length == 4 else length-1)
            for name, weight in [('qk_matmul', qk[..., :length]), ('pv_matmul', pv[..., :length, :])]:
                got = stats._weight_counts(name, 0, 'decode', weight, spec, meta)
                expected = operand_counts(weight.transpose(-2, -1), spec, config)
                self.assertEqual(got, expected)

    def test_dense_geometry_is_not_the_papers_sparse_peak_tops(self):
        config = EBBConfig(input_bits=8, frequency_hz=275e6)
        eight = measure_ebb_mapping(torch.full((128, 32), 127.), 8, 32, 32, config)
        tops8 = 2*128*32*32 / eight['compute_seconds']['dense'] / 1e12
        sixteen = measure_ebb_mapping(torch.full((128, 16), 32767.), 16, 16, 32,
                                      replace(config, input_bits=16), weight_bits=16)
        tops16 = 2*128*16*32 / sixteen['compute_seconds']['dense'] / 1e12
        self.assertAlmostEqual(tops8, 2.2528)
        self.assertAlmostEqual(tops16, 0.5632)

    @unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('datasets'),
                         'Install requirements-model.txt')
    def test_local_profile_cli_teacher_forced_greedy_and_short_calibration(self):
        import yaml
        from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from quant.qwen_wrapper import wrap_qwen_model
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = yaml.safe_load((root/'config/qwen2_14b_ebb_f8i4.yaml').read_text())
            config['quantization']['scale_dir'] = str(tmp/'scales')
            model_dir = tmp/'model'
            model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)).eval()
            model.save_pretrained(model_dir)
            vocabulary = {'[UNK]':0, '[PAD]':1, '[EOS]':2}
            vocabulary.update({f'w{i}':i for i in range(3,64)})
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocabulary,
                    unk_token='[UNK]')), unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
            tokenizer.save_pretrained(model_dir)
            sm = QuantStatManager(str(tmp/'scales'))
            wrap_qwen_model(model, config['quantization'], stat_manager=sm)
            tokens = torch.arange(12).reshape(1,12)
            with torch.no_grad():model.model(tokens, use_cache=False)
            sm.save_all_scales()
            cfg_file = tmp/'config.yaml';cfg_file.write_text(yaml.safe_dump(config))
            torch.save(tokens, tmp/'tokens.pt')
            for mode in ('teacher', 'greedy'):
                command = [sys.executable, '-m', 'scripts.profile_ebb', '--config', str(cfg_file),
                           '--model-path', str(model_dir), '--output-dir', str(tmp/mode),
                           '--token-file', str(tmp/'tokens.pt'), '--skip-calibration', '--device', 'cpu',
                           '--scale-dir', str(tmp/'scales'),
                           '--prefill-length', '8', '--decode-steps', '3', '--checkpoint-every', '1']
                if mode == 'greedy':command.append('--greedy-decode')
                result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                doc = json.loads((tmp/mode/'ebb_summary.json').read_text())
                self.assertEqual(doc['workload']['status'], 'complete')
                self.assertEqual(doc['workload']['completed_decode_steps'], 3)
                self.assertFalse(doc['workload']['calibration']['performed'])
                self.assertEqual(doc['phases']['prefill']['counts']['calls'], 9)
                self.assertEqual(doc['phases']['decode']['counts']['calls'], 27)
                self.assertEqual(len((tmp/mode/'ebb_trace.jsonl').read_text().splitlines()), 36)

            # Run actual calibration/forward passes against a saved checkpoint.
            # The source YAML still says 8192; neither loader nor model may use it
            # when profiling eight tokens, unless calibration is explicitly overridden.
            from transformers import AutoModelForCausalLM
            from scripts.profile_ebb import main
            original_load = AutoModelForCausalLM.from_pretrained
            for calibration_length in (None, 16):
                seen_lengths = []

                def load_local_model(*args, **kwargs):
                    loaded = original_load(*args, **kwargs)
                    loaded.model.register_forward_pre_hook(
                        lambda module, positional, kw: seen_lengths.append(
                            (kw['input_ids'] if 'input_ids' in kw else positional[0]).shape[-1]),
                        with_kwargs=True)
                    return loaded

                def local_calibration_loader(*args, **kwargs):
                    ids = torch.arange(kwargs['seq_length']).reshape(1, -1) % 64
                    return [dict(input_ids=ids, attention_mask=torch.ones_like(ids))]

                name = 'auto_calibration' if calibration_length is None else 'explicit_calibration'
                arguments = ['--config', str(cfg_file), '--model-path', str(model_dir),
                             '--output-dir', str(tmp/name), '--scale-dir', str(tmp/(name+'_scales')),
                             '--token-file', str(tmp/'tokens.pt'), '--device', 'cpu',
                             '--prefill-length', '8', '--decode-steps', '3', '--calibration-samples', '1']
                if calibration_length is not None:
                    arguments += ['--calibration-length', str(calibration_length)]
                with mock.patch('transformers.AutoModelForCausalLM.from_pretrained',
                                side_effect=load_local_model), \
                     mock.patch('others.data.CalibrationDataLoader',
                                side_effect=local_calibration_loader) as loader, \
                     redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    main(arguments)
                length = 8 if calibration_length is None else calibration_length
                self.assertEqual(loader.call_args.kwargs['seq_length'], length)
                self.assertEqual(loader.call_args.kwargs['min_text_tokens'], length)
                self.assertEqual(seen_lengths, [length, 8, 1, 1, 1])
                doc = json.loads((tmp/name/'ebb_summary.json').read_text())
                self.assertEqual(doc['workload']['status'], 'complete')
                self.assertTrue(doc['workload']['calibration']['performed'])
                self.assertEqual(doc['workload']['calibration']['seq_length'], length)
                self.assertEqual(doc['workload']['calibration']['completed_batches'], 1)
                self.assertEqual(doc['workload']['calibration']['operators_recalibrated'], 9)
                self.assertEqual(doc['workload']['calibration']['operators_reused'], 0)
                self.assertEqual(doc['phases']['decode']['counts']['calls'], 27)


if __name__ == '__main__':
    unittest.main()
