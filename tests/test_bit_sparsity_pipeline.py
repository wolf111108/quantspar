"""Real encoded-bit and saved OPT/Qwen pipeline regressions; no checkpoints/downloads."""
from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import yaml

from quant import QuantizedLinear, QuantStatManager
from quant.cim_stats import sparse_counts
from quant.quant_spec import parse_quant_spec, quant_awo, resolve_model_dtype, safe_scale_from_tensor

ROOT = Path(__file__).resolve().parents[1]


class BitSparsityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def manager(self, **kwargs):
        return QuantStatManager(self.tmp.name, collect_mapping=False, **kwargs)

    def test_counter_ratio_and_reset_use_total_bits(self):
        sm = self.manager()
        sm.set_phase('prefill')
        sm.collect_activation_sparsity(6, 1, 18, 12, 12, 12)
        self.assertEqual(sm.shift_bit_total_count, 18)
        self.assertEqual(sm.shift_bit_0_count / sm.shift_bit_total_count, 2/3)
        self.assertEqual(sm.activation_0bit_count, 12)
        self.assertEqual(sm.total_bit_count, 18)
        self.assertEqual(sm.phase_sparsity['prefill']['total_0bit_count'], 12)
        sm.reset_sparsity()
        self.assertEqual(sm.shift_bit_total_count, 0)
        self.assertEqual(sm.shift_bit_0_count, 0)
        self.assertEqual(sm.total_bit_count, 0)
        self.assertEqual(sm.sparsity_records, {})
        self.assertIsNone(sm.get_sparsity_summary()['total']['bit_zero_ratio'])

    def test_all_fp_roles_use_the_same_explicit_mantissa_bits(self):
        # Known native codes, including subnormals and negative values. The
        # expected counts come from scalar integer masks, independent of Torch.
        cases = [
            ('e4m3', 3, [0., 1., 1.875, -1.5, 2**-9], [0x00, 0x38, 0x3f, 0xbc, 0x01]),
            ('e5m2', 2, [0., 1., 1.75, -1.5, 2**-16], [0x00, 0x3c, 0x3f, 0xbe, 0x01]),
            ('fp16', 10, [0., 1., 1+2**-10, -1.5, 2**-24], [0, 0x3c00, 0x3c01, 0xbe00, 1]),
            ('bf16', 7, [0., 1., 1+2**-7, -1.5, 2**-133], [0, 0x3f80, 0x3f81, 0xbfc0, 1]),
        ]
        for fmt, width, values, raw in cases:
            with self.subTest(format=fmt):
                sm = self.manager()
                spec = parse_quant_spec(fmt)
                a = torch.tensor(values).reshape(1, -1)
                w = a.clone()
                expected_zero = len(raw)*width - sum((r & ((1 << width)-1)).bit_count() for r in raw)
                for name in ('q_proj', 'qk_matmul', 'pv_matmul'):
                    sm.collect_quant_activation(name, 0, a, a, w, spec, spec, None, None, 5, 5)
                rows = sm.get_sparsity_summary()['records']
                self.assertEqual({r['operand'] for r in rows}, {'activation', 'weight', 'Q', 'K', 'attention_probs', 'V'})
                for row in rows:
                    self.assertEqual(row['counted_width'], width)
                    self.assertEqual(row['bits'], 5*width)
                    self.assertEqual(row['zero_bits'], expected_zero)
                self.assertEqual(sm.total_bit_count, 6*5*width)
                self.assertEqual(sm.total_0bit_count, 6*expected_zero)

    def test_static_weight_dedup_phase_totals_and_export(self):
        sm = self.manager(cache_static_weight_counts=True)
        a = torch.tensor([[0., 1., 1.875, -1.5]])  # 8 / 12 zero mantissa bits
        w = torch.tensor([[-8., -1., 0., 7.]])  # 8 / 16 zero two's-complement bits
        sa, sw = parse_quant_spec('e4m3'), parse_quant_spec(4)
        for phase in ('prefill', 'prefill', 'decode'):
            sm.set_phase(phase)
            sm.collect_quant_activation('q_proj', 0, a, a, w, sw, sa, None, None, 4, 1)
        doc = sm.export_sparsity_stats(Path(self.tmp.name)/'bits.json', Path(self.tmp.name)/'bits.csv')
        self.assertEqual(doc['total']['bits'], 68)
        self.assertEqual(doc['total']['zero_bits'], 40)
        self.assertEqual(doc['total']['bit_zero_ratio'], 40/68)
        self.assertEqual(sm.total_bit_count, 68)
        self.assertEqual(sm.phase_sparsity['prefill']['total_bit_count'], 40)
        self.assertEqual(sm.phase_sparsity['decode']['total_bit_count'], 28)
        weights = [r for r in doc['records'] if r['operand'] == 'weight']
        self.assertEqual([r['observations'] for r in weights], [1, 1])
        self.assertEqual(json.loads((Path(self.tmp.name)/'bits.json').read_text())['total']['zero_bits'], 40)
        with (Path(self.tmp.name)/'bits.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)

    def test_quant_forward_calls_statistics_without_mapping_or_units(self):
        op = QuantizedLinear(4, 2, a_bit='e4m3', w_bit=4, o_bit='none',
                             mode='scale_inspection', scale_root_str=self.tmp.name)
        op.set_layer_info('q_proj', 0)
        x = torch.tensor([[0., 1., -1.5, 2.]])
        sm = self.manager()
        op(x, stat_collector=sm)
        self.assertEqual(sm.sparsity_records, {})  # calibration collects scales only
        op.save_scales()
        op.mode = 'quant_forward'
        with mock.patch('quant.stat_manager.measure_mapping', side_effect=AssertionError('mapping called')), \
             mock.patch('quant.stat_manager.unit_sparse_counts', side_effect=AssertionError('units called')):
            op(x, stat_collector=sm)
        self.assertGreater(sm.total_bit_count, 0)
        self.assertEqual(len(sm.sparsity_records), 2)
        self.assertFalse(sm.enable_unit_sparsity)
        self.assertEqual(sm.cim_records, [])
        self.assertFalse(any(sm.unit_sparsity.values()))

    def test_fp16_is_a_real_unscaled_quantizer(self):
        spec = parse_quant_spec('fp16')
        self.assertTrue(spec.enabled)
        x = torch.tensor([1.0003, 1.0011, -2.003, 70000.])
        q = quant_awo(x, 1, spec, out_dtype=torch.float32, chunk_size=2)
        torch.testing.assert_close(q, x.clamp(-65504, 65504).half().float(), rtol=0, atol=0)
        self.assertEqual(float(q[0]), 1.)
        self.assertEqual(safe_scale_from_tensor(x, spec), 1)
        with self.assertRaisesRegex(ValueError, 'scale=1'):
            quant_awo(x, .1, spec)

    def test_fp16_linear_quantizes_output_without_bypassing_operands(self):
        op = QuantizedLinear(4, 2, a_bit='fp16', w_bit='fp16', o_bit='fp16',
                             mode='scale_inspection', scale_root_str=self.tmp.name)
        x = torch.tensor([[1.0003, -.2001, 2.003, -.41003]])
        op(x); op.save_scales(); op.mode = 'quant_forward'
        sm = self.manager()
        got = op(x, stat_collector=sm)
        reference = torch.nn.functional.linear(x.half().float(), op.weight.half().float(), op.bias).half().float()
        torch.testing.assert_close(got, reference, rtol=0, atol=0)
        self.assertEqual({r['counted_width'] for r in sm.get_sparsity_summary()['records']}, {10})

    def test_full_fp16_model_dtype_and_default_config(self):
        config = {'quantization': {'model_family':'opt', 'quantize_matmul':True}}
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'):
            config['quantization'][name] = dict(a_bit='fp16', w_bit='fp16', o_bit='fp16')
        for name in ('qk_matmul', 'pv_matmul'):
            config['quantization'][name] = dict(A_bit='fp16', B_bit='fp16', O_bit='fp16')
        with mock.patch('torch.cuda.is_bf16_supported', return_value=True):
            self.assertIs(resolve_model_dtype(config), torch.float16)
        config['model'] = {'dtype':'bf16'}
        with self.assertRaisesRegex(ValueError, 'Full FP16'):
            resolve_model_dtype(config)
        q = yaml.safe_load((ROOT/'config/qwen2_14b_linear_matmul_f8i4.yaml').read_text())
        self.assertFalse(q['unit_sparsity']['enabled'])
        self.assertFalse(q['quantization']['mixed_precision'])
        for name, layer in q['quantization'].items():
            if isinstance(layer, dict) and 'outlier_ratio' in layer:
                expected = 0 if name in {'qk_matmul', 'pv_matmul'} else 0.0001
                self.assertEqual(layer['outlier_ratio'], expected, name)
        self.assertIn('outlier_v3', q['quantization']['scale_dir'])

    def test_native_fp16_templates_have_isolated_scales_and_valid_context(self):
        for path, family in (('opt_1.3b_linear_matmul_fp16.yaml', 'opt'),
                             ('qwen2_14b_linear_matmul_fp16.yaml', 'qwen2')):
            config = yaml.safe_load((ROOT/'config'/path).read_text())
            self.assertIs(resolve_model_dtype(config, 'cpu'), torch.float16)
            self.assertEqual(config['quantization']['model_family'], family)
            self.assertIn('fp16_native_v2', config['quantization']['scale_dir'])
            self.assertEqual(config['quantization']['calibration_policy']['default'], 'recalibrate')
            self.assertFalse(config['unit_sparsity']['enabled'])
            for layer in config['quantization'].values():
                if isinstance(layer, dict) and 'outlier_ratio' in layer:
                    self.assertEqual(layer['outlier_ratio'], 0)
                    for key, value in layer.items():
                        if key.lower() in {'a_bit', 'w_bit', 'b_bit', 'o_bit'}:
                            self.assertEqual(parse_quant_spec(value).fmt, 'e5m10')
            if family == 'opt':
                profile = config['prefill_decode_profile']
                self.assertLessEqual(profile['prefill_length'] + profile['decode_steps'], 2048)


@unittest.skipUnless(importlib.util.find_spec('transformers'), 'Install requirements-model.txt')
class SavedModelPipelineTests(unittest.TestCase):
    def test_saved_opt_and_qwen_calibration_ppl_and_decode(self):
        from transformers import OPTConfig, OPTForCausalLM, Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        spec = importlib.util.spec_from_file_location('quant_pipeline_under_test', ROOT/'0103_quant_pipeline_main.py')
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        for family in ('opt', 'qwen2'):
            for variant in ('int8', 'f8i4', 'fp16', 'int8_outlier', 'f8i4_outlier', 'fp16_outlier'):
                fmt = variant.removesuffix('_outlier')
                masked = variant.endswith('_outlier')
                ratio = .0001 if masked else 0.
                with self.subTest(family=family, format=variant), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    model_dir = root/'model'
                    torch.manual_seed(9)
                    if family == 'opt':
                        model = OPTForCausalLM(OPTConfig(vocab_size=32, hidden_size=16, ffn_dim=32,
                            num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64,
                            dropout=0., attention_dropout=0.))
                        names = ('q_proj','k_proj','v_proj','out_proj','fc1','fc2')
                    else:
                        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=64, attention_dropout=0.))
                        names = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
                    model.save_pretrained(model_dir)
                    tok = Tokenizer(WordLevel({'[UNK]':0,'[PAD]':1,'[EOS]':2, **{f'w{i}':i for i in range(3,32)}}, unk_token='[UNK]'))
                    tok.pre_tokenizer = Whitespace()
                    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]').save_pretrained(model_dir)
                    a_bit, w_bit = (8, 8) if fmt == 'int8' else (('e4m3', 4) if fmt == 'f8i4' else ('fp16', 'fp16'))
                    quant = dict(model_family=family, quantize_linear=True, quantize_matmul=True,
                        scale_dir=str(root/'scales'), mixed_precision=False, calibration_policy={'default':'recalibrate'})
                    for name in names:
                        quant[name] = dict(a_bit=a_bit, w_bit=w_bit, o_bit=a_bit, outlier_ratio=ratio)
                    for name in ('qk_matmul','pv_matmul'):
                        quant[name] = dict(A_bit=a_bit, B_bit=a_bit, O_bit=a_bit, outlier_ratio=ratio)
                    config = dict(model={'family':family, 'attn_implementation':'eager'}, quantization=quant,
                        calibration=dict(dataset='local-test', num_samples=1, seq_length=8),
                        evaluation=dict(datasets=[dict(name='local-test', streaming=True, seq_length=8,
                            max_eval_tokens=24, min_doc_tokens=1)]),
                        prefill_decode_profile=dict(enabled=True, streaming=True, prefill_length=8,
                            decode_steps=2, num_samples=1, max_eval_tokens=10, min_doc_tokens=1),
                        unit_sparsity=dict(enabled=True))  # old YAML must not re-enable units
                    config_path = root/'config.yaml'
                    config_path.write_text(yaml.safe_dump(config))
                    ids = torch.arange(3,11).reshape(1,-1)
                    loader = [dict(input_ids=ids, attention_mask=torch.ones_like(ids))]
                    document = [{'text':' '.join(f'w{3+i%29}' for i in range(48))}]
                    argv = ['pipeline', '--config', str(config_path), '--model-path', str(model_dir),
                            '--device', 'cpu', '--results-dir', str(root/'reports'),
                            '--stats-output-dir', str(root/'stats')]
                    with mock.patch('sys.argv', argv), mock.patch.object(pipeline, 'CalibrationDataLoader', return_value=loader), \
                         mock.patch('perplexity._load_hf_dataset', return_value=document), redirect_stdout(io.StringIO()):
                        pipeline.main()
                    full = json.loads((root/'stats/config_full_forward_bit_sparsity.json').read_text())
                    pd = json.loads((root/'stats/config_prefill_decode_bit_sparsity.json').read_text())
                    self.assertGreater(full['total']['bits'], 0)
                    self.assertEqual(full['outlier_sparsity']['present'], masked)
                    self.assertEqual(pd['outlier_sparsity']['present'], masked)
                    self.assertEqual({r['phase'] for r in pd['records']}, {'prefill','decode'})
                    self.assertEqual({r['operand'] for r in pd['records']}, {'activation','weight','Q','K','attention_probs','V'})
                    for row in pd['records']:
                        expected = (4 if row['operand']=='weight' else 3) if fmt == 'f8i4' else (8 if fmt=='int8' else 10)
                        self.assertEqual(row['counted_width'], expected)
                        self.assertEqual(row['outlier_masked'], masked)
                        self.assertTrue(0 <= row['bit_zero_ratio'] <= 1)
                        if row['operand'] == 'weight':
                            expected_calls = 2 if masked and row['phase'] == 'decode' else 1
                            self.assertEqual(row['observations'], expected_calls)
                    self.assertFalse((root/'stats/unit_sparsity').exists())
                    text = next((root/'reports').glob('*.txt')).read_text()
                    self.assertIn('full_forward: sum zero bits', text)
                    self.assertIn('prefill_decode: sum zero bits', text)
                    self.assertIn('Perplexity:', text)
                    if masked:
                        self.assertIn('outlier mask channels excluded', text)
                        self.assertIn('high-precision sidepath operands excluded', text)


if __name__ == '__main__':
    unittest.main()
