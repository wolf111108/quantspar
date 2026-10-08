"""Masked normal-operand statistics and real-domain outlier regressions."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from quant import QuantizedLinear, QuantizedMatMul, QuantStatManager
from quant.quant_spec import parse_quant_spec as type_spec


def reference_codes(values, scale, spec):
    """Native grid reference, independent of quant_awo and the sidepath."""
    normalized = values.float() / torch.as_tensor(scale)
    if spec.kind == 'int':
        limit = 2 ** (spec.bits - 1)
        return normalized.round().clamp(-limit, limit - 1)
    dtype = {'e4m3': torch.float8_e4m3fn, 'e5m10': torch.float16}[spec.fmt]
    maximum = torch.finfo(dtype).max
    return normalized.clamp(-maximum, maximum).to(dtype).float()


def reference_output(values, scale, spec):
    if not spec.enabled:
        return values
    scores = values.reshape(-1, values.shape[-1]).abs().amax(0)
    channel = int(scores.argmax())  # fixtures use ratio small enough for one channel
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[..., channel] = True
    codes = reference_codes(values.masked_fill(mask, 0), scale, spec)
    return torch.where(mask, values, codes * scale)


class OutlierSparsityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def manager(self, **kwargs):
        return QuantStatManager(self.tmp.name, collect_mapping=False,
                                cache_static_weight_counts=True, **kwargs)

    def calibrate(self, op, *inputs):
        op.mode = 'scale_inspection'
        op(*inputs)
        op.save_scales()
        op.mode = 'quant_forward'

    def test_mask_zeros_counts_refresh_weights_each_call_and_export_scope(self):
        op = QuantizedLinear(4, 2, bias=False, a_bit='e4m3', w_bit=4,
                             o_bit='none', outlier_ratio=.0001,
                             scale_root_str=self.tmp.name)
        op.set_layer_info('q_proj', 0)
        with torch.no_grad():
            op.weight.copy_(torch.tensor([[1., 2., 3., 100.], [1., 0., 4., 0.]]))
        first = torch.tensor([[9., 1., 1., 1.]])
        second = torch.tensor([[1., 1., 9., 1.]])
        self.calibrate(op, first)
        sm = self.manager()
        with mock.patch('quant.stat_manager.measure_mapping', side_effect=AssertionError('mapping called')), \
             mock.patch('quant.stat_manager.unit_sparse_counts', side_effect=AssertionError('units called')):
            for x in (first, second):
                torch.testing.assert_close(op(x, stat_collector=sm), op(x), rtol=0, atol=0)
        # W codes for the two input-dependent masks are:
        # [0,4,6,0; 0,0,7,0] and [2,4,0,0; 2,0,0,0].
        # Integer bit counts below are scalar two's-complement references.
        expected_weight_zeros = 64 - sum(v.bit_count() for v in (0,4,6,0,0,0,7,0,2,4,0,0,2,0,0,0))
        doc = sm.export_sparsity_stats(Path(self.tmp.name)/'outlier.json', Path(self.tmp.name)/'outlier.csv')
        weight = next(row for row in doc['records'] if row['operand'] == 'weight')
        activation = next(row for row in doc['records'] if row['operand'] == 'activation')
        self.assertEqual(weight['bits'], 64)
        self.assertEqual(weight['zero_bits'], expected_weight_zeros)
        self.assertEqual(weight['observations'], 2)
        self.assertEqual(weight['counting'], 'each_quantized_forward')
        self.assertEqual(activation['zero_elements'], 2)
        self.assertEqual(activation['bits'], 24)
        self.assertEqual(activation['zero_bits'], 12)  # zero + three native 0x7e codes per call
        self.assertEqual(doc['total']['bits'], 88)
        self.assertEqual(doc['total']['bit_zero_ratio'], (12 + expected_weight_zeros)/88)
        self.assertEqual(sm._static_weight_counts, {})
        self.assertTrue(all(row['outlier_masked'] for row in doc['records']))
        self.assertEqual(doc['schema_version'], 2)
        self.assertTrue(doc['outlier_sparsity']['present'])
        self.assertEqual(doc['outlier_sparsity']['mask_generated_zeros'], 'included')
        self.assertFalse(doc['outlier_sparsity']['high_precision_sidepath_counted'])
        with (Path(self.tmp.name)/'outlier.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['outlier_masked'], 'True')
        self.assertEqual(json.loads((Path(self.tmp.name)/'outlier.json').read_text())['total'], doc['total'])

    def test_linear_shapes_channel_scales_bias_and_outputs_match_reference(self):
        weight = torch.tensor([[1., .5, .2, 100.], [.3, .2, .1, .6], [-.3, .9, 1.1, .2]])
        bias = torch.tensor([.125, -.25, .03125])
        for shape in ((4,), (2,4), (1,2,4), (1,1,2,4)):
            for output in ('none', 'e4m3', 8, 'fp16'):
                with self.subTest(shape=shape, output=output):
                    op = QuantizedLinear(4, 3, a_bit='e4m3', w_bit=4, o_bit=output,
                        outlier_ratio=.0001, weight_scale_granularity='output_channel',
                        scale_root_str=self.tmp.name)
                    with torch.no_grad():
                        op.weight.copy_(weight); op.bias.copy_(bias)
                    x = torch.tensor([9., 1., .3, .2]).expand(shape).clone()
                    self.calibrate(op, x)
                    x_mask = torch.zeros_like(x, dtype=torch.bool); x_mask[..., 0] = True
                    w_mask = torch.zeros_like(weight, dtype=torch.bool)
                    w_mask[:, 0] = True; w_mask[0, 3] = True
                    x_normal, w_normal = x.masked_fill(x_mask, 0), weight.masked_fill(w_mask, 0)
                    xs = reference_codes(x_normal, op.a_interval, op.a_spec) * op.a_interval
                    ws = reference_codes(w_normal, op.w_interval[:, None], op.w_spec) * op.w_interval[:, None]
                    xp, wp = x.masked_fill(~x_mask, 0), weight.masked_fill(~w_mask, 0)
                    expected = (torch.nn.functional.linear(xs, ws)
                                + torch.nn.functional.linear(xp, wp)
                                + torch.nn.functional.linear(xp, ws)
                                + torch.nn.functional.linear(xs, wp) + bias)
                    expected = reference_output(expected, op.o_interval, op.o_spec)
                    got = op(x, stat_collector=self.manager())
                    self.assertEqual(got.shape, expected.shape)
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_small_scales_and_half_inputs_preserve_output_and_int16_codes(self):
        small = QuantizedLinear(4, 2, bias=False, a_bit='e4m3', w_bit=4, o_bit='e4m3',
            outlier_ratio=.0001, scale_root_str=self.tmp.name)
        with torch.no_grad():
            small.weight.copy_(torch.tensor([[.3,.1,.2,.05], [.4,.2,.1,.3]]))
        x = torch.tensor([[1e-5, .5e-6, -.6e-6, .3e-6]])
        self.calibrate(small, x)
        self.assertLess(small.o_interval, 2**-16)
        self.assertTrue(bool((small(x, stat_collector=self.manager()) != 0).all()))

        wide = QuantizedLinear(4, 2, bias=False, a_bit=16, w_bit=16, o_bit='none',
            outlier_ratio=.0001, scale_root_str=self.tmp.name).half()
        with torch.no_grad():
            wide.weight.fill_(1)
        half_input = torch.tensor([[40000., 1003., .125, -1000.]], dtype=torch.float16)
        self.calibrate(wide, half_input)
        sm = self.manager()
        self.assertTrue(bool(torch.isfinite(wide(half_input, stat_collector=sm)).all()))
        records = sm.get_sparsity_summary()['records']
        weight = next(row for row in records if row['operand'] == 'weight')
        activation = next(row for row in records if row['operand'] == 'activation')
        self.assertEqual(weight['zero_bits'], 38)  # [0,32767,32767,32767] per output row
        expected = [0,32767,4,-32669]
        self.assertEqual(activation['zero_bits'], 64-sum((v & 0xffff).bit_count() for v in expected))

    def test_matmul_reduction_masks_and_counts_match_square_and_batched_reference(self):
        a = torch.tensor([[.3,-.7,0.,10.], [.2,.5,0.,9.]])
        b = torch.tensor([[.1,7.,.2,.4], [.4,.6,.8,.9], [.2,.3,.1,.4], [.8,.9,.1,.2]])
        for name in ('qk_matmul', 'pv_matmul'):
            for ndim in (2,3,4):
                for n in (3,4):
                    for formats in (('e4m3','e4m3','e4m3'), (8,4,'none')):
                        with self.subTest(name=name, ndim=ndim, n=n, formats=formats):
                            A = a.expand(*([2]*(ndim-2)), *a.shape).clone()
                            B = b[:, :n].expand(*([2]*(ndim-2)), 4, n).clone()
                            op = QuantizedMatMul(A_bit=formats[0], B_bit=formats[1], O_bit=formats[2],
                                outlier_ratio=.0001, scale_root_str=self.tmp.name)
                            op.set_layer_info(name, 0)
                            self.calibrate(op, A, B)
                            am = torch.zeros_like(A, dtype=torch.bool); am[...,3] = True
                            bm = torch.zeros_like(B, dtype=torch.bool)
                            bm[...,3,:] = True; bm[...,0,1] = True
                            an, bn = A.masked_fill(am, 0), B.masked_fill(bm, 0)
                            torch.testing.assert_close(op._split_outlier_operands(A, B)[1], bn, rtol=0, atol=0)
                            ac = reference_codes(an, op.A_interval, op.A_spec)
                            bc = reference_codes(bn, op.B_interval, op.B_spec)
                            ad, bd = ac*op.A_interval, bc*op.B_interval
                            ap, bp = A.masked_fill(~am,0), B.masked_fill(~bm,0)
                            expected = ad@bd + ap@bp + ap@bd + ad@bp
                            expected = reference_output(expected, op.O_interval, op.O_spec)
                            sm = self.manager()
                            got = op(A, B, stat_collector=sm)
                            torch.testing.assert_close(got, expected, rtol=0, atol=0)
                            rows = sm.get_sparsity_summary()['records']
                            b_role = 'K' if name == 'qk_matmul' else 'V'
                            row = next(r for r in rows if r['operand'] == b_role)
                            self.assertEqual(row['elements'], B.numel())
                            self.assertEqual(row['zero_elements'], int((bc == 0).sum()))
                            self.assertTrue(row['outlier_masked'])

    def test_sidepath_mapping_and_hardware_backends_remain_rejected(self):
        tensor = torch.ones(1,2)
        for kwargs in ({}, {'collect_mapping':False, 'ebb_config':{'enabled':True}},
                       {'collect_mapping':False, 'bitlet_config':{'enabled':True}}):
            sm = QuantStatManager(self.tmp.name, **kwargs)
            self.addCleanup(sm.close)
            with self.assertRaisesRegex(ValueError, 'sidepath'):
                sm.collect_quant_activation('q_proj',0,tensor,tensor,tensor,
                    type_spec(4),type_spec('e4m3'),None,None,2,1,outlier_masked=True)
            self.assertEqual(sm.total_bit_count, 0)
            self.assertEqual(sm.sparsity_records, {})

    def test_masked_and_unmasked_records_have_distinct_scopes(self):
        sm = self.manager()
        x = torch.tensor([[0.,1.5]])
        weight = torch.tensor([[0.,1.]])
        for masked in (True, False):
            sm.collect_quant_activation('q_proj',0,x,x,weight,type_spec(4),type_spec('e4m3'),
                                         None,None,2,1,outlier_masked=masked)
        doc = sm.get_sparsity_summary()
        self.assertEqual(len(doc['records']), 4)
        self.assertEqual(len(doc['phase_operands']), 4)
        self.assertEqual({r['outlier_masked'] for r in doc['records']}, {True,False})


if __name__ == '__main__':
    unittest.main()
