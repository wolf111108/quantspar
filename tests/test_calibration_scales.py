"""Regression checks for calibration across multiple batches."""

import tempfile
import unittest

import torch

from quant.quant_linear import QuantizedLinear
from quant.quant_matmul import QuantizedMatMul
from quant.quant_spec import merge_calibration_scale


class CalibrationScaleTests(unittest.TestCase):
    def test_linear_keeps_earlier_peak_and_persists_it(self):
        with tempfile.TemporaryDirectory() as scale_dir:
            layer = QuantizedLinear(
                2, 2, bias=False, mode="scale_inspection",
                a_bit=8, w_bit=8, o_bit=8, scale_root_str=scale_dir,
            )
            with torch.no_grad():
                layer.weight.copy_(torch.tensor([[2., 0.], [0., 3.]]))
            first_output = layer.scale_inspection(torch.tensor([[8., 4.]]))
            first_activation = layer.a_interval
            first_output_scale = layer.o_interval
            layer.scale_inspection(torch.tensor([[1., 1.]]))
            self.assertEqual(layer.a_interval, first_activation)
            self.assertEqual(layer.o_interval, first_output_scale)
            self.assertGreater(first_output.abs().max().item(), 1)
            layer.save_scales()
            layer.a_interval = layer.o_interval = None
            layer._load_scales()
            self.assertEqual(layer.a_interval, first_activation)
            self.assertEqual(layer.o_interval, first_output_scale)

    def test_matmul_keeps_independent_operand_and_output_peaks(self):
        matmul = QuantizedMatMul(
            mode="scale_inspection", A_bit=8, B_bit=8, O_bit=8,
        )
        matmul.scale_inspection(
            torch.tensor([[8., 1.]]), torch.tensor([[2.], [1.]]))
        first = (matmul.A_interval, matmul.B_interval, matmul.O_interval)
        matmul.scale_inspection(
            torch.tensor([[1., 1.]]), torch.tensor([[1.], [1.]]))
        self.assertEqual(
            (matmul.A_interval, matmul.B_interval, matmul.O_interval), first)

    def test_masked_weight_channel_scales_keep_earlier_batch(self):
        layer = QuantizedLinear(
            3, 2, bias=False, mode="scale_inspection",
            a_bit=8, w_bit=8, o_bit=8,
            outlier_ratio=0.34, weight_scale_granularity="output_channel",
        )
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[9., 2., 1.], [8., 5., 1.]]))
        layer.scale_inspection(torch.tensor([[10., 1., 1.]]))
        first = layer.w_interval.clone()
        layer.scale_inspection(torch.tensor([[1., 1., 10.]]))
        self.assertTrue(torch.equal(layer.w_interval, first))
        self.assertEqual(tuple(layer.w_interval.shape), (2,))

    def test_per_channel_scale_max_and_shape_check(self):
        high = torch.tensor([1., 4.])
        low = torch.tensor([0.5, 2.])
        self.assertTrue(torch.equal(
            merge_calibration_scale(high, low), high))
        self.assertTrue(torch.equal(
            merge_calibration_scale(low, high), high))
        with self.assertRaises(ValueError):
            merge_calibration_scale(high, torch.ones(3))


if __name__ == "__main__":
    unittest.main()
