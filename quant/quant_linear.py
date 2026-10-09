"""
Quantized Linear Layer Implementation.

This module provides a quantized linear layer that supports:
- Multiple quantization modes (raw, scale_inspection, quant_forward)
- Configurable bit widths for activations, weights, and outputs
- Scale calibration and quantized inference
"""

import os
import sys
import math
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pickle
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional, Tuple, Dict
from .utils import Round, LINEAR_SHIFT_NUM


from .quant_spec import (
    QuantSpec,
    parse_quant_spec,
    safe_scale_from_tensor,
    safe_scale_per_token,
    merge_calibration_scale,
    quant_awo,
    fp8_dtype,
    fp8_max,
    int_qmax,
)

class QuantizedLinear(nn.Linear):
    """
    Quantized linear layer supporting multiple quantization modes.

    Modes:
        - 'raw': No quantization, standard linear layer
        - 'scale_inspection': Collect scale statistics for calibration
        - 'quant_forward': Quantized forward pass using pre-calibrated scales
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: str = "raw",
        a_bit: int = 8,
        w_bit: int = 8,
        o_bit: int = 8,
        d_bit: Optional[int] = None,
        p: Optional[int] = None,
        scale_root_str: str = "",
        outlier_ratio: float = 0.0,   # 新增，默认0表示不做outlier处理
        dynamic_activation: bool = False,  # per-token dynamic activation quantization
        weight_scale_granularity: str = "scalar",  # scalar | output_channel
        # -----------------------------------------------------------
        # Mixed-precision per-token activation quantization
        #
        # mixed_precision=True 时，按 token 重要性分配三种浮点精度：
        #   high (high_ratio)                  → FP16 (E5M10)
        #   mid  (1 - high_ratio - low_ratio) → FP8  (E4M3)
        #   low  (low_ratio)                  → FP4  (E2M1)
        # 重要性基于 activation L2 norm 排序。
        # 注意：这是用于构造论文所述 token-dependent precision workload 的
        # 自定义策略；论文没有规定这三档格式或比例。
        # -----------------------------------------------------------
        mixed_precision: bool = False,
        mp_high_ratio: float = 0.2,
        mp_low_ratio: float = 0.3,
        **kwargs
    ):
        """
        Initialize quantized linear layer.

        Args:
            in_features: Input feature size
            out_features: Output feature size
            bias: Whether to use bias
            mode: Quantization mode ('raw', 'scale_inspection', 'quant_forward')
            a_bit: Activation quantization bits
            w_bit: Weight quantization bits
            o_bit: Output quantization bits
            d_bit: Digit size for quantization
            p: Parallelism parameter
            scale_root_str: Root directory for scale files
        """
        super().__init__(in_features, out_features, bias)

        self.a_spec: QuantSpec = parse_quant_spec(a_bit)
        self.w_spec: QuantSpec = parse_quant_spec(w_bit)
        self.o_spec: QuantSpec = parse_quant_spec(o_bit)

        # Quantization parameters
        self.mode = mode
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.o_bit = o_bit
        self.digit_size = d_bit
        self.parallelism = p

        self.bitnet_weight_scale = None
        self.bitnet_online_quant = False
        self.is_bitnet = False


        # Quantization intervals (scales)
        self.w_interval: Optional[float] = None
        self.a_interval: Optional[float] = None
        self.o_interval: Optional[float] = None

        if self.a_spec.kind == "int":
            self.a_qmax = int_qmax(self.a_spec.bits)
        else:
            self.a_qmax = None
        if self.w_spec.kind == "int":
            self.w_qmax = int_qmax(self.w_spec.bits)
        else:
            self.w_qmax = None
        if self.o_spec.kind == "int":
            self.o_qmax = int_qmax(self.o_spec.bits)
        else:
            self.o_qmax = None

        # Scale root directory
        self.scale_root_str = scale_root_str

        # Round function
        self.round = Round

        # Layer identification
        self.layer_name = ""
        self.layer_idx = 0

        self.outlier_ratio = outlier_ratio
        # Element-wise weight outlier selection unioned into the channel mask
        # during both calibration and the sidepath forward (reference behavior:
        # outliermore=True keeps weight-side top-k elements on the FP16 path).
        self.outliermore = True
        self.calibration_policy = "recalibrate"
        self._calibration_action_cache = None

        # ------------------------------------------------------------
        # Per-token dynamic activation quantization
        #
        # dynamic_activation=True 时，在 quant_forward 中实时计算
        # 每个 token 独立的 a_interval（沿 hidden 维度取 max_abs），
        # 而不使用 calibration 阶段存盘的静态标量 a_interval。
        # 仅作用于 activation，weight / output scale 仍为静态。
        # ------------------------------------------------------------
        self.dynamic_activation = dynamic_activation
        self.weight_scale_granularity = str(
            weight_scale_granularity or "scalar"
        ).lower()
        if self.weight_scale_granularity not in {"scalar", "output_channel"}:
            raise ValueError(
                "weight_scale_granularity must be 'scalar' or "
                f"'output_channel', got {weight_scale_granularity!r}"
            )

        # Mixed-precision per-token activation quantization
        self.mixed_precision = mixed_precision
        self.mp_high_ratio = mp_high_ratio
        self.mp_low_ratio = mp_low_ratio

    def set_layer_info(self, layer_name: str, layer_idx: int):
        """Set layer name and index for scale file management."""
        self.layer_name = layer_name
        self.layer_idx = layer_idx


    def _scale_file_paths(self):
        return (
            os.path.join(self.scale_root_str, f"{self.layer_name}_w_scale_{self.layer_idx}.p"),
            os.path.join(self.scale_root_str, f"{self.layer_name}_a_scale_{self.layer_idx}.p"),
            os.path.join(self.scale_root_str, f"{self.layer_name}_o_scale_{self.layer_idx}.p"),
        )

    def _scale_files_exist(self) -> bool:
        return all(os.path.exists(p) for p in self._scale_file_paths())

    def _resolve_calibration_action(self) -> str:
        if self._calibration_action_cache is not None:
            return self._calibration_action_cache

        p = str(self.calibration_policy).lower()
        if p == "auto":
            action = "reuse" if self._scale_files_exist() else "recalibrate"
        elif p in {"reuse", "recalibrate"}:
            action = p
        else:
            raise ValueError(
                f"Invalid calibration_policy={self.calibration_policy} "
                f"for {self.layer_name}_{self.layer_idx}"
            )

        self._calibration_action_cache = action
        return action

    def hardware_profiling(
        self,
        x: torch.Tensor,
        HW: str = "Systolic",
    ) -> Dict[str, int]:
        """Profile tiled hardware cost for a linear forward.

        The input matrix x is split into tiles of shape [M, K], and the weight
        matrix is split into tiles of shape [K, N] to form output tiles of
        shape [M, N].

        Returns a dictionary containing estimated operation count and memory
        traffic in bytes.
        """
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])

        token_num, in_features = x.shape
        out_features = self.out_features

        if HW == "Systolic":
            from others.hardware.HW_proxy.Systolic_proxy import Systolic_HW_info, Systolic_proxy
            M, N, K = Systolic_HW_info()
            external_memory_accesses, compute_latency_ratio, latency_cycles = Systolic_proxy(M, N, K, x, self.weight)
        elif HW == "PAFCIM":
            from others.hardware.HW_proxy.PAFCIM_proxy import PAFCIM_HW_info, PAFCIM_proxy
            M, N, K = PAFCIM_HW_info()
            external_memory_accesses, compute_latency_ratio, latency_cycles = PAFCIM_proxy(M, N, K, x, self.weight)
        else:
            raise ValueError(f"Unknown HW type: {HW}")

        if M <= 0 or N <= 0 or K <= 0:
            raise ValueError("Tile dimensions M, N, K must be positive integers")

        tile_rows = (token_num + M - 1) // M
        tile_cols = (out_features + N - 1) // N
        tile_depths = (in_features + K - 1) // K

        total_macs = token_num * out_features * in_features

        return {
            "tile_rows": tile_rows,
            "tile_cols": tile_cols,
            "tile_depths": tile_depths,
            "macs": total_macs,
            "external_memory_accesses": external_memory_accesses,
            "compute_latency_ratio": compute_latency_ratio,
            "latency_cycles": latency_cycles
        }

    def forward(
        self,
        x: torch.Tensor,
        collect_stats: bool = False,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:
        """
        Forward pass with quantization.

        Args:
            x: Input tensor
            collect_stats: Whether to collect statistics
            stat_collector: Statistics collector object

        Returns:
            Output tensor (quantized or not depending on mode)
        """


        # Use stored stat_manager if not provided
        if stat_collector is None and hasattr(self, '_stat_manager'):
            stat_collector = self._stat_manager

        if self.mode == 'raw':
            if self.is_bitnet:
                return self.bitnet_fp16(x, stat_collector, collect= False)
            else:
                return F.linear(x, self.weight, self.bias)
        elif self.mode == "scale_inspection":
            if self.is_bitnet:
                return self.bitnet_fp16(x, stat_collector, collect= False)
            else:
                return self.scale_inspection(x, stat_collector)
        elif self.mode == "quant_forward":
            if self.is_bitnet:
                return self.bitnet_fp16(x, stat_collector, collect= True)
            else:
                return self.quant_forward(x, stat_collector)
                # return self.raw_forward(x, stat_collector)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

    def quant_weight(self, w: torch.Tensor) -> torch.Tensor:
        # Quantize weight tensor.
        return (w / self.w_interval).round_().clamp_(
            -self.w_qmax, self.w_qmax - 1
        )

    def quant_input(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input tensor.
        return (x / self.a_interval).round_().clamp_(
            -self.a_qmax, self.a_qmax - 1
        )


    def quant_bias(self, b: torch.Tensor) -> torch.Tensor:
        # Quantize bias tensor.
        biasfp32 = b.to(torch.float32)
        bias_sim = self.round(
            biasfp32 / (self.a_interval * self.w_interval)
        )
        return bias_sim

    def scale_inspection_bitnet(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:
        """
        Inspect and collect scale statistics for calibration.

        w_scale and a_scale are from FP values.
        o_scale is from quantized output (to match quant_forward behavior).
        But we return FP output for numerical stability.
        """

        if self.is_bitnet:
            from transformers.integrations.bitnet import ActQuant, WeightQuant
        if self.bitnet_online_quant:
            weight = WeightQuant.apply(self.weight)
        else:
            weight = WeightQuant.apply(self.weight)

        x_quant = ActQuant.apply(x)

        out = F.linear(x_quant, weight, self.bias)
        action = self._resolve_calibration_action()
        if action == "reuse":
            return out
        # Calculate weight and activation scales from FP values
        if self.outlier_ratio > 0.0:
            channel_mask = self.get_outlier_mask_channel(x, self.outlier_ratio)

            x_channel_mask = channel_mask.view(1, 1, -1)   # [1, 1, H]
            w_channel_mask = channel_mask.view(1, -1)      # [1, H]
            del channel_mask
            if self.outliermore:
                w_outlier_mask = self._get_outlier_mask_1d(self.weight, self.outlier_ratio)
                w_channel_mask = w_channel_mask | w_outlier_mask
                del w_outlier_mask
            x_normal_fp = x * (~x_channel_mask).to(dtype=x.dtype)
            x_normal_fp = x_normal_fp.to(torch.float32)

            w_normal_fp = self.weight * (~w_channel_mask).to(dtype=self.weight.dtype)
            w_normal_fp = w_normal_fp.to(torch.float32)

            del w_channel_mask
            del x_channel_mask

            self.a_interval = safe_scale_from_tensor(x_normal_fp, self.a_spec)
            if self.a_interval == 0:
                self.a_interval = None

            self.w_interval = safe_scale_from_tensor(w_normal_fp, self.w_spec)

            del x_normal_fp
            del w_normal_fp

            channel_mask = self.get_outlier_mask_channel(out, self.outlier_ratio)

            normal_idx = torch.nonzero(~channel_mask, as_tuple=False).flatten()

            o_normal_fp = out.index_select(dim=-1, index=normal_idx).to(torch.float32)

            self.o_interval = safe_scale_from_tensor(o_normal_fp, self.o_spec)
        else:
            self.a_interval = safe_scale_from_tensor(x_quant, self.a_spec)
            self.w_interval = 1
            self.o_interval = safe_scale_from_tensor(out, self.o_spec)

            # Collect statistics if collector is provided
        if stat_collector is not None:
            stat_collector.collect_linear_stats(
                self.layer_name,
                self.layer_idx,
                self.w_interval,
                self.a_interval,
                self.o_interval
            )

        # Return FP output for numerical stability during calibration
        return out


    def scale_inspection(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:
        """
        Inspect and collect scale statistics for calibration.

        w_scale and a_scale are from FP values.
        o_scale is from quantized output (to match quant_forward behavior).
        But we return FP output for numerical stability.
        """
        out = F.linear(x, self.weight, self.bias)
        action = self._resolve_calibration_action()
        if action == "reuse":
            return out

        # ------------------------------------------------------------
        # Mixed-precision per-token calibration 分支
        #
        # 在 mixed_precision 模式下：
        #   - a_interval 是 per-token 动态计算的，推理时不使用存盘值，
        #     但仍需写入一个占位值让 save_scales 正常工作。
        #   - w_interval 用全精度 weight 直接计算（与静态一致）。
        #   - o_interval 必须基于"量化后 x 的输出"来计算，
        #     以反映 mixed_precision 下真实的输出分布。
        # ------------------------------------------------------------
        if self.mixed_precision:
            return self._scale_inspection_mixed_precision(
                x, stat_collector
            )

        if self.dynamic_activation and self.outlier_ratio == 0.0:
            return self._scale_inspection_dynamic(
                x, stat_collector
            )

        # Calibration and inference use the same full-shape masks.
        if self.outlier_ratio > 0.0:
            x_normal, w_normal, _, _ = self._split_outlier_operands(x)
            self.a_interval = merge_calibration_scale(
                self.a_interval, safe_scale_from_tensor(x_normal, self.a_spec))
            self.w_interval = merge_calibration_scale(
                self.w_interval, self._weight_scale(w_normal))
            channels = self.get_outlier_mask_channel(out, self.outlier_ratio)
            out_mask = channels.view(*([1] * (out.ndim - 1)), -1)
            self.o_interval = merge_calibration_scale(
                self.o_interval, safe_scale_from_tensor(out.masked_fill(out_mask, 0), self.o_spec))
        else:
            self.a_interval = merge_calibration_scale(
                self.a_interval, safe_scale_from_tensor(x, self.a_spec))
            self.w_interval = merge_calibration_scale(
                self.w_interval, self._weight_scale(self.weight))
            self.o_interval = merge_calibration_scale(
                self.o_interval, safe_scale_from_tensor(out, self.o_spec))

            # Collect statistics if collector is provided
        if stat_collector is not None:
            stat_collector.collect_linear_stats(
                self.layer_name,
                self.layer_idx,
                self.w_interval,
                self.a_interval,
                self.o_interval
            )

        # Return FP output for numerical stability during calibration
        return out

    def raw_forward(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:
        """
        Forward pass without quantization.

        This method is used when the mode is 'raw' or during scale inspection.

        Args:
            x: Input tensor
            stat_collector: Optional statistics collector

        Returns:
            Output tensor (unquantized)
        """
        out = F.linear(x, self.weight, self.bias)

        in_features = self.weight.size(1)
        out_features = self.weight.size(0)

        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name,
                self.layer_idx,
                x,
                x,
                self.weight,
                self.w_spec,
                self.a_spec,
                self.digit_size,
                self.parallelism,
                in_features,
                out_features
            )
        return out

    def _is_w4_bf16_weight_only(self) -> bool:
        """Return True for BF16 activation/output + per-channel INT4 weight."""
        return (
            self.weight_scale_granularity == "output_channel"
            and self.a_spec.kind == "bf"
            and not self.a_spec.enabled
            and self.w_spec.kind == "int"
            and self.w_spec.enabled
            and self.w_spec.bits == 4
            and self.o_spec.kind == "bf"
            and not self.o_spec.enabled
        )

    def _prequantize_w4_output_channel(self):
        """Convert weights in place to INT4 codes and cache row scales.

        The repository's symmetric INT convention is ``[-qmax, qmax-1]`` with
        ``scale=max_abs/(qmax-0.5)``.  For INT4 this gives codes in [-8, 7].
        Codes are exactly representable in BF16; the per-output-channel scale
        remains FP32 and is applied after the BF16 GEMM.
        """
        if getattr(self, "_w4_weights_prequantized", False):
            return

        qmax = int_qmax(self.w_spec.bits)
        with torch.no_grad():
            row_max = self.weight.detach().abs().amax(dim=1).float()
            scale = torch.clamp(row_max / (qmax - 0.5), min=1e-12)

            # Process output-channel chunks to avoid materializing a full FP32
            # copy of a large projection weight.
            chunk_rows = max(1, 1_048_576 // max(1, self.in_features))
            for start in range(0, self.out_features, chunk_rows):
                stop = min(start + chunk_rows, self.out_features)
                codes = torch.div(
                    self.weight[start:stop].float(),
                    scale[start:stop, None],
                )
                codes = torch.round(codes).clamp_(-qmax, qmax - 1)
                self.weight[start:stop].copy_(
                    codes.to(self.weight.dtype)
                )

        self._w4_output_channel_scale = scale
        self._w4_weights_prequantized = True

    def w4_output_channel_stats(self) -> Dict[str, object]:
        """Return compact metadata for the W4-BF16 weight-only path."""
        if not getattr(self, "_w4_weights_prequantized", False):
            return {
                "enabled": self._is_w4_bf16_weight_only(),
                "prequantized": False,
            }
        scale = self._w4_output_channel_scale.float()
        return {
            "enabled": True,
            "prequantized": True,
            "granularity": "output_channel",
            "bits": 4,
            "activation_dtype": "bf16",
            "output_dtype": "bf16",
            "scale_min": float(scale.min().item()),
            "scale_mean": float(scale.mean().item()),
            "scale_max": float(scale.max().item()),
        }

    def _w4_bf16_forward(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """BF16 input/output forward with per-output-channel INT4 weights."""
        self._prequantize_w4_output_channel()

        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name,
                self.layer_idx,
                x,
                x,
                self.weight,
                self.w_spec,
                self.a_spec,
                self.digit_size,
                self.parallelism,
                self.in_features,
                self.out_features,
            )

        # self.weight now contains exact INT4 integer codes stored in BF16.
        # Keep the GEMM in BF16 and apply the small per-channel FP32 scale
        # afterwards, avoiding a second full-size dequantized weight tensor.
        # In normal model wrapping this weight is already BF16.  The cast
        # keeps direct module construction correct when x is BF16 but the
        # QuantizedLinear was created in FP32.
        weight_codes = self.weight.to(dtype=x.dtype)
        out = F.linear(x, weight_codes, None).float()
        scale_view = self._w4_output_channel_scale.view(
            *([1] * (out.dim() - 1)), -1
        )
        out = out * scale_view
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype)

    def quant_forward(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:

        if self._is_w4_bf16_weight_only():
            return self._w4_bf16_forward(x, stat_collector)

        self._load_scales()

        # ------------------------------------------------------------
        # Mixed-precision per-token activation quantization 分支
        # 按 token 重要性分配 FP8 / INT8 / INT4 三档精度
        # ------------------------------------------------------------
        if self.mixed_precision:
            return self._quant_forward_mixed_precision(
                x, stat_collector
            )

        # ------------------------------------------------------------
        # Per-token dynamic activation quantization 分支
        # 不使用静态 a_interval，实时按 hidden 维度计算 per-token scale
        # ------------------------------------------------------------
        if self.dynamic_activation and self.outlier_ratio == 0.0:
            return self._quant_forward_dynamic(x, stat_collector)

        if self.outlier_ratio > 0.0:
            return self._quant_forward_with_outlier(x, stat_collector)

        x_code = quant_awo(x, self.a_interval, self.a_spec, out_dtype=torch.float32)
        w_code, w_deq = self._quantized_weight()
        x_deq = x_code * self.a_interval
        out_real = F.linear(x_deq, w_deq, self.bias.float() if self.bias is not None else None)
        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name, self.layer_idx, x_code, x_code, w_code,
                self.w_spec, self.a_spec, self.digit_size, self.parallelism,
                self.in_features, self.out_features,
            )
        return self._quantize_output_from_real(out_real, x.dtype)

    def _weight_scale(self, weight):
        if self.weight_scale_granularity == "output_channel" and self.w_spec.enabled:
            return safe_scale_per_token(weight.float(), self.w_spec).squeeze(-1)
        return safe_scale_from_tensor(weight, self.w_spec)

    def _quantized_weight(self):
        scale = torch.as_tensor(self.w_interval,device=self.weight.device,dtype=torch.float32)
        broadcast = scale[:,None] if scale.ndim == 1 else scale
        codes = quant_awo(self.weight,broadcast,self.w_spec,out_dtype=torch.float32)
        return codes, codes * broadcast

    def _mixed_precision_specs(self):
        """Return the three activation formats used by the synthetic MP policy.

        ``e5m10`` is real FP16 pass-through.  It must not be routed through
        ``fp8_dtype``/``fp8_max`` in ``quant_spec.py``.
        """
        return (
            QuantSpec(kind="fp", fmt="e5m10", enabled=False),
            QuantSpec(kind="fp", fmt="e4m3", enabled=True),
            QuantSpec(kind="fp", fmt="e2m1", enabled=True),
        )

    @staticmethod
    def _safe_token_scale_from_max(
        x: torch.Tensor,
        format_max: float,
    ) -> torch.Tensor:
        """Return a finite positive per-token scale with shape ``[..., 1]``."""
        x_f32 = torch.nan_to_num(
            x.detach().to(torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        max_abs = x_f32.abs().amax(dim=-1, keepdim=True)
        scale = max_abs / float(format_max)
        return torch.where(
            torch.isfinite(scale) & (scale > 0),
            scale,
            torch.ones_like(scale),
        )

    @staticmethod
    def _fake_quantize_e2m1(normalized: torch.Tensor) -> torch.Tensor:
        """Quantize normalized values to the signed E2M1 finite grid.

        The magnitude grid is ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}``.  Midpoint
        thresholds match the E2M1 encoder used by ``QuantStatManager``.
        """
        v = torch.nan_to_num(
            normalized.to(torch.float32),
            nan=0.0,
            posinf=6.0,
            neginf=-6.0,
        ).clamp(-6.0, 6.0)
        abs_v = v.abs()
        qmag = torch.zeros_like(abs_v)
        qmag = torch.where(abs_v >= 0.25, torch.full_like(qmag, 0.5), qmag)
        qmag = torch.where(abs_v >= 0.75, torch.full_like(qmag, 1.0), qmag)
        qmag = torch.where(abs_v >= 1.25, torch.full_like(qmag, 1.5), qmag)
        qmag = torch.where(abs_v >= 1.75, torch.full_like(qmag, 2.0), qmag)
        qmag = torch.where(abs_v >= 2.50, torch.full_like(qmag, 3.0), qmag)
        qmag = torch.where(abs_v >= 3.50, torch.full_like(qmag, 4.0), qmag)
        qmag = torch.where(abs_v >= 5.00, torch.full_like(qmag, 6.0), qmag)
        return torch.copysign(qmag, v)

    def _quantize_activation_per_token(
        self,
        x: torch.Tensor,
        spec: QuantSpec,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize one token subset and return ``(code, scale, dequantized)``.

        This function intentionally handles FP16 and FP4 locally:

        * E5M10 is explicit FP16 pass-through with scale 1.
        * E4M3 uses a per-token max-abs scale and native PyTorch FP8.
        * E2M1 uses a per-token max-abs scale and a software FP4 grid.

        Only genuine FP8 formats are allowed to reach FP8-specific helpers in
        ``quant_spec.py``.  This avoids treating ``e5m10`` as an FP8 format.
        """
        x_f32 = torch.nan_to_num(
            x.to(torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        fmt = (getattr(spec, "fmt", "") or "").lower().strip()

        if fmt in {"e5m10", "fp16", "float16"}:
            # High-importance tokens are genuinely stored/processed as FP16.
            # A separate per-token scale is unnecessary for FP16 pass-through.
            scale = torch.ones(
                *x_f32.shape[:-1], 1,
                dtype=torch.float32,
                device=x_f32.device,
            )
            code = x_f32.to(torch.float16).to(torch.float32)
            return code, scale, code

        if fmt in {"e4m3", "fp8_e4m3", "float8_e4m3"}:
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError(
                    "This PyTorch build does not support torch.float8_e4m3fn"
                )
            scale = self._safe_token_scale_from_max(x_f32, 448.0)
            normalized = (x_f32 / scale).clamp(-448.0, 448.0)
            code = normalized.to(torch.float8_e4m3fn).to(torch.float32)
            return code, scale, code * scale

        if fmt in {"e2m1", "fp4", "fp4_e2m1"}:
            scale = self._safe_token_scale_from_max(x_f32, 6.0)
            code = self._fake_quantize_e2m1(x_f32 / scale)
            return code, scale, code * scale

        # Fixed-format dynamic activation may still use INT or another format.
        # Preserve the original generic path for formats supported by quant_spec.
        scale = safe_scale_per_token(x_f32, spec, dim=-1).to(torch.float32)
        scale = torch.where(
            torch.isfinite(scale) & (scale > 0),
            scale,
            torch.ones_like(scale),
        )
        code = quant_awo(
            x_f32,
            scale,
            spec,
            out_dtype=torch.float32,
            chunk_size=1_048_576,
        ).to(torch.float32)
        return code, scale, code * scale

    def _quantize_output_from_real(
        self,
        out_real: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Apply the configured output quantizer to a real-domain tensor."""
        if self.o_spec.kind in {"int", "fp", "bf"} and getattr(
            self.o_spec, "enabled", True
        ):
            out_code = quant_awo(
                out_real,
                self.o_interval,
                self.o_spec,
                out_dtype=torch.float32,
                chunk_size=1_048_576,
            ).to(torch.float32)
            return (out_code * self.o_interval).to(output_dtype)
        return out_real.to(output_dtype)

    def _scale_inspection_dynamic(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """Calibration for fixed-format per-token dynamic activation scaling.

        Future token scales are not calibrated. They are computed online from
        each token. Calibration only stores the static weight/output scales and
        an activation-scale placeholder required by the existing file format.
        """
        x_f32 = x.to(torch.float32)
        _, _, x_deq = self._quantize_activation_per_token(x_f32, self.a_spec)

        self.w_interval = self._weight_scale(self.weight)
        w_code, w_deq = self._quantized_weight()

        out_real = F.linear(x_deq, w_deq, bias=None)
        if self.bias is not None:
            out_real = out_real + self.bias.to(torch.float32)

        self.o_interval = safe_scale_from_tensor(out_real, self.o_spec)
        self.a_interval = safe_scale_from_tensor(x_f32, self.a_spec)

        if stat_collector is not None:
            stat_collector.collect_linear_stats(
                self.layer_name,
                self.layer_idx,
                self.w_interval,
                self.a_interval,
                self.o_interval,
            )

        return out_real.to(x.dtype)

    def _quant_forward_dynamic(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """Fixed-format per-token dynamic activation quantization.

        Each token obtains an online activation scale. Weight and output scales
        remain static. Bias is added in the real output domain before output
        quantization.
        """
        x_code, a_interval, x_deq = self._quantize_activation_per_token(
            x.to(torch.float32), self.a_spec
        )

        w_code, w_deq = self._quantized_weight()

        out_real = F.linear(x_deq, w_deq, bias=None)
        if self.bias is not None:
            out_real = out_real + self.bias.to(torch.float32)

        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name,
                self.layer_idx,
                x_code,
                x_code,
                w_code,
                self.w_spec,
                self.a_spec,
                self.digit_size,
                self.parallelism,
                self.weight.size(1),
                self.weight.size(0),
            )

        return self._quantize_output_from_real(out_real, x.dtype)

    def _classify_token_importance(
        self,
        x: torch.Tensor,
    ) -> tuple:
        """Classify tokens by peak-to-average ratio (PAR) while preserving execution order.

        PAR = max_abs / mean_abs per token.  A high PAR indicates a heavy-tailed
        distribution where a few outlier dimensions dominate — these tokens suffer
        most from per-token scale quantization and are assigned higher precision.
        Unlike L2 norm, PAR is not washed out by LayerNorm whitening.
        """
        if x.dim() != 2:
            raise ValueError(f"token classifier expects [N, H], got {tuple(x.shape)}")
        if not (0.0 <= self.mp_high_ratio <= 1.0):
            raise ValueError("mp_high_ratio must be in [0, 1]")
        if not (0.0 <= self.mp_low_ratio <= 1.0):
            raise ValueError("mp_low_ratio must be in [0, 1]")
        if self.mp_high_ratio + self.mp_low_ratio > 1.0:
            raise ValueError("mp_high_ratio + mp_low_ratio must be <= 1")

        x_f32 = x.detach().to(torch.float32).abs()
        mean_abs = x_f32.mean(dim=-1)
        max_abs = x_f32.amax(dim=-1)
        token_scores = max_abs / mean_abs.clamp(min=1e-8)
        n_tokens = token_scores.numel()
        if n_tokens == 0:
            empty = torch.empty(0, dtype=torch.long, device=x.device)
            return empty, empty, empty

        n_high = int(round(n_tokens * self.mp_high_ratio))
        n_low = int(round(n_tokens * self.mp_low_ratio))
        n_high = min(max(n_high, 0), n_tokens)
        n_low = min(max(n_low, 0), n_tokens - n_high)
        n_mid = n_tokens - n_high - n_low

        sorted_indices = torch.argsort(token_scores, descending=True)
        high_idx = sorted_indices[:n_high]
        mid_idx = sorted_indices[n_high:n_high + n_mid]
        low_idx = sorted_indices[n_high + n_mid:]
        return high_idx, mid_idx, low_idx

    def _quant_forward_mixed_precision(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """Per-token FP16/FP8/FP4 activation quantization.

        This implements a synthetic token-dependent-precision workload. The
        paper motivates variable token precision but does not prescribe these
        formats, ratios, or the L2-norm classifier.
        """
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).to(torch.float32)
        n_tokens = x_2d.shape[0]

        high_idx, mid_idx, low_idx = self._classify_token_importance(x_2d)
        high_spec, mid_spec, low_spec = self._mixed_precision_specs()

        x_deq_all = torch.empty_like(x_2d)
        code_all = torch.empty_like(x_2d)
        mp_codes = []  # [(local_code, global_indices, spec), ...]

        for indices, spec in (
            (high_idx, high_spec),
            (mid_idx, mid_spec),
            (low_idx, low_spec),
        ):
            if indices.numel() == 0:
                continue
            code, _, dequantized = self._quantize_activation_per_token(
                x_2d.index_select(0, indices), spec
            )
            x_deq_all.index_copy_(0, indices, dequantized)
            code_all.index_copy_(0, indices, code)
            mp_codes.append((code, indices, spec))

        w_code, w_deq = self._quantized_weight()

        out_real = F.linear(x_deq_all, w_deq, bias=None)
        if self.bias is not None:
            out_real = out_real + self.bias.to(torch.float32)

        if stat_collector is not None:
            stat_collector.collect_quant_activation_mixed_precision(
                self.layer_name,
                self.layer_idx,
                code_all,
                mp_codes,
                w_code,
                self.w_spec,
                self.digit_size,
                self.parallelism,
                self.weight.size(1),
                self.weight.size(0),
                mp_high_ratio=self.mp_high_ratio,
                mp_low_ratio=self.mp_low_ratio,
            )

        out = self._quantize_output_from_real(out_real, x.dtype)
        return out.reshape(*original_shape[:-1], self.out_features)

    def _scale_inspection_mixed_precision(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """Calibration aligned with the mixed-precision inference path."""
        # original_shape = x.shape
        # x_2d = x.reshape(-1, original_shape[-1]).to(torch.float32)
        # high_idx, mid_idx, low_idx = self._classify_token_importance(x_2d)
        # high_spec, mid_spec, low_spec = self._mixed_precision_specs()

        # x_deq_all = torch.empty_like(x_2d)
        # for indices, spec in (
        #     (high_idx, high_spec),
        #     (mid_idx, mid_spec),
        #     (low_idx, low_spec),
        # ):
        #     if indices.numel() == 0:
        #         continue
        #     _, _, dequantized = self._quantize_activation_per_token(
        #         x_2d.index_select(0, indices), spec
        #     )
        #     x_deq_all.index_copy_(0, indices, dequantized)

        # self.w_interval = safe_scale_from_tensor(
        #     self.weight.to(torch.float32), self.w_spec
        # )
        # w_code = quant_awo(
        #     self.weight,
        #     self.w_interval,
        #     self.w_spec,
        #     out_dtype=torch.float32,
        #     chunk_size=1_048_576,
        # ).to(torch.float32)

        out_real = F.linear(x, self.weight, self.bias)

        self.w_interval = self._weight_scale(self.weight)
        self.o_interval = safe_scale_from_tensor(out_real, self.o_spec)
        self.a_interval = 0

        if stat_collector is not None:
            stat_collector.collect_linear_stats(
                self.layer_name,
                self.layer_idx,
                self.w_interval,
                self.a_interval,
                self.o_interval,
            )

        return out_real

    def _split_outlier_operands(self, x):
        """Return full-shape normal/protected tensors; masked zeros stay present.

        The weight mask unions the activation-derived outlier channel mask with
        an element-wise weight top-k mask when ``outliermore`` is enabled
        (default), matching the reference calibration-side logic.
        """
        channels = self.get_outlier_mask_channel(x, self.outlier_ratio)
        x_mask = channels.view(*([1] * (x.ndim - 1)), -1)
        w_mask = channels.view(1, -1)
        if self.outliermore:
            w_mask = w_mask | self._get_outlier_mask_1d(
                self.weight, self.outlier_ratio)
        x_float, w_float = x.float(), self.weight.float()
        return (x_float.masked_fill(x_mask, 0),
                w_float.masked_fill(w_mask, 0),
                x_float.masked_fill(~x_mask, 0),
                w_float.masked_fill(~w_mask, 0))

    def _quant_forward_with_outlier(self, x, stat_collector=None):
        """Quantize normal operands and retain all protected cross terms."""
        x_normal, w_normal, x_protected, w_protected = self._split_outlier_operands(x)
        x_code = quant_awo(x_normal, self.a_interval, self.a_spec,
                           out_dtype=torch.float32)
        w_scale = torch.as_tensor(self.w_interval, device=self.weight.device,
                                  dtype=torch.float32)
        w_broadcast = w_scale[:, None] if w_scale.ndim == 1 else w_scale
        w_code = quant_awo(w_normal, w_broadcast, self.w_spec,
                           out_dtype=torch.float32)
        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name, self.layer_idx, x_code, x_code, w_code,
                self.w_spec, self.a_spec, self.digit_size, self.parallelism,
                self.in_features, self.out_features, outlier_masked=True,
            )

        x_deq, w_deq = x_code * self.a_interval, w_code * w_broadcast
        out_real = (F.linear(x_deq, w_deq)
                    + F.linear(x_protected, w_protected)
                    + F.linear(x_protected, w_deq)
                    + F.linear(x_deq, w_protected))
        if self.bias is not None:
            out_real = out_real + self.bias.float()
        if self.o_spec.enabled:
            channels = self.get_outlier_mask_channel(out_real, self.outlier_ratio)
            out_mask = channels.view(*([1] * (out_real.ndim - 1)), -1)
            out_code = quant_awo(out_real.masked_fill(out_mask, 0),
                                 self.o_interval, self.o_spec, out_dtype=torch.float32)
            out_real = torch.where(out_mask, out_real, out_code * self.o_interval)
        return out_real.to(x.dtype)

    def bitnet_forward(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None,
        collect: bool = False
    ) -> torch.Tensor:
        if self.is_bitnet:
            from transformers.integrations.bitnet import ActQuant, WeightQuant
        if self.bitnet_online_quant:
            weight = WeightQuant.apply(self.weight)
        else:
            weight = WeightQuant.apply(self.weight)

        x_quant = ActQuant.apply(x)



        M0 = torch.tensor(
            self.w_interval * self.a_interval / self.o_interval,
            device=x_quant.device,
            dtype=torch.float32,
        )
        M0 = self.round(M0 * LINEAR_SHIFT_NUM)

        x_code = quant_awo(
            x_quant,
            self.a_interval,
            self.a_spec,
            out_dtype=x.dtype,
            chunk_size=1_048_576,
        )

        w_code = weight

        if self.bias is not None:
            bias_sim = self.quant_bias(self.bias)
        else:
            bias_sim = None

        x_code = x_code.to(torch.float32)
        w_code = w_code.to(torch.float32)


        in_features = self.weight.size(1)
        out_features = self.weight.size(0)
        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name,
                self.layer_idx,
                x_code,
                x_code,
                self.a_spec,
                self.digit_size,
                self.parallelism,
                in_features,
                out_features
            )

        if bias_sim is not None:
            bias_code = bias_sim.to(torch.float32)
        else:
            bias_code = None

        acc_code = F.linear(x_code, w_code, bias_code)

        scale_to_output = self.a_interval * self.w_interval / self.o_interval

        if self.o_spec.kind == "int":
            M0 = torch.tensor(scale_to_output, device=x_quant.device, dtype=torch.float32)
            M0 = self.round(M0 * LINEAR_SHIFT_NUM)

            out_code = acc_code.mul(M0)
            out_code = torch.div(
                out_code,
                LINEAR_SHIFT_NUM,
                rounding_mode="floor",
            )

            out = out_code.mul(self.o_interval).to(x.dtype)
            return out

        if self.o_spec.kind == "fp":
            out_scaled = acc_code.mul(scale_to_output)

            dtype = fp8_dtype(self.o_spec.fmt)
            max_val = fp8_max(self.o_spec.fmt)

            out_code = out_scaled.clamp(-max_val, max_val).to(dtype).float()
            out = out_code.mul(self.o_interval).to(x.dtype)
            return out

        # output 不量化
        out = F.linear(x, self.weight, self.bias)

        output = F.linear(
            x_quant,
            weight,
            self.bias,
        )

        in_features = weight.size(1)
        out_features = weight.size(0)

        if self.layer_name == "gate_proj":
            pass

        if not self.bitnet_online_quant:
            if self.bitnet_weight_scale is None:
                raise RuntimeError(
                    f"Missing weight_scale in "
                    f"{self.layer_name}_{self.layer_idx}"
                )

            output = output * self.bitnet_weight_scale

        return output

    def bitnet_fp16(
        self,
        x: torch.Tensor,
        stat_collector: Optional[object] = None,
        collect: bool = False
    ) -> torch.Tensor:
        if self.is_bitnet:
            from transformers.integrations.bitnet import ActQuant, WeightQuant
        if self.bitnet_online_quant:
            weight = WeightQuant.apply(self.weight)
        else:
            weight = WeightQuant.apply(self.weight)

        x_quant = ActQuant.apply(x)

        w_code = weight

        in_features = self.weight.size(1)
        out_features = self.weight.size(0)
        if collect and stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name,
                self.layer_idx,
                x,
                x,
                self.weight,
                self.w_spec,
                self.a_spec,
                self.digit_size,
                self.parallelism,
                in_features,
                out_features
            )

        # output 不量化
        out = F.linear(x, weight, self.bias)

        if self.layer_name == "gate_proj":
            pass

        if not self.bitnet_online_quant:
            if self.bitnet_weight_scale is None:
                raise RuntimeError(
                    f"Missing weight_scale in "
                    f"{self.layer_name}_{self.layer_idx}"
                )

        return out

    def _get_outlier_mask_1d(self, tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        返回 bool mask，True 表示是 outlier（保留 FP16）。
        离群值定义为绝对值最大的元素，数量约占总元素数的 ratio（至少1个，最多 numel-1 个）。
        当所有元素绝对值相等时，返回全 False。
        """
        if ratio <= 0.0:
            return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)

        numel = tensor.numel()
        if numel == 0:
            return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)

        # 至少选1个，最多选 numel-1 个，确保正常部分非空
        k = max(1, min(int(numel * ratio), numel - 1))

        flat_abs = tensor.abs().flatten()
        # 获取第 k 大的值
        threshold = torch.topk(flat_abs, k).values.min()

        # 处理阈值等于最小值的情况
        min_val = flat_abs.min()
        if threshold == min_val:
            # 只选严格大于最小值的元素，避免全选
            outlier_mask_flat = flat_abs > min_val
        else:
            outlier_mask_flat = flat_abs >= threshold

        # 如果离群数量为0（如全等值），返回全False
        if outlier_mask_flat.sum() == 0:
            return torch.zeros(tensor.shape, dtype=torch.bool, device=tensor.device)

        # 恢复原始形状
        return outlier_mask_flat.view(tensor.shape)


    def get_outlier_mask_channel(self, tensor: torch.Tensor, ratio: float) -> torch.Tensor:
        """带 outlier 保护的量化前向计算。"""

        if not 0 <= ratio <= 1:
            raise ValueError("outlier_ratio must be in [0, 1]")
        if ratio == 0 or tensor.numel() == 0:
            return torch.zeros(tensor.shape[-1], dtype=torch.bool, device=tensor.device)

        # 兼容 [B, S, H] 和 [1, H] / [S, H] / [H] 等各种情况
        tensor_2d = tensor.reshape(-1, tensor.shape[-1])   # [N, H]
        channel_score = tensor_2d.abs().amax(dim=0)        # [H]

        k = max(1, int(channel_score.numel() * ratio))

        protected_idx = torch.topk(channel_score, k).indices

        channel_mask = torch.zeros_like(channel_score, dtype=torch.bool)
        channel_mask[protected_idx] = True

        return channel_mask

    def _load_scales(self):
        """Load quantization scales from files."""
        # Construct scale file paths
        w_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_w_scale_{self.layer_idx}.p"
        )
        a_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_a_scale_{self.layer_idx}.p"
        )
        o_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_o_scale_{self.layer_idx}.p"
        )

        # Load scales
        with open(w_scale_file, 'rb') as f:
            self.w_interval = pickle.load(f)

        with open(a_scale_file, 'rb') as f:
            self.a_interval = pickle.load(f)

        with open(o_scale_file, 'rb') as f:
            self.o_interval = pickle.load(f)

    def save_scales(self):
        """Save current scales to files."""
        if self.w_interval is None or self.a_interval is None or self.o_interval is None:
            raise ValueError("Scales not computed yet. Run scale_inspection first.")

        os.makedirs(self.scale_root_str, exist_ok=True)

        # Save weight scale
        w_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_w_scale_{self.layer_idx}.p"
        )
        with open(w_scale_file, 'wb') as f:
            pickle.dump(self.w_interval, f)

        # Save activation scale
        a_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_a_scale_{self.layer_idx}.p"
        )
        with open(a_scale_file, 'wb') as f:
            pickle.dump(self.a_interval, f)

        # Save output scale
        o_scale_file = os.path.join(
            self.scale_root_str,
            f"{self.layer_name}_o_scale_{self.layer_idx}.p"
        )
        with open(o_scale_file, 'wb') as f:
            pickle.dump(self.o_interval, f)

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'mode={self.mode}, '
            f'a_bit={self.a_bit}, '
            f'w_bit={self.w_bit}, '
            f'o_bit={self.o_bit}'
        )


    def log_quant_error(self, value: float, layer_name: str = None, layer_idx: int = None):
        import json as _json
        import os as _os
        layer_name = layer_name if layer_name is not None else self.layer_name
        layer_idx = layer_idx if layer_idx is not None else self.layer_idx
        json_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "quant-error", "error.jsonl"
        )
        _os.makedirs(_os.path.dirname(json_path), exist_ok=True)
        entry = {"layer_name": layer_name, "layer_idx": layer_idx, "value": float(value)}
        with open(json_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")

