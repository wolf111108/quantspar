"""
Quantized Matrix Multiplication Implementation.

This module provides quantized matrix multiplication layers for:
- Attention score computation: Q @ K^T
- Attention output computation: P @ V

Supported modes:
- raw
- scale_inspection
- quant_forward

This version uses torch.matmul instead of torch.bmm, so it supports:
- 2D: [M, K] @ [K, N]
- 3D: [B, M, K] @ [B, K, N]
- 4D: [B, H, M, K] @ [B, H, K, N]
"""
from typing import Optional, Tuple, Dict
import json
import os
import pickle
from typing import Optional

import torch
import torch.nn as nn

from .utils import Round, MATMUL_SHIFT_NUM

from .quant_spec import (       # add
    QuantSpec,                  # add
    parse_quant_spec,           # add
    safe_scale_from_tensor,     # add
    safe_scale_per_token,
    quant_awo,                  # add
    fp8_dtype,                  # add
    fp8_max,                    # add
)


class MatMul(nn.Module):
    """Simple non-quantized matrix multiplication module."""

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


class QuantizedMatMul(nn.Module):
    """
    Quantized matrix multiplication layer.

    This layer is intended for:
      - qk_matmul: Q @ K^T
      - pv_matmul: P @ V

    Supports flexible A/B/O formats:
      - INT: 8, 4, ...
      - FP: e4m3, e5m2
      - disabled output: fp16 / none, depending on QuantSpec behavior
    """

    def __init__(
        self,
        mode: str = "raw",
        A_bit: Optional[int] = None,
        B_bit: Optional[int] = None,
        O_bit: Optional[int] = None,
        scale_root_str: Optional[str] = None,
        d_bit: Optional[int] = None,
        p: Optional[int] = None,
        outlier_ratio: float = 0.0,
        mixed_precision: bool = False,
        mp_high_ratio: float = 0.1,
        mp_low_ratio: float = 0.65,
    ):
        super().__init__()

        self.mode = mode

        self.A_bit = A_bit
        self.B_bit = B_bit
        self.O_bit = O_bit

        self.A_spec: QuantSpec = parse_quant_spec(A_bit)  # add
        self.B_spec: QuantSpec = parse_quant_spec(B_bit)  # add
        self.O_spec: QuantSpec = parse_quant_spec(O_bit)  # add

        self.digit_size = d_bit
        self.parallelism = p

        self.scale_root_str = scale_root_str or ""
        self.round = Round
        self.is_bitnet = False  # 默认不是Bitnet

        self.A_interval: Optional[float] = None
        self.B_interval: Optional[float] = None
        self.O_interval: Optional[float] = None

        # self.A_qmax = 2 ** (self.A_bit - 1) if self.A_bit is not None else None  # delete
        # self.B_qmax = 2 ** (self.B_bit - 1) if self.B_bit is not None else None  # delete
        # self.O_qmax = 2 ** (self.O_bit - 1) if self.O_bit is not None else None  # delete

        self.layer_name = ""
        self.layer_idx = 0

        self.outlier_ratio = outlier_ratio

        self.calibration_policy = "recalibrate"
        self._calibration_action_cache = None

        # Mixed-precision per-token activation quantization
        self.mixed_precision = mixed_precision
        self.mp_high_ratio = mp_high_ratio
        self.mp_low_ratio = mp_low_ratio

    def set_layer_info(self, layer_name: str, layer_idx: int):
        self.layer_name = layer_name
        self.layer_idx = layer_idx

    def _matmul(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        General batched matmul.

        Supports 2D / 3D / 4D / higher-dimensional batched matmul.
        """

        return torch.matmul(A, B)

    def _check_bits(self):
        if self.A_bit is None or self.B_bit is None or self.O_bit is None:
            raise ValueError(
                f"{self.layer_name}_{self.layer_idx}: "
                f"A_bit/B_bit/O_bit must be set. "
                f"Got A_bit={self.A_bit}, B_bit={self.B_bit}, O_bit={self.O_bit}."
            )

        if self.A_spec is None or self.B_spec is None or self.O_spec is None:  # add
            raise ValueError(                                                # add
                f"{self.layer_name}_{self.layer_idx}: invalid QuantSpec. "    # add
                f"A_spec={self.A_spec}, B_spec={self.B_spec}, O_spec={self.O_spec}"  # add
            )                                                                # add

    @staticmethod
    def _safe_interval_from_tensor(x: torch.Tensor, qmax: int) -> float:
        """
        Old INT-only helper. Kept for compatibility, but flexible path uses
        safe_scale_from_tensor().
        """
        max_abs = x.detach().abs().max()
        if max_abs == 0 or torch.isnan(max_abs) or torch.isinf(max_abs):
            return 1.0
        return (max_abs / (qmax - 0.5)).item()

    def _scale_file_paths(self):
        return (
            os.path.join(
                self.scale_root_str,
                f"{self.layer_name}_A_scale_{self.layer_idx}.p",
            ),
            os.path.join(
                self.scale_root_str,
                f"{self.layer_name}_B_scale_{self.layer_idx}.p",
            ),
            os.path.join(
                self.scale_root_str,
                f"{self.layer_name}_O_scale_{self.layer_idx}.p",
            ),
        )

    def _scale_files_exist(self) -> bool:
        return all(os.path.exists(p) for p in self._scale_file_paths())

    def _resolve_calibration_action(self) -> str:
        if self._calibration_action_cache is not None:
            return self._calibration_action_cache

        policy = str(self.calibration_policy).lower()

        if policy == "auto":
            action = "reuse" if self._scale_files_exist() else "recalibrate"
        elif policy in {"reuse", "recalibrate"}:
            action = policy
        else:
            raise ValueError(
                f"Invalid calibration_policy={self.calibration_policy} "
                f"for {self.layer_name}_{self.layer_idx}"
            )

        self._calibration_action_cache = action
        return action

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        collect_stats: bool = False,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        if stat_collector is None and hasattr(self, "_stat_manager"):
            stat_collector = self._stat_manager

        if self.mode == "raw":
            if self.is_bitnet:
                return self.bitnet_forward(A, B, stat_collector)
            else:
                return self._matmul(A, B)

        if self.mode == "scale_inspection":
            if self.is_bitnet:
                return self.bitnet_forward(A, B, stat_collector)
            else:
                # return self.scale_inspection(A, B, stat_collector)
                return self.scale_inspection(A, B, stat_collector)

        if self.mode == "quant_forward":
            if self.is_bitnet:
                return self.bitnet_forward(A, B, stat_collector)
            else:
                # raw_forward 不量化, qk/pv 的混精统计(SACIM latency)会全为 0,
                # 加速比测量必须走 quant_forward → _quant_forward_mixed_precision
                return self.quant_forward(A, B, stat_collector)
        raise NotImplementedError(f"Mode {self.mode} not implemented")

    # def quant_input(                                            # delete
    #     self,                                                    # delete
    #     x: torch.Tensor,                                         # delete
    #     interval: float,                                         # delete
    #     qmax: int,                                               # delete
    # ) -> torch.Tensor:                                           # delete
    #     return (x / interval).round().clamp(-qmax, qmax - 1)      # delete

    def scale_inspection(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        # """
        # Collect A/B/O scales.

        # A_interval is collected from A.
        # B_interval is collected from B.
        # O_interval is collected from raw output O = A @ B.
        # """
        # self._check_bits()

        # out = self._matmul(A, B)

        # action = self._resolve_calibration_action()
        # if action == "reuse":
        #     return self._matmul(A, B)

        # if self.outlier_ratio > 0.0:
        #     A_normal_mask = ~self._get_outlier_mask_1d(A, self.outlier_ratio)
        #     A_normal = A * A_normal_mask
        #     if A_normal.numel() == 0:
        #         pass
        #     self.A_interval = safe_scale_from_tensor(A_normal, self.A_spec)

        #     B_normal_mask = ~self._get_outlier_mask_1d(B, self.outlier_ratio)
        #     B_normal = B * B_normal_mask
        #     if B_normal.numel() == 0:
        #         pass
        #     self.B_interval = safe_scale_from_tensor(B_normal, self.B_spec)

        #     O_normal_mask = ~self._get_outlier_mask_1d(out, self.outlier_ratio)
        #     O_normal = out * O_normal_mask
        #     if O_normal.numel() == 0:
        #         pass
        #     self.O_interval = safe_scale_from_tensor(O_normal, self.O_spec)
        # else:
        #     self.A_interval = safe_scale_from_tensor(A, self.A_spec)
        #     self.B_interval = safe_scale_from_tensor(B, self.B_spec)
        #     self.O_interval = safe_scale_from_tensor(out, self.O_spec)

        # if stat_collector is not None:
        #     stat_collector.collect_matmul_stats(
        #         self.layer_name,
        #         self.layer_idx,
        #         self.A_interval,
        #         self.B_interval,
        #         self.O_interval,
        #     )

        # return out

        """
        Inspect and collect scale statistics for calibration.
        
        w_scale and a_scale are from FP values.
        o_scale is from quantized output (to match quant_forward behavior).
        But we return FP output for numerical stability.
        """
        out = self._matmul(A, B)
        action = self._resolve_calibration_action()
        if action == "reuse":
            return out
        
        # if self.mixed_precision and self.outlier_ratio == 0.0:
        #     return self._scale_inspection_mixed_precision(
        #         A, stat_collector
        #     )

        # if self.dynamic_activation and self.outlier_ratio == 0.0:
        #     return self._scale_inspection_dynamic(
        #         A, stat_collector
        #     )
        
        # Calculate weight and activation scales from FP values
        if self.mixed_precision and self.outlier_ratio == 0.0:
            return self._scale_inspection_mixed_precision(
                A, B, stat_collector
            )
        if self.outlier_ratio > 0.0:
            A_normal, B_normal, _, _ = self._split_outlier_operands(A, B)
            self.A_interval = safe_scale_from_tensor(A_normal, self.A_spec)
            self.B_interval = safe_scale_from_tensor(B_normal, self.B_spec)
            channels = self.get_outlier_mask_channel(out, self.outlier_ratio)
            out_mask = channels.view(*([1] * (out.ndim - 1)), -1)
            self.O_interval = safe_scale_from_tensor(out.masked_fill(out_mask, 0), self.O_spec)
        else:
            self.A_interval = safe_scale_from_tensor(A, self.A_spec)
            self.B_interval = safe_scale_from_tensor(B, self.B_spec)
            self.O_interval = safe_scale_from_tensor(out, self.O_spec)
        
        if stat_collector is not None:
            stat_collector.collect_matmul_stats(
                self.layer_name,
                self.layer_idx,
                self.A_interval,
                self.B_interval,
                self.O_interval,
            )
        
        # Return FP output for numerical stability during calibration
        return out

    def _scale_inspection_mixed_precision(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """Calibration aligned with the mixed-precision inference path."""

        out_real = self._matmul(A, B)

        self.O_interval = safe_scale_from_tensor(out_real, self.O_spec)
        self.B_interval = safe_scale_from_tensor(B, self.B_spec)
        self.A_interval = 0

        if stat_collector is not None:
            stat_collector.collect_matmul_stats(
                self.layer_name,
                self.layer_idx,
                self.A_interval,
                self.B_interval,
                self.O_interval,
            )

        return out_real

    def raw_forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None
    ) -> torch.Tensor:
        """
        Forward pass without quantization.
        
        This method is used when the mode is 'raw' or during scale inspection.
        
        Args:
            A: Input tensor
            B: Input tensor
            stat_collector: Optional statistics collector

        Returns:
            Output tensor (unquantized)
        """
        out = self._matmul(A, B)

        in_features = A.size(-1)
        out_features = B.size(-1)

        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name,
                self.layer_idx,
                A,
                A,
                B,
                self.B_spec,
                self.A_spec,
                self.digit_size,
                self.parallelism,
                in_features,
                out_features
            )

        return out

    def quant_forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """
        Flexible quantized matmul.

        A ~= A_interval * A_code
        B ~= B_interval * B_code
        acc_code = A_code @ B_code

        Then output is handled according to O_spec:
          - INT: floor/round into integer output code
          - FP: cast into FP8 output code
          - disabled: no output quantization
        """

        self._check_bits()
        self._load_scales()

        if self.mixed_precision and self.outlier_ratio == 0.0:
            return self._quant_forward_mixed_precision(
                A, B, stat_collector
            )

        if self.outlier_ratio > 0.0:
            return self._quant_forward_with_outlier(A, B, stat_collector)
        A_code = quant_awo(A,self.A_interval,self.A_spec,out_dtype=torch.float32)
        B_code = quant_awo(B,self.B_interval,self.B_spec,out_dtype=torch.float32)
        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name,self.layer_idx,A_code,A_code,B_code,self.B_spec,self.A_spec,
                self.digit_size,self.parallelism,A.size(-1),B.size(-1),
            )
        out_real = self._matmul(A_code*self.A_interval,B_code*self.B_interval)
        if self.O_spec.enabled:
            out_code = quant_awo(out_real,self.O_interval,self.O_spec,out_dtype=torch.float32)
            out_real = out_code*self.O_interval
        return out_real.to(A.dtype)


    def _split_outlier_operands(self, A, B):
        """Protect paired reduction channels, including square QK/PV operands.

        When ``outliermore`` is enabled (default), B additionally keeps its own
        element-wise top-k outliers on the FP16 path, matching the Linear-side
        reference behavior.
        """
        if A.ndim < 2 or B.ndim < 2 or A.shape[-1] != B.shape[-2]:
            raise ValueError("Outlier MatMul requires A[..., M, K] and B[..., K, N]")
        channels = self.get_outlier_mask_channel(A, self.outlier_ratio)
        A_mask = channels.view(*([1] * (A.ndim - 1)), -1)
        B_mask = channels.view(*([1] * (B.ndim - 2)), -1, 1)
        if getattr(self, "outliermore", True):
            B_mask = B_mask | self._get_outlier_mask_1d(B, self.outlier_ratio)
        A_float, B_float = A.float(), B.float()
        return (A_float.masked_fill(A_mask, 0),
                B_float.masked_fill(B_mask, 0),
                A_float.masked_fill(~A_mask, 0),
                B_float.masked_fill(~B_mask, 0))

    def _quant_forward_with_outlier(self, A, B, stat_collector=None):
        """Count masked normal codes once; execute every protected cross term."""
        A_normal, B_normal, A_protected, B_protected = self._split_outlier_operands(A, B)
        A_code = quant_awo(A_normal, self.A_interval, self.A_spec,
                           out_dtype=torch.float32)
        B_code = quant_awo(B_normal, self.B_interval, self.B_spec,
                           out_dtype=torch.float32)
        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                self.layer_name, self.layer_idx, A_code, A_code, B_code,
                self.B_spec, self.A_spec, self.digit_size, self.parallelism,
                A.size(-1), B.size(-1), outlier_masked=True,
            )

        A_deq, B_deq = A_code * self.A_interval, B_code * self.B_interval
        out_real = (self._matmul(A_deq, B_deq)
                    + self._matmul(A_protected, B_protected)
                    + self._matmul(A_protected, B_deq)
                    + self._matmul(A_deq, B_protected))
        if self.O_spec.enabled:
            channels = self.get_outlier_mask_channel(out_real, self.outlier_ratio)
            out_mask = channels.view(*([1] * (out_real.ndim - 1)), -1)
            out_code = quant_awo(out_real.masked_fill(out_mask, 0),
                                 self.O_interval, self.O_spec, out_dtype=torch.float32)
            out_real = torch.where(out_mask, out_real, out_code * self.O_interval)
        return out_real.to(A.dtype)

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

    def _mixed_precision_specs(self):
        """Return the three activation formats used by the synthetic MP policy.

        ``e5m10`` is real FP16 pass-through.  It must not be routed through
        ``fp8_dtype``/``fp8_max`` in ``quant_spec.py``.
        ``e2m1`` uses per-token max-abs scale and a software FP4 grid.
        """
        return (
            QuantSpec(kind="fp", fmt="e5m10", enabled=False),
            QuantSpec(kind="fp", fmt="e4m3", enabled=True),
            QuantSpec(kind="fp", fmt="e2m1", enabled=True),
        )


    def _quant_forward_mixed_precision(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:
        """Per-token FP16/FP8/FP4 activation quantization.

        This implements a synthetic token-dependent-precision workload. The
        paper motivates variable token precision but does not prescribe these
        formats, ratios, or the PAR classifier.
        """
        original_shape = A.shape
        A_2d = A.reshape(-1, original_shape[-1]).to(torch.float32)
        n_tokens = A_2d.shape[0]

        high_idx, mid_idx, low_idx = self._classify_token_importance(A_2d)
        high_spec, mid_spec, low_spec = self._mixed_precision_specs()

        A_deq_all = torch.empty_like(A_2d)
        code_all = torch.empty_like(A_2d)
        mp_codes = []  # [(local_code, global_indices, spec), ...]

        for indices, spec in (
            (high_idx, high_spec),
            (mid_idx, mid_spec),
            (low_idx, low_spec),
        ):
            if indices.numel() == 0:
                continue
            code, _, dequantized = self._quantize_activation_per_token(
                A_2d.index_select(0, indices), spec
            )
            A_deq_all.index_copy_(0, indices, dequantized)
            code_all.index_copy_(0, indices, code)
            mp_codes.append((code, indices, spec))

        # Keep 2D code_all for statistics, reshape A_deq_all for matmul
        code_all_2d = code_all  # [n_tokens, in_features] for stats
        A_deq_all = A_deq_all.reshape(original_shape)

        B_code = quant_awo(
            B,
            self.B_interval,
            self.B_spec,
            out_dtype=torch.float32,
            chunk_size=1_048_576,
        ).to(torch.float32)

        out_real = self._matmul(A_deq_all, B_code) * self.B_interval
        in_features = A.size(-1)
        out_features = B.size(-1)
        # attention matmul: A 原始形状为 (bsz, num_heads, q_len, head_dim),
        # A_2d 行数 = num_heads × q_len → Mapping_stat_dynamic 需要 num_heads
        # (original_shape 是 torch.Size 元组, 用 len() 判断维数)
        n_heads = int(original_shape[-3]) if len(original_shape) == 4 else 1
        if stat_collector is not None:
            stat_collector.collect_quant_activation_mixed_precision(
                self.layer_name,
                self.layer_idx,
                code_all_2d,
                mp_codes,
                B_code,
                self.B_spec,
                self.digit_size,
                self.parallelism,
                in_features,
                out_features,
                n_heads=n_heads,
                mp_high_ratio=self.mp_high_ratio,
                mp_low_ratio=self.mp_low_ratio,
            )

        # ---- output quantization (A as activation, B as weight) ----
        if self.O_spec.kind in {"int", "fp", "bf"} and getattr(
            self.O_spec, "enabled", True
        ):
            out_code = quant_awo(
                out_real,
                self.O_interval,
                self.O_spec,
                out_dtype=torch.float32,
                chunk_size=1_048_576,
            ).to(torch.float32)
            out = (out_code * self.O_interval).to(A.dtype)
        else:
            out = out_real.to(A.dtype)
        return out.reshape(*original_shape[:-1], out_features)

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

    def bitnet_forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        stat_collector: Optional[object] = None,
    ) -> torch.Tensor:

        # 对于 matmul (qk/pv), A 和 B 是 4D tensor:
        #   qk_matmul: A=(B,H,S,D), B=(B,H,S,D) → in_features=D, out_features=S
        #   pv_matmul: A=(B,H,S,S), B=(B,H,S,D) → in_features=S, out_features=D
        # 对于 linear, B 是 2D weight: (out_features, in_features)
        if B.dim() >= 3:
            # matmul: in_features = A 的最后一维, out_features = B 的最后一维
            in_features = A.shape[-1]
            out_features = B.shape[-1]
        else:
            # linear: B 是 (out_features, in_features) 的 weight matrix
            in_features = B.size(1)
            out_features = B.size(0)

        A_sim = A
        B_sim = B

        if stat_collector is not None:
            stat_collector.collect_quant_activation(
                f"{self.layer_name}",  #add
                self.layer_idx,
                A_sim,
                A,
                B_sim,
                self.B_spec,
                self.A_spec,                                                    # add
                self.digit_size,
                self.parallelism,
                in_features,
                out_features
            )

        out = self._matmul(A, B)

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


    def _load_scales(self):
        A_scale_file, B_scale_file, O_scale_file = self._scale_file_paths()

        missing = [
            path for path in (A_scale_file, B_scale_file, O_scale_file)
            if not os.path.exists(path)
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing scale files for {self.layer_name}_{self.layer_idx}: "
                f"{missing}"
            )

        with open(A_scale_file, "rb") as f:
            self.A_interval = pickle.load(f)

        with open(B_scale_file, "rb") as f:
            self.B_interval = pickle.load(f)

        with open(O_scale_file, "rb") as f:
            self.O_interval = pickle.load(f)

        if self.A_interval is None or self.B_interval is None or self.O_interval is None:
            raise ValueError(
                f"Invalid loaded scales for {self.layer_name}_{self.layer_idx}: "
                f"A={self.A_interval}, B={self.B_interval}, O={self.O_interval}"
            )

    def save_scales(self):
        if self.A_interval is None or self.B_interval is None or self.O_interval is None:
            raise ValueError(
                f"Scales not computed for {self.layer_name}_{self.layer_idx}. "
                f"Run scale_inspection first."
            )

        os.makedirs(self.scale_root_str, exist_ok=True)

        A_scale_file, B_scale_file, O_scale_file = self._scale_file_paths()

        with open(A_scale_file, "wb") as f:
            pickle.dump(self.A_interval, f)

        with open(B_scale_file, "wb") as f:
            pickle.dump(self.B_interval, f)

        with open(O_scale_file, "wb") as f:
            pickle.dump(self.O_interval, f)

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, "
            f"A_bit={self.A_bit}, "
            f"B_bit={self.B_bit}, "
            f"O_bit={self.O_bit}, "
            f"A_spec={self.A_spec.name()}, "
            f"B_spec={self.B_spec.name()}, "
            f"O_spec={self.O_spec.name()}, "
            f"layer={self.layer_name}_{self.layer_idx}"
        )

    def log_quant_error(
        self,
        value: float,
        layer_name: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ):
        """
        Append quantization MSE log.
        """
        layer_name = layer_name if layer_name is not None else self.layer_name
        layer_idx = layer_idx if layer_idx is not None else self.layer_idx

        if self.scale_root_str is None or self.scale_root_str == "":
            return

        try:
            os.makedirs(self.scale_root_str, exist_ok=True)
            log_file = os.path.join(self.scale_root_str, "quant_error_log.jsonl")

            record = {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "mse": float(value),
                "mode": self.mode,
                "A_format": self.A_spec.name(),
                "B_format": self.B_spec.name(),
                "O_format": self.O_spec.name(),
            }

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        except Exception:
            pass
