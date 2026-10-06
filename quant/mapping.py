from dataclasses import dataclass
from re import M, X
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import math  


def _to_fp8_e4m3_bits(tensor):
    """Native E4M3FN encoding including ties, subnormal carry and raw bits."""
    raw=tensor.float().clamp(-448,448).to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
    exp=((raw.long()>>3)&15).float()
    nonzero=tensor!=0
    hi=torch.where(nonzero,exp,torch.full_like(exp,-float("inf"))).amax(-1)
    lo=torch.where(nonzero,exp,torch.full_like(exp,float("inf"))).amin(-1)
    span=torch.where(nonzero.any(-1),hi-lo,torch.zeros_like(hi))
    return span,raw&7,raw



def _to_int8_bits(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 FP tensor 转换为对称 INT8 格式，返回 exp_range(恒为0)、magnitude 矩阵和完整 8-bit 模式。

    对称 INT8 格式 (1 sign + 7 magnitude):
    - 值域: -127 ~ +127 (0 有唯一表示)
    - 溢出时 clamp 到 ±127
    - 无 exponent 字段, 因此 exp_range_per_row 恒为 0

    量化方式: round(tensor * 127 / abs_max_per_row), abs_max_per_row 为每行非零元素绝对值最大值。
    若一行全零, 则量化结果全零。

    Returns:
        exp_range_per_row: shape [rows], 恒为 0 (INT8 无 exponent)
        sign_mantissa: shape 同 tensor, 每元素低 7 位 = magnitude(7bit) (0MMMMMMM)
        int8_bits: shape 同 tensor, 完整 8-bit INT8 bit pattern (uint8)
    """
    sign = tensor < 0
    abs_x = tensor.abs()
    zero_mask = abs_x == 0

    # ---- 每行非零 abs_max ----
    abs_x_for_max = abs_x.clone()
    abs_x_for_max[zero_mask] = -1.0  # 零值不影响 max
    row_abs_max = abs_x_for_max.max(dim=-1).values
    all_zero_row = row_abs_max < 0.0
    row_abs_max = torch.where(all_zero_row, torch.ones_like(row_abs_max), row_abs_max)

    # ---- 量化: round(x * 127 / row_abs_max) ----
    # 广播: row_abs_max shape [rows] → 扩展到 tensor shape
    scale = row_abs_max.unsqueeze(-1)  # shape [rows, 1]
    # 防止 scale=0 (全零行已设为 1)
    quantized = torch.round(abs_x * 127.0 / scale).clamp(max=127.0)

    # 零值强制为 0
    quantized = torch.where(zero_mask, torch.zeros_like(quantized), quantized)

    # ---- 构造输出 ----
    magnitude_u8 = quantized.to(torch.uint8)       # M: 0~127
    sign_u8 = sign.to(torch.uint8)                 # S: 0 or 1

    # (1) exp_range_per_row: INT8 无 exponent, 恒为 0
    exp_range_per_row = torch.zeros_like(row_abs_max)  # shape [rows]

    # (2) sign_mantissa: 仅保留 magnitude(7bit) → 7-bit uint8
    sign_mantissa = magnitude_u8     # 0b0MMMMMMM

    # (3) int8_bits: 完整 8-bit = sign(1) | magnitude(7)
    int8_bits = ((sign_u8 << 7) | magnitude_u8).to(torch.uint8)

    return exp_range_per_row, sign_mantissa, int8_bits


def _to_bf16_e8m7_bits(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 FP tensor 转换为 BF16 E8M7 格式，返回每行 exp 差值、sign+mantissa 矩阵和完整 16-bit 模式。

    BF16 格式 (1 sign + 8 exponent + 7 mantissa):
    - E=0,   M=0~127 : subnormal,  value = (M/128) × 2^(-126)
    - E=1~254, M=0~127: normal,    value = (1 + M/128) × 2^(E-127)
    - E=255, M=0     : Inf (+/-)
    - E=255, M≠0     : NaN
    - 最大有限值 = (1 + 127/128) × 2^127 ≈ 3.39e38
    - 溢出时映射到 Inf (E=255, M=0)

    Returns:
        exp_range_per_row: shape [rows], 每行中非零元素最大 exp 与最小 exp 之差
        sign_mantissa: shape 同 tensor, 每个元素低 7 位 = {mantissa(7bit)} (0MMMMMMM)
        bf16_bits: shape 同 tensor, 完整 16-bit BF16 bit pattern (uint16)
    """
    sign = tensor < 0
    abs_x = tensor.abs()
    zero_mask = abs_x == 0

    # frexp is not implemented for BFloat16 on CUDA; cast to float32 first
    if abs_x.dtype == torch.bfloat16:
        abs_x_f32 = abs_x.to(torch.float32)
    else:
        abs_x_f32 = abs_x

    # ---- 特殊值: Inf / NaN → 直接映射到 BF16 Inf/NaN ----
    inf_mask = abs_x_f32.isinf()
    nan_mask = abs_x_f32.isnan()
    special_mask = inf_mask | nan_mask

    # 对非特殊值做 frexp; 特殊值用 1.0 占位 (后续会被覆盖)
    abs_x_safe = torch.where(special_mask, torch.ones_like(abs_x_f32), abs_x_f32)

    mant, exp = torch.frexp(abs_x_safe)  # mant ∈ [0.5, 1.0), exp 真实无偏指数
    exp_unbiased = exp - 1               # 使 mant ∈ [1.0, 2.0)
    bias = 127
    exp_field = (exp_unbiased + bias).to(torch.float32)  # BF16 biased exponent

    frac = mant * 2.0 - 1.0             # mantissa fraction ∈ [0.0, 1.0)
    mantissa = torch.round(frac * 128.0)  # 量化到 7-bit mantissa

    # 处理进位: mantissa=128 → carry, exp_field+1, mantissa=0
    carry = mantissa == 128.0
    exp_field = exp_field + carry.to(exp_field.dtype)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)

    # ---- 溢出: exp_field > 254 或输入为 Inf/NaN → E=255 ----
    # Inf: E=255, M=0; NaN: E=255, M=非零 (这里用 M=1 表示 NaN)
    overflow = exp_field > 254.0
    exp_field = exp_field.clamp(max=255.0)
    # 溢出 → Inf (M=0)
    mantissa = torch.where(overflow, torch.zeros_like(mantissa), mantissa)
    # 输入 NaN → BF16 NaN (E=255, M=1)
    exp_field = torch.where(nan_mask, torch.full_like(exp_field, 255.0), exp_field)
    mantissa = torch.where(nan_mask, torch.ones_like(mantissa), mantissa)
    # 输入 Inf → BF16 Inf (E=255, M=0)
    exp_field = torch.where(inf_mask, torch.full_like(exp_field, 255.0), exp_field)
    mantissa = torch.where(inf_mask, torch.zeros_like(mantissa), mantissa)

    # ---- E=255 且非溢出时, mantissa 必须 ≤ 127 (M≠0 为 NaN) ----
    # BF16 中 E=255, M≠0 = NaN; 我们只在溢出时设 E=255, M=0 (Inf)
    # 正常值不会到达 E=255, 所以这里 clamp 是安全兜底
    mantissa = torch.where(
        (exp_field == 255.0) & ~overflow,
        mantissa.clamp(max=127.0),
        mantissa,
    )

    del mant, exp, frac, carry, overflow  # 释放中间变量

    # ---- Subnormal: exp_field < 1 ----
    subnormal = exp_field < 1.0

    # Subnormal mantissa: value = M/128 × 2^(-126), 所以 M = round(abs_x / 2^(-133))
    # (因为 M/128 × 2^(-126) = M × 2^(-133))
    sub_mantissa = torch.round(abs_x_f32 / (2.0 ** (-133))).clamp(min=0.0, max=127.0)

    # 当 subnormal mantissa 进位到 128 时, 升格为 E=1, M=0
    sub_carry = sub_mantissa == 128.0
    subnormal = subnormal & ~sub_carry  # 进位后不再是 subnormal
    sub_mantissa = torch.where(sub_carry, torch.zeros_like(sub_mantissa), sub_mantissa)
    exp_field = torch.where(sub_carry, torch.ones_like(exp_field), exp_field)

    # 应用 subnormal
    exp_field = torch.where(subnormal, torch.zeros_like(exp_field), exp_field)
    mantissa = torch.where(subnormal, sub_mantissa, mantissa)

    mantissa = mantissa.clamp(min=0.0, max=127.0)

    # ---- 构造输出 ----
    # torch.where 不支持 uint16, 先用 int32 做 where 操作再转 uint16
    exp_field_i32 = exp_field.to(torch.int32)      # E: 0~255
    mantissa_i32 = mantissa.to(torch.int32)         # M: 0~127
    sign_i32 = sign.to(torch.int32)                 # S: 0 or 1

    # 零值处理: E=0, M=0, S=0
    exp_field_i32 = torch.where(zero_mask, torch.zeros_like(exp_field_i32), exp_field_i32)
    mantissa_i32 = torch.where(zero_mask, torch.zeros_like(mantissa_i32), mantissa_i32)
    sign_i32 = torch.where(zero_mask, torch.zeros_like(sign_i32), sign_i32)

    exp_field_u16 = exp_field_i32.to(torch.uint16)
    mantissa_u16 = mantissa_i32.to(torch.uint16)
    sign_u16 = sign_i32.to(torch.uint16)

    # (1) exp_range_per_row: 每行非零元素 max_exp - min_exp
    #    零值 exp 不参与 max/min, 用特殊值替代
    exp_for_max = exp_field.clone()
    exp_for_max[zero_mask] = -1.0             # max 时零值不影响
    exp_for_min = exp_field.clone()
    exp_for_min[zero_mask] = float('inf')     # min 时零值不影响

    row_max_exp = exp_for_max.max(dim=-1).values
    row_min_exp = exp_for_min.min(dim=-1).values

    # 如果一行全是零, min 会是 inf, max 会是 -1; 此时 exp_range = 0
    all_zero_row = row_min_exp == float('inf')
    row_min_exp = torch.where(all_zero_row, torch.zeros_like(row_min_exp), row_min_exp)
    row_max_exp = torch.where(all_zero_row, torch.zeros_like(row_max_exp), row_max_exp)

    exp_range_per_row = (row_max_exp - row_min_exp)  # shape [rows]

    # (2) mantissa only: 每元素仅保留 mantissa(7bit) → 7-bit uint8
    sign_mantissa = mantissa_i32.to(torch.uint8)     # 0b0MMMMMMM

    # (3) bf16_bits: 完整 16-bit BF16 bit pattern = sign(1) | exp(8) | mantissa(7)
    bf16_bits = ((sign_i32 << 15) | (exp_field_i32 << 7) | mantissa_i32).to(torch.uint16)

    return exp_range_per_row, sign_mantissa, bf16_bits


@dataclass
class CIM_sys:
    """
    Macro define

    kind:
        h:
        w:
        Nmacro:
        Nadder:
    """
    h: int
    w: int
    Nmacro: int
    Nadder: int
    freq: int

def _effective_h(CIM: CIM_sys, in_features: int) -> int:
    """计算每个 bank 实际容纳的维度数。

    当 in_features < Nadder * h 时, 每个 bank 只需容纳
    in_features / Nadder 个维度, 而不是完整的 h。
    例如: in_features=128, Nadder=16 → h_eff=8 (而非 64)。
    """
    h_eff = max(1, min(CIM.h, math.ceil(in_features / CIM.Nadder)))
    return h_eff


def padding_x(
        x: torch.Tensor,
        in_features: int,
        out_features: int,
        CIM: CIM_sys,
        K: int,
        M: int,
        N: int,
        Kr: int,
        Mr: int,
        Nr: int,
    ) -> torch.Tensor:

    h = CIM.h
    w = CIM.w
    Nadder = CIM.Nadder
    Nmacro = CIM.Nmacro

    # h_eff: 每个 bank 实际容纳的维度数 (当 in_features < Nadder*h 时更小)
    h_eff = _effective_h(CIM, in_features)

    Nbatch = x.shape[0]
    Nhead = x.shape[1]
    Nseq = x.shape[2]
    Ndim = x.shape[3]

    pad_dim = (K * Kr * Nadder * h_eff - x.shape[3] % (K * Kr * Nadder * h_eff)) % (K * Kr * Nadder * h_eff)

    x_pad = x

    if pad_dim > 0:
        # F.pad 从最后一维往前:
        #   (dim3_l, dim3_r, dim2_l, dim2_r, dim1_l, dim1_r, dim0_l, dim0_r)
        x_pad = F.pad(x_pad, (0, pad_dim,       # Ndim  dim=3: 不 pad
                                0, 0,  # Nseq  dim=2: 右侧补 pad_S
                               0, 0,      # Nhead dim=1: 不 pad
                               0, 0))     # Nbatch dim=0: 不 pad

    pad_seq = (M * Mr - x.shape[2] % (M * Mr)) % (M * Mr)

    if pad_seq > 0:
        # F.pad 从最后一维往前:
        #   (dim3_l, dim3_r, dim2_l, dim2_r, dim1_l, dim1_r, dim0_l, dim0_r)
        x_pad = F.pad(x_pad, (0, 0,       # Ndim  dim=3: 不 pad
                                0, pad_seq,  # Nseq  dim=2: 右侧补 pad_S
                               0, 0,      # Nhead dim=1: 不 pad
                               0, 0))     # Nbatch dim=0: 不 pad

    # reshape: 使用 h_eff 而非硬编码 h 或 in_features==128 hack
    x_div = x_pad.reshape(Nbatch, Nhead, Mr, M, Kr, K, Nadder, h_eff).permute(0, 1, 2, 4, 3, 5, 6, 7)

    return x_div

def Mapping_stat(
        CIM: CIM_sys,
        x: torch.Tensor,
        in_features: int,
        out_features: int,
        is_prefill: bool = True,
        method: str = "as_fp8"
    ):

    if method == "as_fp8":
        return as_latency(CIM, x, in_features, out_features, is_prefill)
    elif method == "sy_fp8":
        return sy_latency(CIM, x, in_features, out_features, is_prefill)
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'as_fp8' and 'sy_fp8'.")


def M_main_mapping_factor(
        CIM: CIM_sys,
        x: torch.Tensor,
        in_features: int,
        out_features: int,
        is_prefill: bool = True
    ):

    K_factor = 1
    N_factor = 1
    M_factor = CIM.Nmacro


    Nseq = x.shape[2]
    if is_prefill:
        # 使用 h_eff 而非 CIM.h, 使得 in_features=128 时 K_round 正确
        h_eff = _effective_h(CIM, in_features)
        K_round = math.ceil(in_features / (K_factor * CIM.Nadder * h_eff))
        M_round = math.ceil(Nseq / (M_factor * 1))
        N_round = math.ceil(out_features / (N_factor * CIM.w))

        return K_factor, M_factor, N_factor, K_round, M_round, N_round
    else:
        # 使用 h_eff 而非 CIM.h, 使得 in_features=128 时 K_round 正确
        h_eff = _effective_h(CIM, in_features)
        K_round = math.ceil(in_features / (K_factor * CIM.Nadder * h_eff))
        M_round = math.ceil(Nseq / (M_factor * 1))
        N_round = math.ceil(out_features / (N_factor * CIM.w))

        return K_factor, M_factor, N_factor, K_round, M_round, N_round



def compute_optimal_macro_layout_prefill(
    Nmacro: int,
    in_features: int,
    out_features: int,
    seq_length: int,
    h: int = 64,
    w: int = 48,
    Nadder: int = 16,
):
    if Nmacro <= 0:
        raise ValueError(
            f"Nmacro must be positive, got {Nmacro}"
        )

    if in_features <= 0:
        raise ValueError(
            f"in_features must be positive, got {in_features}"
        )

    if out_features <= 0:
        raise ValueError(
            f"out_features must be positive, got {out_features}"
        )

    if seq_length <= 0:
        raise ValueError(
            f"seq_length must be positive, got {seq_length}"
        )

    if h <= 0 or w <= 0 or Nadder <= 0:
        raise ValueError(
            f"h, w and Nadder must be positive, "
            f"got h={h}, w={w}, Nadder={Nadder}"
        )

    # ------------------------------------------------------------
    # 一个 Macro 在一个 K round 中可以覆盖的输入维度
    #
    # 16 banks × 64 dimensions = 1024 dimensions
    # ------------------------------------------------------------
    k_capacity_per_macro_round = Nadder * h

    # 完整 GEMM 在三个方向上至少需要多少个基础 tile
    total_K_tiles = math.ceil(
        in_features / k_capacity_per_macro_round
    )

    total_M_tiles = seq_length

    total_N_tiles = math.ceil(
        out_features / w
    )

    max_K_factor = min(
        Nmacro,
        total_K_tiles,
    )

    max_M_factor = min(
        Nmacro,
        total_M_tiles,
    )

    # N_factor 固定为 1
    N_factor = 1

    best_layout = None
    best_score = None

    for K_factor in range(1, max_K_factor + 1):
        for M_factor in range(1, max_M_factor + 1):

                macros_used = (
                    K_factor
                    * M_factor
                    * N_factor
                )

                if macros_used > Nmacro:
                    continue

                # ====================================================
                # 三个方向仍需顺序执行的轮数
                # ====================================================

                # K_factor 个 Macro group 同时处理不同 K slice
                K_rounds = math.ceil(
                    in_features
                    / (
                        K_factor
                        * k_capacity_per_macro_round
                    )
                )

                # M_factor 个 Macro 处理不同 token
                M_rounds = math.ceil(
                    seq_length / M_factor
                )

                # N_factor=1 固定, 输出通道不并行
                N_rounds = math.ceil(
                    out_features / w
                )

                # 粗粒度总串行轮数
                total_serial_rounds = (
                    K_rounds
                    * M_rounds
                    * N_rounds
                )

                # ====================================================
                # 计算三个维度上的 padding/utilization
                # ====================================================

                K_capacity = (
                    K_factor
                    * K_rounds
                    * k_capacity_per_macro_round
                )

                M_capacity = (
                    M_factor
                    * M_rounds
                )

                N_capacity = (
                    N_factor
                    * N_rounds
                    * w
                )

                K_utilization = (
                    in_features / K_capacity
                )

                M_utilization = (
                    seq_length / M_capacity
                )

                N_utilization = (
                    out_features / N_capacity
                )

                macro_utilization = (
                    macros_used / Nmacro
                )

                overall_utilization = (
                    K_utilization
                    * M_utilization
                    * N_utilization
                    * macro_utilization
                )

                # K 并行时会产生 K_factor 份 partial sum。
                # 这里先用一个无量纲的简单 penalty 做次级比较。
                # 真正计算 latency 时仍应使用精确 reduction cycles。
                K_reduction_penalty = (
                    0
                    if K_factor == 1
                    else (
                        (K_factor - 1)
                        * M_rounds
                        * N_rounds
                    )
                )

                # ====================================================
                # 评分
                #
                # Python tuple 按顺序比较：
                #   1. 总串行轮数越小越好
                #   2. K reduction 越少越好
                #   3. 综合利用率越高越好
                #   4. 同条件优先 K > M > N
                # ====================================================
                score = (
                    total_serial_rounds,
                    # K_reduction_penalty,
                    # -overall_utilization,
                    -K_factor,
                    -M_factor,
                    -N_factor,
                )

                if best_score is None or score < best_score:
                    best_score = score

                    best_layout = {
                        "K_factor": K_factor,
                        "M_factor": M_factor,
                        "N_factor": N_factor,
                        "K_rounds": K_rounds,
                        "M_rounds": M_rounds,
                        "N_rounds": N_rounds,
                        "macros_used": macros_used,
                        "num_groups": (
                            K_factor * N_factor
                        ),
                        "macros_per_group": M_factor,
                        "K_utilization": K_utilization,
                        "M_utilization": M_utilization,
                        "N_utilization": N_utilization,
                        "macro_utilization": macro_utilization,
                        "overall_utilization": overall_utilization,
                        "total_serial_rounds": total_serial_rounds,
                        "K_reduction_penalty": K_reduction_penalty,
                    }

    if best_layout is None:
        raise RuntimeError(
            "Unable to find a valid macro layout"
        )

    return (
        best_layout["K_factor"],
        best_layout["M_factor"],
        best_layout["N_factor"],
        best_layout["K_rounds"],
        best_layout["M_rounds"],
        best_layout["N_rounds"],
    )

def compute_optimal_macro_layout_decode(
    Nmacro: int,
    in_features: int,
    out_features: int,
    seq_length: int,
    h: int = 64,
    w: int = 48,
    Nadder: int = 16,
):
    if Nmacro <= 0:
        raise ValueError(
            f"Nmacro must be positive, got {Nmacro}"
        )

    if in_features <= 0:
        raise ValueError(
            f"in_features must be positive, got {in_features}"
        )

    if out_features <= 0:
        raise ValueError(
            f"out_features must be positive, got {out_features}"
        )

    if seq_length <= 0:
        raise ValueError(
            f"seq_length must be positive, got {seq_length}"
        )

    if h <= 0 or w <= 0 or Nadder <= 0:
        raise ValueError(
            f"h, w and Nadder must be positive, "
            f"got h={h}, w={w}, Nadder={Nadder}"
        )

    # ------------------------------------------------------------
    # 一个 Macro 在一个 K round 中可以覆盖的输入维度
    #
    # 16 banks × 64 dimensions = 1024 dimensions
    # ------------------------------------------------------------
    k_capacity_per_macro_round = Nadder * h

    # 完整 GEMM 在三个方向上至少需要多少个基础 tile
    total_K_tiles = math.ceil(
        in_features / k_capacity_per_macro_round
    )

    total_M_tiles = seq_length

    total_N_tiles = math.ceil(
        out_features / w
    )

    max_K_factor = min(
        Nmacro,
        total_K_tiles,
    )

    # max_M_factor = min(
    #     Nmacro,
    #     total_M_tiles,
    # )

    max_N_factor = min(
        Nmacro,
        total_N_tiles,
    )

    # # N_factor 固定为 1
    # N_factor = 1

    M_factor = 1

    best_layout = None
    best_score = None

    for K_factor in range(1, max_K_factor + 1):
        for N_factor in range(1, max_N_factor + 1):

                macros_used = (
                    K_factor
                    * M_factor
                    * N_factor
                )

                if macros_used > Nmacro:
                    continue

                # ====================================================
                # 三个方向仍需顺序执行的轮数
                # ====================================================

                # K_factor 个 Macro group 同时处理不同 K slice
                K_rounds = math.ceil(
                    in_features
                    / (
                        K_factor
                        * k_capacity_per_macro_round
                    )
                )

                # M_factor 个 Macro 处理不同 token
                M_rounds = math.ceil(
                    seq_length / M_factor
                )

                # N_factor=1 固定, 输出通道不并行
                N_rounds = math.ceil(
                    out_features / (w * N_factor)
                )

                # 粗粒度总串行轮数
                total_serial_rounds = (
                    K_rounds
                    * M_rounds
                    * N_rounds
                )

                # ====================================================
                # 计算三个维度上的 padding/utilization
                # ====================================================

                K_capacity = (
                    K_factor
                    * K_rounds
                    * k_capacity_per_macro_round
                )

                M_capacity = (
                    M_factor
                    * M_rounds
                )

                N_capacity = (
                    N_factor
                    * N_rounds
                    * w
                )

                K_utilization = (
                    in_features / K_capacity
                )

                M_utilization = (
                    seq_length / M_capacity
                )

                N_utilization = (
                    out_features / N_capacity
                )

                macro_utilization = (
                    macros_used / Nmacro
                )

                overall_utilization = (
                    K_utilization
                    * M_utilization
                    * N_utilization
                    * macro_utilization
                )

                # K 并行时会产生 K_factor 份 partial sum。
                # 这里先用一个无量纲的简单 penalty 做次级比较。
                # 真正计算 latency 时仍应使用精确 reduction cycles。
                K_reduction_penalty = (
                    0
                    if K_factor == 1
                    else (
                        (K_factor - 1)
                        * M_rounds
                        * N_rounds
                    )
                )

                # ====================================================
                # 评分
                #
                # Python tuple 按顺序比较：
                #   1. 总串行轮数越小越好
                #   2. K reduction 越少越好
                #   3. 综合利用率越高越好
                #   4. 同条件优先 K > M > N
                # ====================================================
                score = (
                    # total_serial_rounds,
                    # K_reduction_penalty,
                    -overall_utilization,
                    -K_factor,
                    -M_factor,
                    -N_factor,
                )

                if best_score is None or score < best_score:
                    best_score = score

                    best_layout = {
                        "K_factor": K_factor,
                        "M_factor": M_factor,
                        "N_factor": N_factor,
                        "K_rounds": K_rounds,
                        "M_rounds": M_rounds,
                        "N_rounds": N_rounds,
                        "macros_used": macros_used,
                        "num_groups": (
                            K_factor * N_factor
                        ),
                        "macros_per_group": M_factor,
                        "K_utilization": K_utilization,
                        "M_utilization": M_utilization,
                        "N_utilization": N_utilization,
                        "macro_utilization": macro_utilization,
                        "overall_utilization": overall_utilization,
                        "total_serial_rounds": total_serial_rounds,
                        "K_reduction_penalty": K_reduction_penalty,
                    }

    if best_layout is None:
        raise RuntimeError(
            "Unable to find a valid macro layout"
        )

    return (
        best_layout["K_factor"],
        best_layout["M_factor"],
        best_layout["N_factor"],
        best_layout["K_rounds"],
        best_layout["M_rounds"],
        best_layout["N_rounds"],
    )


def Mapping_stat_dynamic(CIM, sm_codes, sm_widths, exp_codes, in_features,
                         out_features, attn=True, n_heads=28,
                         mp_high_ratio=0.1, mp_low_ratio=0.65,
                         is_prefill=True, as_l=1):
    """Map actual per-token mantissa widths; nominal ratios do not set baseline.

    Exponent costs are excluded. Independent batch/head operands serialize.
    Both dense and sparse execute the same padded bank/macro aggregation.
    """
    from .cim_stats import aggregate_bank_steps
    if as_l not in (0,1): raise ValueError("as_l must be 0 or 1")
    if sm_codes.ndim==2:
        if attn:
            if n_heads<=0 or sm_codes.shape[0]%n_heads:
                raise ValueError("attention rows must divide into n_heads")
            codes=sm_codes.reshape(1,n_heads,-1,in_features)
            widths=sm_widths.reshape(1,n_heads,-1)
        else:
            codes=sm_codes[None,None];widths=sm_widths.reshape(1,1,-1)
    elif sm_codes.ndim==3:
        codes=sm_codes[:,None];widths=sm_widths[:,None]
    elif sm_codes.ndim==4:
        codes=sm_codes;widths=sm_widths
    else: raise ValueError("mixed mapping expects 2D/3D/4D codes")
    if codes.shape[-1]!=in_features or tuple(widths.shape)!=tuple(codes.shape[:-1]):
        raise ValueError("mixed code/width shapes are inconsistent")
    if (widths<0).any() or (widths>10).any() or (widths!=widths.round()).any():
        raise ValueError("mantissa widths must be integers in [0,10]")
    if (codes<0).any() or (codes >= (2**widths[...,None])).any():
        raise ValueError("mantissa code exceeds its declared width")
    fn=compute_optimal_macro_layout_prefill if is_prefill else compute_optimal_macro_layout_decode
    layout=fn(CIM.Nmacro,in_features,out_features,codes.shape[-2],
              h=CIM.h,w=CIM.w,Nadder=CIM.Nadder)
    kf,mf,nf,kr,mr,nr=layout
    counts=torch.zeros_like(codes,dtype=torch.float32)
    for bit in range(10): counts+=((codes.long()>>bit)&1)*(widths[...,None]>bit)
    sparse_bank=padding_x(counts,in_features,out_features,CIM,*layout).sum(-1)
    dense_bank=padding_x(widths[...,None].expand_as(codes).float(),
                         in_features,out_features,CIM,*layout).sum(-1)
    sparse=aggregate_bank_steps(sparse_bank,bool(as_l))
    dense=aggregate_bank_steps(dense_bank,bool(as_l))
    work=sparse_bank.sum()/max(1,mf*kf*CIM.Nadder)
    return work,sparse,dense,sparse*nr,dense*nr,work*nr




def as_latency(CIM, x, in_features, out_features, is_prefill=True):
    """Compatibility tuple using actual FP8 mantissa codes in both phases."""
    from .cim_stats import measure_mapping
    result=measure_mapping(CIM,x,"e4m3",in_features,out_features,is_prefill)
    nr=result["layout"][-1]
    sparse=result["sparse_steps"];dense=result["dense_steps"]
    return sparse/nr,sparse/nr,dense/nr,sparse,dense,sparse

       


def sy_latency(CIM, x, in_features, out_features, is_prefill=True):
    """Synchronous counterpart using the SAME FP8 bit scope and mapping."""
    from .cim_stats import measure_mapping
    result=measure_mapping(CIM,x,"e4m3",in_features,out_features,is_prefill,asynchronous=False)
    nr=result["layout"][-1];sparse=result["sparse_steps"];dense=result["dense_steps"]
    return sparse/nr,sparse/nr,dense/nr,sparse,dense,sparse
