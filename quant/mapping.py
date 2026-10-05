from dataclasses import dataclass
from re import M, X
from typing import Any, Optional, Tuple

from networkx import k_crust
import torch
import torch.nn.functional as F
import math  


def _to_fp8_e4m3_bits(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 FP tensor 转换为 E4M3 FP8 格式，返回每行 exp 差值、sign+mantissa 矩阵和完整 8-bit 模式。

    FP8 E4M3FN 格式 (1 sign + 4 exponent + 3 mantissa):
    - E=0,   D=0~7  : subnormal,  value = (D/8) × 2^(-6)
    - E=1~14, D=0~7 : normal,     value = (1 + D/8) × 2^(E-7)
    - E=15,  D=0~6  : normal,     value = (1 + D/8) × 2^8
    - E=15,  D=7    : NaN
    - 最大有限值 = (1 + 6/8) × 2^8 = 448
    - 溢出时映射到 NaN (E=15, M=7, 与 PyTorch 行为一致)

    Returns:
        exp_range_per_row: shape [rows], 每行中非零元素最大 exp 与最小 exp 之差
        sign_mantissa: shape 同 tensor, 每个元素低 3 位 = {mantissa(3bit)} (0MMM)
        fp8_bits: shape 同 tensor, 完整 8-bit FP8 bit pattern (uint8)
    """
    sign = tensor < 0
    abs_x = tensor.abs()
    zero_mask = abs_x == 0

    # frexp is not implemented for BFloat16 on CUDA; cast to float32 first
    if abs_x.dtype == torch.bfloat16:
        abs_x_f32 = abs_x.to(torch.float32)
    else:
        abs_x_f32 = abs_x

    mant, exp = torch.frexp(abs_x_f32)  # mant ∈ [0.5, 1.0), exp 真实无偏指数
    exp_unbiased = exp - 1               # 使 mant ∈ [1.0, 2.0)
    bias = 7
    exp_field = (exp_unbiased + bias).to(torch.float32)  # FP8 biased exponent

    frac = mant * 2.0 - 1.0             # mantissa fraction ∈ [0.0, 1.0)
    mantissa = torch.round(frac * 8.0)  # 量化到 3-bit mantissa

    # 处理进位: mantissa=8 → carry, exp_field+1, mantissa=0
    carry = mantissa == 8.0
    exp_field = exp_field + carry.to(exp_field.dtype)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)

    # ---- 溢出: exp_field > 15 → E=15, M=7 (NaN, 与 PyTorch 行为一致) ----
    overflow = exp_field > 15.0
    exp_field = exp_field.clamp(max=15.0)
    mantissa = torch.where(overflow, torch.full_like(mantissa, 7.0), mantissa)

    # ---- E=15 且非溢出时, mantissa 必须 ≤ 6 (D=7 为 NaN, 仅溢出时使用) ----
    mantissa = torch.where(
        (exp_field == 15.0) & ~overflow,
        mantissa.clamp(max=6.0),
        mantissa,
    )

    del mant, exp, frac, carry, overflow  # 释放中间变量

    # ---- Subnormal: exp_field < 1 ----
    subnormal = exp_field < 1.0

    # Subnormal mantissa: value = D/8 × 2^(-6), 所以 D = round(abs_x / 2^(-9))
    # (因为 D/8 × 2^(-6) = D × 2^(-9))
    sub_mantissa = torch.round(abs_x_f32 / (2.0 ** (-9))).clamp(min=0.0, max=7.0)

    # 当 subnormal mantissa 进位到 8 时, 升格为 E=1, M=0
    sub_carry = sub_mantissa == 8.0
    subnormal = subnormal & ~sub_carry  # 进位后不再是 subnormal
    sub_mantissa = torch.where(sub_carry, torch.zeros_like(sub_mantissa), sub_mantissa)
    exp_field = torch.where(sub_carry, torch.ones_like(exp_field), exp_field)

    # 应用 subnormal
    exp_field = torch.where(subnormal, torch.zeros_like(exp_field), exp_field)
    mantissa = torch.where(subnormal, sub_mantissa, mantissa)

    mantissa = mantissa.clamp(min=0.0, max=7.0)

    # ---- 构造输出 ----
    exp_field_u8 = exp_field.to(torch.uint8)      # E: 0~15
    mantissa_u8 = mantissa.to(torch.uint8)         # M: 0~7
    sign_u8 = sign.to(torch.uint8)                 # S: 0 or 1

    # 零值处理: E=0, M=0, S=0
    exp_field_u8 = torch.where(zero_mask, torch.zeros_like(exp_field_u8), exp_field_u8)
    mantissa_u8 = torch.where(zero_mask, torch.zeros_like(mantissa_u8), mantissa_u8)
    sign_u8 = torch.where(zero_mask, torch.zeros_like(sign_u8), sign_u8)

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

    # (2) mantissa only: 每元素仅保留 mantissa(3bit) → 3-bit uint8
    sign_mantissa = mantissa_u8     # 0b0MMM

    # (3) aligned_fp8_bits: 将 sign_mantissa 左移 exp 后得到的值
    #     由于 exp 取值 0~15，最大位宽理论上为 4 + 15 = 19
    #aligned_fp8_bits = (sign_mantissa.long() << exp_field_u8.long())

    return exp_range_per_row, sign_mantissa, 0


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
    x_div = x_pad.reshape(Nbatch, Nhead, Mr, Kr, M, K, Nadder, h_eff)

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


def Mapping_stat_dynamic(
        CIM: CIM_sys,
        sm_codes: torch.Tensor,
        sm_widths: torch.Tensor,
        exp_codes: torch.Tensor,
        in_features: int,
        out_features: int,
        attn: bool = True,
        n_heads: int = 28,
        mp_high_ratio: float = 0.1,
        mp_low_ratio: float = 0.65,
        ):
    """Map token-dependent FP precision onto the EffLoc prefill datapath.

    n_heads: attn=True 时用于 (num_heads*S, D) -> (1, num_heads, S, D) 的
    reshape。不同模型 head 数不同 (Qwen2.5-7B=28, 14B=40, OPT=32)，
    由调用方按实际张量形状传入; 默认 28 保持向后兼容。

    The returned latency values are effective-bit steps. Physical time is
    obtained by multiplying these steps by ``cycles_per_effective_bit``
    (three clocks in the target hardware) exactly once in
    ``_cycles_to_seconds``.

    sm_codes use the 0MMM convention: mantissa bits only, no sign bit.
    sm_widths is the number of mantissa bits per element (3 for FP8,
    7 for BF16, 10 for FP16, 1 for FP4).
    """
    if True:    

        if sm_codes.dim() == 3:
            # (B, S, D) -> (B, 1, S, D)
            sm_codes_pad = sm_codes.unsqueeze(0)
            sm_widths_pad = sm_widths.unsqueeze(0)
            exp_codes_pad = exp_codes.unsqueeze(0)
        elif sm_codes.dim() == 2:
            if attn:
                # (num_heads*S, D) -> (1, n_heads, S, D)
                if n_heads <= 0 or sm_codes.shape[0] % n_heads != 0:
                    raise ValueError(
                        f"attn reshape 失败: rows={sm_codes.shape[0]} 不能被 "
                        f"n_heads={n_heads} 整除。请传入正确的 head 数。"
                    )
                sm_codes_pad = sm_codes.reshape(1, n_heads, sm_codes.shape[0]//n_heads, sm_codes.shape[1])
                sm_widths_pad = sm_widths.reshape(1, n_heads, sm_codes.shape[0] // n_heads)
                exp_codes_pad = exp_codes.reshape(1, n_heads, sm_codes.shape[0]//n_heads, exp_codes.shape[1])
            else:
                sm_codes_pad = sm_codes.unsqueeze(0).unsqueeze(0)
                sm_widths_pad = sm_widths.unsqueeze(0).unsqueeze(0)
                exp_codes_pad = exp_codes.unsqueeze(0).unsqueeze(0)
        elif sm_codes.dim() == 4:
            sm_codes_pad = sm_codes
            sm_widths_pad = sm_widths
            exp_codes_pad = exp_codes
        elif sm_codes.dim() != 4:
            raise ValueError(
                f"Mapping_stat 只支持 2D/3D/4D 输入, 收到 {sm_codes.dim()}D"
            )
        if True:#as_l:
            # K, M, N, Kr, Mr, Nr = M_main_mapping_factor(CIM, sm_codes_pad, in_features, out_features,True)
            K, M, N, Kr, Mr, Nr = compute_optimal_macro_layout_prefill(CIM.Nmacro, in_features, out_features, sm_codes_pad.shape[2])
        else:
            K, M, N, Kr, Mr, Nr = compute_optimal_macro_layout_prefill(CIM.Nmacro, in_features, out_features, sm_codes_pad.shape[2])

        # ============================================================
        # sm_codes / exp_codes: per-token per-dimension, 用 padding_x 处理
        # sm_widths: per-token only, 需要单独处理
        # ============================================================
        sm_div = padding_x(sm_codes_pad, in_features, out_features, CIM, K, M, N, Kr, Mr, Nr)
        exp_div = padding_x(exp_codes_pad, in_features, out_features, CIM, K, M, N, Kr, Mr, Nr)

        # sm_widths 是 per-token 的 (1D), 不能用 padding_x (需要 4D)
        # 单独处理: pad token 维度, reshape 到 (Nbatch, Nhead, Mr, M)
        n_tokens = sm_codes_pad.shape[2]
        pad_tokens_w = (M * Mr - n_tokens % (M * Mr)) % (M * Mr)
        sm_widths_4d = sm_widths_pad  # 已经是 (Nbatch, Nhead, n_tokens) 或类似
        # 确保 sm_widths_pad 是 3D: (Nbatch, Nhead, n_tokens)
        if sm_widths_pad.dim() == 1:
            sm_widths_4d = sm_widths_pad.unsqueeze(0).unsqueeze(0)  # (1, 1, n_tokens)
        elif sm_widths_pad.dim() == 2:
            sm_widths_4d = sm_widths_pad.unsqueeze(0)  # (1, Nhead_or_batch, n_tokens)
        # Pad token (dim=2) to M*Mr
        if pad_tokens_w > 0:
            sm_widths_4d = F.pad(sm_widths_4d.to(torch.int64), (0, pad_tokens_w), value=0)
        # Reshape to (Nbatch, Nhead, Mr, M)
        width_div = sm_widths_4d.reshape(sm_widths_4d.shape[0], sm_widths_4d.shape[1], Mr, M)

        # ============================================================
        # Popcount with width masking
        # sm_div shape: (Nbatch, Nhead, Mr, Kr, M, K, Nadder, h)
        # width_div shape: (Nbatch, Nhead, Mr, M)
        # 0MMM: sm_codes contain mantissa bits only (no sign bit).
        # max_bits covers all formats: FP16=10, BF16=7, FP8=3, FP4=1.
        # ============================================================
        max_bits = 10
        shifts = torch.arange(max_bits, device=sm_codes.device, dtype=torch.int64)
        bit_values = (sm_div.unsqueeze(-1) >> shifts) & 1
        # bit_values shape: (Nbatch, Nhead, Mr, Kr, M, K, Nadder, h, max_bits)
        # width_div 需要广播到 (Nbatch, Nhead, Mr, 1, M, 1, 1, 1, 1)
        bit_valid = (
            shifts.view(1, 1, 1, 1, 1, 1, 1, 1, max_bits)
            < width_div.view(width_div.shape[0], width_div.shape[1], Mr, 1, M, 1, 1, 1, 1)
        )
        popcount_sum = (bit_values * bit_valid).sum(dim=(-1, -2)).float()
        # popcount_sum shape: (Nbatch, Nhead, Mr, Kr, M, K, Nadder)

        nonzero_mask = (exp_div != 0) | (sm_div != 0)
        large = torch.full_like(exp_div, 1 << 30)
        small = torch.full_like(exp_div, -(1 << 30))
        exp_min = torch.where(nonzero_mask, exp_div, large).min(dim=-1).values
        exp_max = torch.where(nonzero_mask, exp_div, small).max(dim=-1).values
        all_zero = ~nonzero_mask.any(dim=-1)
        exp_min = torch.where(all_zero, torch.zeros_like(exp_min), exp_min)
        exp_max = torch.where(all_zero, torch.zeros_like(exp_max), exp_max)
        exp_range = (exp_max - exp_min).float()
        # exp_range shape: (Nbatch, Nhead, Mr, Kr, M, K, Nadder)

        # Match ordinary Mapping_stat: mantissa-only popcount plus 2*exp-range.
        bank_steps = popcount_sum# + 2.0 * exp_range
        # bank_steps shape: (Nbatch, Nhead, Mr, Kr, M, K, Nadder)

        # ============================================================
        # Macro-internal bank barrier, macro-external asynchronous execution.
        # bank barrier: amax over Nadder (dim=-1)
        # ============================================================
        round_latency = bank_steps.amax(dim=-1)
        # round_latency shape: (Nbatch, Nhead, Mr, Kr, M, K)
        if True:#as_l:
            # 每个 macro 顺序处理所有 cycle (Mr) 和 K-round (Kr)
            # squeeze Nhead=1, K=1 for simplicity
            macro_total = round_latency.sum(dim=(0, 1, 2, 3))
            # macro_total shape: (Nbatch, M)

            # 整层延迟 = 最慢 macro
            layer_latency_single_weight_tile = macro_total.max()
        else:
            # round_latency shape: (Nbatch, Nhead, Mr, Kr, M, K)
            # # Step 2: K 同步 → 每 K macro 耗时
            k_max = round_latency.amax(dim=-1)

            # Step 2: M 同步 → 每 M macro 耗时
            m_max = k_max.amax(dim=-1)

            # 单层总延迟 = 最慢 macro 的总执行时间 (直接取全局最大值标量)
            layer_latency_single_weight_tile = m_max.sum()

        # ============================================================
        # Dense baseline: 与 round_latency 同形状, 每个值 = h_eff * bit_width
        # round_latency shape: (Nbatch, Nhead, Mr, Kr, M, K)
        # 每个 bank 满载工作量 = h_eff × width (per-token bit 数)
        # bank 内 Nadder 个 bank 满载相同 → amax(Nadder) = h_eff * width
        # 所以 dense_round_latency 每个位置 = h_eff * width_div 对应位置的 bit 数
        # ============================================================


        # ============================================================
        # Dense baseline 聚合: 与 round_latency → layer_latency_single_weight_tile
        # 完全相同的聚合路径
        #
        # 满载基线 = h(64) × 平均尾数位宽。位宽按混精三档配比加权:
        #   high → FP16 (10 mantissa bits), mid → FP8 (3), low → FP4 (1)
        # 配比由调用方从 QuantizedLinear / QuantizedMatMul 实例直接传入
        # (self.mp_high_ratio / self.mp_low_ratio), 与实际量化行为同源。
        # ============================================================
        mp_mid_ratio = 1.0 - mp_high_ratio - mp_low_ratio
        if mp_high_ratio < 0 or mp_low_ratio < 0 or mp_mid_ratio < 0:
            raise ValueError(
                f"invalid mixed-precision ratios: high={mp_high_ratio}, "
                f"low={mp_low_ratio}, mid={mp_mid_ratio}"
            )
        avg_bits = (
            mp_high_ratio * 10
            + mp_low_ratio * 1
            + mp_mid_ratio * 3
        )

        if round_latency.shape[1] == 1:  # linear (Nhead=1)
           layer_ideal_single_weight_tile = (
               64 * avg_bits * Kr * Mr
           )
        else:  # attention matmul (Nhead>1, 各 head 串行)
           layer_ideal_single_weight_tile = (
               64 * avg_bits * Kr * Mr * round_latency.shape[1]
           )

        # Same return convention as ordinary Mapping_stat.
        weight_cycles = Nr
        idea_sp = bank_steps.sum()/M/K/CIM.Nadder
        boperation = bank_steps  # dense baseline 不再单独计算 boperation
        layer_latency = layer_latency_single_weight_tile * weight_cycles
        layer_ideal_latency = layer_ideal_single_weight_tile * weight_cycles
        ideal_sparsity_op = idea_sp * weight_cycles


        return (
            idea_sp,
            layer_latency_single_weight_tile,
            layer_ideal_single_weight_tile,
            layer_latency,
            layer_ideal_latency,
            ideal_sparsity_op,
        )


def as_latency(
        CIM: CIM_sys,
        x: torch.Tensor,
        in_features: int,
        out_features: int,
        is_prefill: bool = True
        ):
        if is_prefill:

            # ============================================================
            # 统一输入维度: 强制 (Nbatch, Nhead, Nseq, Ndim) 4D
            # ============================================================
            if x.dim() == 3:
            # (B, S, D) -> (B, 1, S, D)
                x_pad = x.unsqueeze(0)
            elif x.dim() == 2:
                # (S, D) -> (1, 1, S, D)
                x_pad = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 4:
                x_pad = x
            elif x.dim() != 4:
                raise ValueError(
                    f"Mapping_stat 只支持 2D/3D/4D 输入, 收到 {x.dim()}D"
                )

            K, M, N, Kr, Mr, Nr =  compute_optimal_macro_layout_prefill(CIM.Nmacro, in_features, out_features, x_pad.shape[2])


            x_div = padding_x(x_pad, in_features, out_features, CIM, K, M, N, Kr, Mr, Nr)

            del x_pad

            exp_range_per_row, sign_mantissa, aligned_fp8_bits = _to_int8_bits(x_div)
            # exp_range_per_row, sign_mantissa, aligned_fp8_bits = _to_fp8_e4m3_bits(x_div)
            # exp_range_per_row, sign_mantissa, aligned_fp8_bits = _to_bf16_e8m7_bits(x_div)
            del aligned_fp8_bits
            del x_div

            # # 每 dim 维度的 active bits (FP8 mantissa only = 3-bit: 0b0MMM)
            # shifts = torch.arange(0, 3, device=sign_mantissa.device)

            # 每 dim 维度的 active bits (BF16 mantissa only = 7-bit: 0b0MMMMMMM)
            shifts = torch.arange(0, 7, device=sign_mantissa.device)

            bits = (sign_mantissa.unsqueeze(-1) >> shifts) & 1
            c = bits.sum(dim=-1).float()


            # 每个 bank 的实际量 = dim方向active_bits之和 + 2×exp_range
            c = c.sum(dim=-1) + exp_range_per_row

            
            fp8_stat_b1 = c + exp_range_per_row
            all_bits_base = bits.shape[-1] * bits.shape[-2]

            del c
            del bits

            # Step 1: bank 同步 → 每 macro 每 cycle 的耗时
            bank_max = fp8_stat_b1.amax(dim=-1)
            # bank_max shape: (Nbatch, Nhead, Mr, Kr, M, K)

            # ============================================================
            # Macro 间异步统计模型 (与 stat_manager.Mapping_stat_bf16 一致)
            #
            # 每个 macro 顺序处理自己分配到的所有 cycle 和 K-round:
            #   macro_total = sum over (Mr, Kr) → 每个 macro 的总执行时间
            #   layer_latency = max over M → 整层由最慢 macro 决定
            # ============================================================

            # Step 2: 每个 macro 累加所有 cycle (Mr) 和 K-round (Kr)
            # bank_max shape: (Nbatch, Nhead, Mr, Kr, M, K)
            # 对 Nhead=1, K=1 的情况, 先 squeeze 再 sum
            macro_total = bank_max.sum(dim=(0, 1, 2, 3))
            # macro_total shape: (Nbatch, M) — 每个 macro 的总执行步数

            # Step 3: 整层延迟 = 最慢 macro
            # macro_total shape: (Nbatch, M) — 每个 macro 的总执行步数
            # 取每个 batch 内最慢 macro，再 sum over batch → 标量
            layer_latency = macro_total.max()
            # layer_latency: 标量 tensor

            # 单层满载总延迟 = 每 cycle 满载开销 × cycle 总数
            layer_ideal_latency = 192 * Kr * Mr
            idea_sp = fp8_stat_b1.sum()/M/K/CIM.Nadder
            # boperation 保持原语义: 理想满载时单元素操作数
            boperation = torch.full_like(fp8_stat_b1, all_bits_base)
            return idea_sp, layer_latency, layer_ideal_latency, layer_latency * Nr, layer_ideal_latency * Nr, idea_sp * Nr
        else:

            # ============================================================
            # 统一输入维度: 强制 (Nbatch, Nhead, Nseq, Ndim) 4D
            # ============================================================
            if x.dim() == 3:
            # (B, S, D) -> (B, 1, S, D)
                x_pad = x.unsqueeze(0)
            elif x.dim() == 2:
                # (S, D) -> (1, 1, S, D)
                x_pad = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 4:
                x_pad = x
            elif x.dim() != 4:
                raise ValueError(
                    f"Mapping_stat 只支持 2D/3D/4D 输入, 收到 {x.dim()}D"
                )

            K, M, N, Kr, Mr, Nr = compute_optimal_macro_layout_decode(CIM.Nmacro, in_features, out_features, x_pad.shape[2])


            x_div = padding_x(x_pad, in_features, out_features, CIM, K, M, N, Kr, Mr, Nr)

            del x_pad

            # exp_range_per_row, sign_mantissa, aligned_fp8_bits = _to_int8_bits(x_div)
            exp_range_per_row, sign_mantissa, aligned_fp8_bits = _to_fp8_e4m3_bits(x_div)
            # exp_range_per_row, sign_mantissa, aligned_fp8_bits = _to_bf16_e8m7_bits(x_div)
            del aligned_fp8_bits
            del x_div

            # 每 dim 维度的 active bits (FP8 mantissa only = 3-bit: 0b0MMM)
            shifts = torch.arange(0, 3, device=sign_mantissa.device)

            # # 每 dim 维度的 active bits (BF16 mantissa only = 7-bit: 0b0MMMMMMM)
            # shifts = torch.arange(0, 7, device=sign_mantissa.device)

            bits = (sign_mantissa.unsqueeze(-1) >> shifts) & 1
            c = bits.sum(dim=-1).float()


            # 每个 bank 的实际量 = dim方向active_bits之和 + 2×exp_range
            c = c.sum(dim=-1) + exp_range_per_row

            
            fp8_stat_b1 = c + exp_range_per_row
            all_bits_base = bits.shape[-1] * bits.shape[-2]

            del c
            del bits

            # Step 1: bank 同步 → 每 macro 每 cycle 的耗时
            bank_max = fp8_stat_b1.amax(dim=-1)
            # bank_max shape: (Nbatch, Nhead, Mr, Kr, M, K)

            # ============================================================
            # Macro 间异步统计模型 (与 stat_manager.Mapping_stat_bf16 一致)
            #
            # 每个 macro 顺序处理自己分配到的所有 cycle 和 K-round:
            #   macro_total = sum over (Mr, Kr) → 每个 macro 的总执行时间
            #   layer_latency = max over M → 整层由最慢 macro 决定
            # ============================================================

            # Step 2: 每个 macro 累加所有 cycle (Mr) 和 K-round (Kr)
            # bank_max shape: (Nbatch, Nhead, Mr, Kr, M, K)
            # 对 Nhead=1, K=1 的情况, 先 squeeze 再 sum
            macro_total = bank_max.sum(dim=(0, 1, 2, 3))
            # macro_total shape: (Nbatch, M) — 每个 macro 的总执行步数

            # Step 3: 整层延迟 = 最慢 macro
            # macro_total shape: (Nbatch, M) — 每个 macro 的总执行步数
            # 取每个 batch 内最慢 macro，再 sum over batch → 标量
            layer_latency = macro_total.max()
            # layer_latency: 标量 tensor

            # 单层满载总延迟 = 每 cycle 满载开销 × cycle 总数
            layer_ideal_latency = 192 * Kr * Mr * bank_max.shape[1]
            idea_sp = fp8_stat_b1.sum()/M/K/CIM.Nadder
            # boperation 保持原语义: 理想满载时单元素操作数
            boperation = torch.full_like(fp8_stat_b1, all_bits_base)
            return idea_sp, layer_latency, layer_ideal_latency, layer_latency * Nr, layer_ideal_latency * Nr, idea_sp * Nr
       


def sy_latency(
        CIM: CIM_sys,
        x: torch.Tensor,
        in_features: int,
        out_features: int,
        is_prefill: bool = True
        ):

        Nmacro = CIM.Nmacro
        Nadder = CIM.Nadder
        h = CIM.h
        w = CIM.w

        if is_prefill:

            # ============================================================
            # 统一输入维度: 强制 (Nbatch, Nhead, Nseq, Ndim) 4D
            # ============================================================
            if x.dim() == 3:
            # (B, S, D) -> (B, 1, S, D)
                x_pad = x.unsqueeze(1)
            elif x.dim() == 2:
                # (S, D) -> (1, 1, S, D)
                x_pad = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 4:
                x_pad = x
            elif x.dim() != 4:
                raise ValueError(
                    f"Mapping_stat 只支持 2D/3D/4D 输入, 收到 {x.dim()}D"
                )

            K, M, N, Kr, Mr, Nr = compute_optimal_macro_layout_decode(CIM.Nmacro, in_features, out_features, x_pad.shape[2])


            x_div = padding_x(x_pad, in_features, out_features, CIM, K, M, N, Kr, Mr, Nr)

            del x_pad

            # exp_range_per_row, sign_mantissa, aligned_fp8_bits = _to_fp8_e4m3_bits(x_div)
            exp_range_per_row, sign_mantissa, aligned_fp8_bits = _to_bf16_e8m7_bits(x_div)
            del aligned_fp8_bits
            del x_div

            # # 每 dim 维度的 active bits
            # shifts = torch.arange(0, 4, device=sign_mantissa.device)

            # 每 dim 维度的 active bits (BF16 mantissa only = 7-bit: 0b0MMMMMMM)
            shifts = torch.arange(0, 7, device=sign_mantissa.device)

            bits = (sign_mantissa.unsqueeze(-1) >> shifts) & 1
            c = bits.sum(dim=-1).float()

            # 每个 bank 的实际量 = dim方向active_bits之和 + 2×exp_range
            c = c.sum(dim=-1) + exp_range_per_row

            
            fp8_stat_b1 = c + exp_range_per_row
            all_bits_base = bits.shape[-1] * bits.shape[-2]

            del c
            del bits

            # ============================================================
            # Macro 间完全异步的同步模型
            #
            # fp8_stat_b1 shape: (Nbatch, Nhead, Mround, kround, M, K, Nadder)
            #                      dim:    0       1      2        3          4          5         6
            #
            # 模型假设:
            #   - 每个 macro 内 Nadder 个 bank 同步 (bank 级屏障)
            #   - macro 之间完全异步, 各自跑完自己的所有 cycle 后才出结果
            #   - 每个 macro 执行完一个 cycle 立即进入下一个, 不等待其它 macro
            #
            # 步骤:
            #   1) bank_max = fp8_stat_b1.amax(Nadder)         → 每 macro 当前 cycle 的耗时
            #   2) macro_total = sum(Ncycle, Nrowcycle)         → 每 macro 的总执行时间(自己跑完所有 cycle)
            #   3) all_bits    = max(Nsamecol, Nsamerow)        → 16 个 macro 中最慢的 = 本次延迟
            # ============================================================

            # Step 1: bank 同步 → 每 macro 每 cycle 的耗时
            bank_max = fp8_stat_b1.amax(dim=-1)

            # Step 2: K 同步 → 每 K macro 耗时
            k_max = bank_max.amax(dim=-1)

            # Step 2: M 同步 → 每 M macro 耗时
            m_max = k_max.amax(dim=-1)

            # 单层总延迟 = 最慢 macro 的总执行时间 (直接取全局最大值标量)
            layer_latency = m_max.sum()

            # 单层满载总延迟 = 每 cycle 满载开销 × cycle 总数
            # 7 mantissa bits × 64 dims per bank = 448
            layer_ideal_latency = 448 * Kr * Mr
            del m_max

            # boperation 保持原语义: 理想满载时单元素操作数
            boperation = torch.full_like(fp8_stat_b1, all_bits_base)
            return fp8_stat_b1, layer_latency, layer_ideal_latency, layer_latency * Nr, layer_ideal_latency * Nr, fp8_stat_b1 * Nr
        else:
            return 0, 0, 0, 0, 0, 0