import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Any

from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from .quant_linear import QuantizedLinear
from .quant_matmul import QuantizedMatMul
from .stat_manager import QuantStatManager
from .opt_wrapper import create_quantized_linear, create_quantized_matmul


QWEN_ATTN_LINEAR_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj"]
QWEN_MLP_LINEAR_NAMES = ["gate_proj", "up_proj", "down_proj"]


def wrap_qwen_linear_only(
    model,
    quant_config: Dict[str, Any],
    mode: str = "scale_inspection",
    stat_manager: Optional[QuantStatManager] = None,
):
    print("Wrapping Qwen model with QuantizedLinear only...")

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError(
            "Expected Qwen-like model with model.model.layers, "
            "but this model does not have that structure."
        )

    layers = model.model.layers
    replaced_count = 0

    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        mlp = layer.mlp

        for name in QWEN_ATTN_LINEAR_NAMES:
            original_layer = getattr(attn, name)

            if not isinstance(original_layer, nn.Linear):
                raise TypeError(
                    f"Layer {layer_idx}.{name} is not nn.Linear: "
                    f"{type(original_layer)}"
                )

            qlayer = create_quantized_linear(
                original_layer=original_layer,
                layer_type=name,
                layer_idx=layer_idx,
                quant_config=quant_config,
                mode=mode,
            )
            qlayer._stat_manager = stat_manager
            setattr(attn, name, qlayer)
            replaced_count += 1

            if stat_manager is not None:
                stat_manager.register_layer(name, layer_idx)

        for name in QWEN_MLP_LINEAR_NAMES:
            original_layer = getattr(mlp, name)

            if not isinstance(original_layer, nn.Linear):
                raise TypeError(
                    f"Layer {layer_idx}.mlp.{name} is not nn.Linear: "
                    f"{type(original_layer)}"
                )

            qlayer = create_quantized_linear(
                original_layer=original_layer,
                layer_type=name,
                layer_idx=layer_idx,
                quant_config=quant_config,
                mode=mode,
            )
            qlayer._stat_manager = stat_manager
            setattr(mlp, name, qlayer)
            replaced_count += 1

            if stat_manager is not None:
                stat_manager.register_layer(name, layer_idx)

    print(f"Qwen Linear-only replaced count: {replaced_count}")
    return model

def inject_qwen_quantized_matmul(
    attention_module: nn.Module,
    layer_idx: int,
    quant_config: Dict[str, Any],
    mode: str = "scale_inspection",
    stat_manager: Optional[QuantStatManager] = None,
):
    """
    Inject qk_matmul and pv_matmul into Qwen2/Qwen2.5 attention.

    Compatible with the Qwen2Attention version whose rotary_emb API is:
        rotary_emb(x, seq_len=...)
    """

    qk_matmul = create_quantized_matmul(
        "qk_matmul",
        layer_idx,
        quant_config,
        mode,
    )
    pv_matmul = create_quantized_matmul(
        "pv_matmul",
        layer_idx,
        quant_config,
        mode,
    )

    qk_matmul._stat_manager = stat_manager
    pv_matmul._stat_manager = stat_manager

    attention_module.qk_matmul = qk_matmul
    attention_module.pv_matmul = pv_matmul
    attention_module.stat_manager = stat_manager

    if stat_manager is not None:
        stat_manager.register_layer("qk_matmul", layer_idx)
        stat_manager.register_layer("pv_matmul", layer_idx)

    original_forward = attention_module.forward
    attention_module._original_forward = original_forward

    def quantized_forward(
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        num_heads = attention_module.num_heads
        num_key_value_heads = attention_module.num_key_value_heads
        head_dim = attention_module.head_dim

        # 1. q/k/v projection
        query_states = attention_module.q_proj(hidden_states)
        key_states = attention_module.k_proj(hidden_states)
        value_states = attention_module.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, num_heads, head_dim
        ).transpose(1, 2)

        key_states = key_states.view(
            bsz, q_len, num_key_value_heads, head_dim
        ).transpose(1, 2)

        value_states = value_states.view(
            bsz, q_len, num_key_value_heads, head_dim
        ).transpose(1, 2)

        # 2. RoPE sequence length
        kv_seq_len = key_states.shape[-2]

        cache_obj = past_key_values if past_key_values is not None else past_key_value

        if cache_obj is not None:
            if hasattr(cache_obj, "get_usable_length"):
                kv_seq_len += cache_obj.get_usable_length(kv_seq_len, attention_module.layer_idx)

        if position_ids is not None:
            rotary_seq_len = max(
                kv_seq_len,
                int(position_ids[:, -1].max().item()) + 1,
            )
        else:
            rotary_seq_len = kv_seq_len
            position_ids = torch.arange(
                q_len,
                dtype=torch.long,
                device=hidden_states.device,
            ).unsqueeze(0).expand(bsz, -1)

        # 关键修复：
        # 这里第二个参数必须是 seq_len=int，而不是 position_ids tensor。
        cos, sin = attention_module.rotary_emb(
            value_states,
            seq_len=rotary_seq_len,
        )

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )

        # 3. KV cache update
        if cache_obj is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }

            try:
                key_states, value_states = cache_obj.update(
                    key_states,
                    value_states,
                    attention_module.layer_idx,
                    cache_kwargs,
                )
            except TypeError:
                key_states, value_states = cache_obj.update(
                    key_states,
                    value_states,
                    attention_module.layer_idx,
                )

        # 4. GQA repeat_kv
        key_states = repeat_kv(
            key_states,
            attention_module.num_key_value_groups,
        )
        value_states = repeat_kv(
            value_states,
            attention_module.num_key_value_groups,
        )

        kv_seq_len = key_states.shape[-2]

        # 5. QK quantized matmul
        # 老版本 Qwen2Attention 用 / sqrt(head_dim)，不一定有 attention_module.scaling。
        scaling = getattr(
            attention_module,
            "scaling",
            1.0 / math.sqrt(head_dim),
        )

        attn_weights = attention_module.qk_matmul(
            query_states * scaling,
            key_states.transpose(-2, -1),
            stat_collector=attention_module.stat_manager,
        )

        expected_shape = (bsz, num_heads, q_len, kv_seq_len)
        if attn_weights.size() != expected_shape:
            raise ValueError(
                f"Attention weights should be of size {expected_shape}, "
                f"but got {attn_weights.size()}"
            )

        # 6. attention mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :kv_seq_len]
            attn_weights = attn_weights + causal_mask

        # 7. softmax
        attn_weights = F.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)

        attn_weights = F.dropout(
            attn_weights,
            p=attention_module.attention_dropout,
            training=attention_module.training,
        )

        # 8. PV quantized matmul
        attn_output = attention_module.pv_matmul(
            attn_weights,
            value_states,
            stat_collector=attention_module.stat_manager,
        )

        expected_output_shape = (bsz, num_heads, q_len, head_dim)
        if attn_output.size() != expected_output_shape:
            raise ValueError(
                f"attn_output should be of size {expected_output_shape}, "
                f"but got {attn_output.size()}"
            )

        # 9. reshape + output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, attention_module.hidden_size)
        attn_output = attention_module.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # 当前 transformers 版本的 Qwen2DecoderLayer 期望 3 个返回值：
        # hidden_states, self_attn_weights, present_key_value
        return attn_output, attn_weights, cache_obj

    attention_module.forward = quantized_forward

def wrap_qwen_model(
    model,
    quant_config: Dict[str, Any],
    mode: str = "scale_inspection",
    stat_manager: Optional[QuantStatManager] = None,
):
    model = wrap_qwen_linear_only(
        model,
        quant_config,
        mode=mode,
        stat_manager=stat_manager,
    )

    if quant_config.get("quantize_matmul", False):
        print("Injecting Qwen qk_matmul/pv_matmul...")

        for layer_idx, layer in enumerate(model.model.layers):
            inject_qwen_quantized_matmul(
                attention_module=layer.self_attn,
                layer_idx=layer_idx,
                quant_config=quant_config,
                mode=mode,
                stat_manager=stat_manager,
            )

        print(f"Injected Qwen QuantizedMatMul for {len(model.model.layers)} layers")

    return model

def switch_quantization_mode_all(model, mode: str):
    valid_modes = {"raw", "scale_inspection", "quant_forward"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid quantization mode: {mode}")

    count = 0
    for module in model.modules():
        if isinstance(module, (QuantizedLinear, QuantizedMatMul)):
            module.mode = mode
            count += 1

    print(f"Switched {count} quantized modules to mode={mode}")
    return model