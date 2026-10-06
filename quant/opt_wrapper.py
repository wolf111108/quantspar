"""
OPT Model Wrapper for Quantization.

This module provides functions to wrap OPT models with quantized layers,
replacing standard Linear layers and MatMul operations with quantized versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import Dict, Optional, Any
from transformers import OPTForCausalLM

from .quant_linear import QuantizedLinear
from .quant_matmul import QuantizedMatMul, MatMul
from .stat_manager import QuantStatManager


def resolve_calibration_policy(
    quant_config: Dict[str, Any],
    layer_type: str,
    layer_idx: int
) -> str:
    cp = quant_config.get("calibration_policy") or {}
    default_policy = cp.get("default") or "recalibrate"
    layer_policy = cp.get("layer_policy") or {}
    per_layer_policy = cp.get("per_layer_policy") or {}

    key = f"{layer_type}_{layer_idx}"
    policy = per_layer_policy.get(key, layer_policy.get(layer_type, default_policy))
    policy = str(policy).lower()

    valid = {"auto", "reuse", "recalibrate"}
    if policy not in valid:
        raise ValueError(
            f"Invalid calibration policy '{policy}' for {key}, valid: {sorted(valid)}"
        )
    return policy

def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    """
    Get a module by its name from the model.
    
    Args:
        model: PyTorch model
        module_name: Dot-separated module name (e.g., 'model.decoder.layers.0.self_attn.q_proj')
        
    Returns:
        The requested module
    """
    names = module_name.split('.')
    module = model
    for name in names:
        module = getattr(module, name)
    return module


def set_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module):
    """
    Set a module by its name in the model.
    
    Args:
        model: PyTorch model
        module_name: Dot-separated module name
        new_module: New module to set
    """
    names = module_name.split('.')
    parent = model
    for name in names[:-1]:
        parent = getattr(parent, name)
    setattr(parent, names[-1], new_module)


def create_quantized_linear(
    original_layer: nn.Linear,
    layer_type: str,
    layer_idx: int,
    quant_config: Dict[str, Any],
    mode: str = "scale_inspection",
    is_bitnet: bool = False,  # 默认不是Bitnet
) -> QuantizedLinear:
    """
    Create a quantized linear layer from an original linear layer.
    
    Args:
        original_layer: Original nn.Linear layer
        layer_type: Type of layer (e.g., 'q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2')
        layer_idx: Layer index in the model
        quant_config: Quantization configuration dictionary
        mode: Quantization mode ('scale_inspection' or 'quant_forward')
        
    Returns:
        QuantizedLinear layer with copied weights
    """
    # Get quantization parameters for this layer type
    layer_config = quant_config.get(layer_type, {})
    
    # Create quantized layer
    quant_layer = QuantizedLinear(
        in_features=original_layer.in_features,
        out_features=original_layer.out_features,
        bias=original_layer.bias is not None,
        mode=mode,
        a_bit=layer_config.get('a_bit', 8),
        w_bit=layer_config.get('w_bit', 8),
        o_bit=layer_config.get('o_bit', 8),
        d_bit=layer_config.get('d_bit', 4),
        p=layer_config.get('p', 4),
        outlier_ratio=layer_config.get('outlier_ratio',   # 新增
                    quant_config.get('outlier_ratio', 0.0)),
        dynamic_activation=quant_config.get('dynamic_activation', False),
        weight_scale_granularity=quant_config.get(
            'weight_scale_granularity', 'scalar'
        ),
        mixed_precision=quant_config.get('mixed_precision', False),
        mp_high_ratio=quant_config.get('mp_high_ratio', 0.2),
        mp_low_ratio=quant_config.get('mp_low_ratio', 0.3),
        scale_root_str=quant_config.get('scale_dir', ''),
    )
    
    quant_layer.calibration_policy = resolve_calibration_policy(
        quant_config, layer_type, layer_idx
    )

    # Copy weights and bias
    # 防御性对齐：把新建的 quant_layer（默认 fp32/CPU）先移动到原层的
    # device 与 dtype，再做 .data 赋值，避免 torch>=2.6 的
    # "incompatible tensor type" 错误（例如 accelerate offload 场景）。
    _ow = original_layer.weight
    if _ow.device.type != "meta":
        quant_layer = quant_layer.to(
            device=_ow.device,
            dtype=_ow.dtype,
        )
    quant_layer.weight.data = original_layer.weight.data.clone()
    if original_layer.bias is not None:
        quant_layer.bias.data = original_layer.bias.data.clone()
    quant_layer.is_bitnet = is_bitnet
    if is_bitnet:
        quant_layer.bitnet_online_quant = original_layer.online_quant
        quant_layer.bitnet_weight_scale = (
            original_layer.weight_scale.detach().clone()
        )
    
    # Set layer info
    quant_layer.set_layer_info(layer_type, layer_idx)
    
    return quant_layer


def create_quantized_matmul(
    layer_type: str,
    layer_idx: int,
    quant_config: Dict[str, Any],
    mode: str = "scale_inspection",
    is_bitnet: bool = False  # 默认不是Bitnet
) -> QuantizedMatMul:
    """
    Create a quantized matrix multiplication layer.
    
    Args:
        layer_type: Type of matmul ('qk_matmul' or 'pv_matmul')
        layer_idx: Layer index in the model
        quant_config: Quantization configuration dictionary
        mode: Quantization mode
        
    Returns:
        QuantizedMatMul layer
    """
    # Get quantization parameters for this layer type
    layer_config = quant_config.get(layer_type, {})
    
    # Create quantized matmul
    quant_matmul = QuantizedMatMul(
        mode=mode,
        A_bit=layer_config.get('A_bit', 8),
        B_bit=layer_config.get('B_bit', 8),
        O_bit=layer_config.get('O_bit', 10),
        scale_root_str=quant_config.get('scale_dir', ''),
        d_bit=layer_config.get('d_bit', 4),
        p=layer_config.get('p', 4),
        outlier_ratio=layer_config.get('outlier_ratio',        # 新增
                      quant_config.get('outlier_ratio', 0.0)),
        mixed_precision=quant_config.get('mixed_precision', False),
        mp_high_ratio=quant_config.get('mp_high_ratio', 0.1),
        mp_low_ratio=quant_config.get('mp_low_ratio', 0.65),
    )
    quant_matmul.is_bitnet = is_bitnet  # 标记是否为 Bitnet 的量化 MatMul
    quant_matmul.calibration_policy = resolve_calibration_policy(
    quant_config, layer_type, layer_idx
    )

    # Set layer info
    quant_matmul.set_layer_info(layer_type, layer_idx)
    
    return quant_matmul


def wrap_opt_model(
    model: OPTForCausalLM,
    quant_config: Dict[str, Any],
    mode: str = "scale_inspection",
    stat_manager: Optional[QuantStatManager] = None
) -> OPTForCausalLM:
    """
    Wrap OPT model with quantized layers.
    
    This function replaces Linear layers in attention and FFN with quantized versions,
    and injects quantized matrix multiplication operations.
    
    Args:
        model: Original OPT model
        quant_config: Quantization configuration dictionary
        mode: Quantization mode ('scale_inspection' for calibration, 'quant_forward' for inference)
        stat_manager: Statistics manager for collecting calibration data
        
    Returns:
        Model with quantized layers
    """
    print("Wrapping OPT model with quantized layers...")
    
    # Get number of layers
    num_layers = len(model.model.decoder.layers)
    
    # Track replaced modules
    replaced_count = 0
    
    # Replace layers in each decoder layer
    for layer_idx in range(num_layers):
        decoder_layer = model.model.decoder.layers[layer_idx]
        
        # Replace attention projection layers
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            original_layer = getattr(decoder_layer.self_attn, proj_name)
            
            if isinstance(original_layer, nn.Linear):
                quant_layer = create_quantized_linear(
                    original_layer,
                    proj_name,
                    layer_idx,
                    quant_config,
                    mode
                )
                quant_layer._stat_manager = stat_manager
                setattr(decoder_layer.self_attn, proj_name, quant_layer)
                replaced_count += 1
                
                # Register with stat manager
                if stat_manager:
                    stat_manager.register_layer(proj_name, layer_idx)
        
        # Replace FFN layers
        for ffn_name in ['fc1', 'fc2']:
            original_layer = getattr(decoder_layer, ffn_name)
            
            if isinstance(original_layer, nn.Linear):
                quant_layer = create_quantized_linear(
                    original_layer,
                    ffn_name,
                    layer_idx,
                    quant_config,
                    mode
                )
                quant_layer._stat_manager = stat_manager
                setattr(decoder_layer, ffn_name, quant_layer)
                replaced_count += 1
                
                # Register with stat manager
                if stat_manager:
                    stat_manager.register_layer(ffn_name, layer_idx)
        
        # Inject quantized matrix multiplication
        # Note: This requires modifying the forward method, which we'll do via monkey patching
        _inject_quantized_matmul(
            decoder_layer.self_attn,
            layer_idx,
            quant_config,
            mode,
            stat_manager
        )
    
    # Optionally replace LM head
    if quant_config.get('lm_head', {}).get('enabled', False):
        original_lm_head = model.lm_head
        if isinstance(original_lm_head, nn.Linear):
            quant_lm_head = create_quantized_linear(
                original_lm_head,
                'lm_head',
                0,  # LM head is not layer-specific
                quant_config,
                mode
            )
            quant_lm_head._stat_manager = stat_manager
            model.lm_head = quant_lm_head
            replaced_count += 1
            
            if stat_manager:
                stat_manager.register_layer('lm_head', 0)
    
    print(f"Replaced {replaced_count} linear layers with quantized versions")
    print(f"Model wrapping complete!")
    
    return model


def _inject_quantized_matmul(
    attention_module: nn.Module,
    layer_idx: int,
    quant_config: Dict[str, Any],
    mode: str,
    stat_manager: Optional[QuantStatManager]
):
    """
    Inject quantized matrix multiplication into attention module.
    
    This function monkey-patches the forward method of OPTAttention to use
    quantized matrix multiplication for Q@K^T and P@V operations.
    
    Args:
        attention_module: OPTAttention module
        layer_idx: Layer index
        quant_config: Quantization configuration
        mode: Quantization mode
        stat_manager: Statistics manager
    """
    # Create quantized matmul modules
    qk_matmul = create_quantized_matmul('qk_matmul', layer_idx, quant_config, mode)
    pv_matmul = create_quantized_matmul('pv_matmul', layer_idx, quant_config, mode)
    
    # Register with stat manager
    if stat_manager:
        stat_manager.register_layer('qk_matmul', layer_idx)
        stat_manager.register_layer('pv_matmul', layer_idx)
    
    # Store quantized matmul modules as attributes
    attention_module.qk_matmul = qk_matmul
    attention_module.pv_matmul = pv_matmul
    attention_module.stat_manager = stat_manager
    
    # Save original forward method
    original_forward = attention_module.forward
    
    # Define new forward method with quantized matmul
    def quantized_forward(
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        **kwargs 
    ):
        """Modified forward with quantized matrix multiplication."""
        
        # ===== Step 1: Get dimensions and compute Q, K, V =====
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        
        # Get query projection
        query_states = attention_module.q_proj(hidden_states) * attention_module.scaling
        
        # Get key, value projections
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = attention_module._shape(attention_module.k_proj(key_value_states), -1, bsz)
            value_states = attention_module._shape(attention_module.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = attention_module._shape(attention_module.k_proj(hidden_states), -1, bsz)
            value_states = attention_module._shape(attention_module.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = attention_module._shape(attention_module.k_proj(hidden_states), -1, bsz)
            value_states = attention_module._shape(attention_module.v_proj(hidden_states), -1, bsz)
        
        if attention_module.is_decoder:
            past_key_value = (key_states, value_states)
        
        # ===== Step 2: Reshape to multi-head format =====
        proj_shape = (bsz * attention_module.num_heads, -1, attention_module.head_dim)
        query_states = attention_module._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        
        src_len = key_states.size(1)
        
        # ===== Step 3: Compute attention weights using QUANTIZED matmul =====
        # 原始: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        # 改为: 使用 qk_matmul
        attn_weights = attention_module.qk_matmul(
            query_states, 
            key_states.transpose(1, 2),
            stat_collector=attention_module.stat_manager
        )
        
        if attn_weights.size() != (bsz * attention_module.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * attention_module.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        # ===== Step 4: Apply attention mask =====
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, attention_module.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * attention_module.num_heads, tgt_len, src_len)
        
        # ===== Step 5: Apply softmax =====
        if attn_weights.dtype == torch.float16:
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = F.softmax(attn_weights, dim=-1)
        
        # ===== Step 6: Apply layer head mask =====
        if layer_head_mask is not None:
            if layer_head_mask.size() != (attention_module.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(attention_module.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, attention_module.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * attention_module.num_heads, tgt_len, src_len)
        
        # ===== Step 7: Prepare output attention weights =====
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, attention_module.num_heads, tgt_len, src_len)
            attn_weights_out = attn_weights_reshaped.view(bsz * attention_module.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
            attn_weights_out = attn_weights
        
        # ===== Step 8: Apply dropout =====
        attn_probs = F.dropout(attn_weights_out, p=attention_module.dropout, training=attention_module.training)
        
        # ===== Step 9: Compute attention output using QUANTIZED matmul =====
        # 原始: attn_output = torch.bmm(attn_probs, value_states)
        # 改为: 使用 pv_matmul
        attn_output = attention_module.pv_matmul(
            attn_probs,
            value_states,
            stat_collector=attention_module.stat_manager
        )
        
        if attn_output.size() != (bsz * attention_module.num_heads, tgt_len, attention_module.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, attention_module.num_heads, tgt_len, attention_module.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # ===== Step 10: Reshape back to original format =====
        attn_output = attn_output.view(bsz, attention_module.num_heads, tgt_len, attention_module.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, attention_module.embed_dim)
        
        # ===== Step 11: Output projection =====
        attn_output = attention_module.out_proj(attn_output)
        
        return attn_output, attn_weights_reshaped, past_key_value
    
    # Replace forward method
    # Note: Full implementation would require more sophisticated patching
    # For now, we store the modules and will use hooks in the calibration/inference code
    attention_module.forward = quantized_forward
    attention_module._original_forward = original_forward


def switch_quantization_mode(

    model: OPTForCausalLM,

    mode: str

) -> OPTForCausalLM:

    """

    Switch quantization mode for all quantized layers in the model.

    

    This function traverses the model and changes the mode of all QuantizedLinear

    and QuantizedMatMul layers to the specified mode.

    

    Args:

        model: OPT model with quantized layers

        mode: Target mode - one of:

            - "raw": No quantization, use original floating point computation

            - "scale_inspection": Collect statistics for scale calculation (calibration)

            - "quant_forward": Use quantized computation (inference)

    

    Returns:

        Model with updated quantization mode

    

    Raises:

        ValueError: If mode is not one of the valid options

    """

    valid_modes = ["raw", "scale_inspection", "quant_forward"]

    if mode not in valid_modes:

        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

    

    print(f"Switching quantization mode to: {mode}")

    

    changed_count = 0

    

    # Traverse all decoder layers

    num_layers = len(model.model.decoder.layers)

    for layer_idx in range(num_layers):

        decoder_layer = model.model.decoder.layers[layer_idx]

        

        # Switch mode for attention projection layers

        for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:

            layer = getattr(decoder_layer.self_attn, proj_name)

            if isinstance(layer, QuantizedLinear):

                layer.mode = mode

                changed_count += 1

        

        # Switch mode for FFN layers

        for ffn_name in ["fc1", "fc2"]:

            layer = getattr(decoder_layer, ffn_name)

            if isinstance(layer, QuantizedLinear):

                layer.mode = mode

                changed_count += 1

        

        # Switch mode for quantized matmul operations

        if hasattr(decoder_layer.self_attn, "qk_matmul"):

            if isinstance(decoder_layer.self_attn.qk_matmul, QuantizedMatMul):

                decoder_layer.self_attn.qk_matmul.mode = mode

                changed_count += 1

        

        if hasattr(decoder_layer.self_attn, "pv_matmul"):

            if isinstance(decoder_layer.self_attn.pv_matmul, QuantizedMatMul):

                decoder_layer.self_attn.pv_matmul.mode = mode

                changed_count += 1

    

    # Switch mode for LM head if quantized

    if isinstance(model.lm_head, QuantizedLinear):

        model.lm_head.mode = mode

        changed_count += 1

    

    print(f"✓ Switched mode for {changed_count} quantized layers")

    

    return model




def unwrap_opt_model(model: OPTForCausalLM) -> OPTForCausalLM:
    """
    Unwrap OPT model, converting quantized layers back to standard layers.
    
    Args:
        model: Model with quantized layers
        
    Returns:
        Model with standard layers
    """
    print("Unwrapping quantized OPT model...")
    
    num_layers = len(model.model.decoder.layers)
    unwrapped_count = 0
    
    for layer_idx in range(num_layers):
        decoder_layer = model.model.decoder.layers[layer_idx]
        
        # Unwrap attention projection layers
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            layer = getattr(decoder_layer.self_attn, proj_name)
            
            if isinstance(layer, QuantizedLinear):
                # Create standard linear layer
                standard_layer = nn.Linear(
                    layer.in_features,
                    layer.out_features,
                    bias=layer.bias is not None
                )
                standard_layer.weight.data = layer.weight.data.clone()
                if layer.bias is not None:
                    standard_layer.bias.data = layer.bias.data.clone()
                
                setattr(decoder_layer.self_attn, proj_name, standard_layer)
                unwrapped_count += 1
        
        # Unwrap FFN layers
        for ffn_name in ['fc1', 'fc2']:
            layer = getattr(decoder_layer, ffn_name)
            
            if isinstance(layer, QuantizedLinear):
                standard_layer = nn.Linear(
                    layer.in_features,
                    layer.out_features,
                    bias=layer.bias is not None
                )
                standard_layer.weight.data = layer.weight.data.clone()
                if layer.bias is not None:
                    standard_layer.bias.data = layer.bias.data.clone()
                
                setattr(decoder_layer, ffn_name, standard_layer)
                unwrapped_count += 1
    
    # Unwrap LM head if quantized
    if isinstance(model.lm_head, QuantizedLinear):
        standard_lm_head = nn.Linear(
            model.lm_head.in_features,
            model.lm_head.out_features,
            bias=model.lm_head.bias is not None
        )
        standard_lm_head.weight.data = model.lm_head.weight.data.clone()
        if model.lm_head.bias is not None:
            standard_lm_head.bias.data = model.lm_head.bias.data.clone()
        
        model.lm_head = standard_lm_head
        unwrapped_count += 1
    
    print(f"Unwrapped {unwrapped_count} quantized layers")
    
    return model
