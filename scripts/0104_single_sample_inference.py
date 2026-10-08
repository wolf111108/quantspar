"""
Main Entry Point for OPT Model Quantization Pipeline.


Usage:
    python 0103_quant_pipeline_main.py --config config/Int8.yaml --model-path /path/to/opt-model
"""

import os
import sys
import argparse
import torch
import time
import pickle
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant import load_config, QuantStatManager
from quant.quant_linear import QuantizedLinear
from quant.quant_matmul import QuantizedMatMul
from quant.quant_spec import resolve_model_dtype

def _load_full_model_dependencies():
    """Load optional full-model dependencies after argument parsing."""
    try:
        from tqdm import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from quant.model_wrapper import wrap_model_by_family
        from quant.qwen_wrapper import switch_quantization_mode_all
        from others.data import CalibrationDataLoader
    except ImportError as exc:
        raise RuntimeError(
            "Full-model entry requires transformers, quant.model_wrapper, "
            "quant.qwen_wrapper and others.data from the complete project. "
            "Use python -m scripts.profile_fp8_int4 for the self-contained core path."
        ) from exc
    globals().update({name:value for name,value in locals().items() if not name.startswith('_')})

def enable_single_sample_inference_for_quantized_linear():
    def _is_single_sample_inference(self):
        return True

    QuantizedLinear._is_single_sample_inference = _is_single_sample_inference


enable_single_sample_inference_for_quantized_linear()

def validate_reuse_layers_have_scales(model):
    missing = []
    for m in model.modules():
        if isinstance(m, (QuantizedLinear, QuantizedMatMul)):
            action = m._resolve_calibration_action()
            if action == "reuse" and (not m._scale_files_exist()):
                missing.append(f"{m.layer_name}_{m.layer_idx}")
    if missing:
        raise FileNotFoundError(
            "These layers are set to reuse but scale files are missing:\n" +
            "\n".join(missing)
        )

def calibration_action_summary(model):
    summary = {"reuse": 0, "recalibrate": 0}
    for m in model.modules():
        if isinstance(m, (QuantizedLinear, QuantizedMatMul)):
            summary[m._resolve_calibration_action()] += 1
    return summary

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete OPT Model Quantization Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to quantization configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to pretrained OPT model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration if scales already exist"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation after quantization"
    )

    parser.add_argument("--stats-output-dir", type=str,
                        help="fresh directory for prefill/decode CIM JSON statistics")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, my dog is cute",
        help="Input text for token generation during evaluation (default: a fixed prompt)"
    )
    parser.add_argument(
        "--nmacro",
        type=int,
        default=32,
        help="Number of tensor-core macro units for Mapping_stat (default: 32)"
    )
    parser.add_argument(
        "--as-l",
        type=int,
        default=1,
        help="Mapping_stat_dynamic sync model: 1=async (as_latency), 0=sync (sy_latency) (default: 1)"
    )

    return parser.parse_args()


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def build_wrapped_model(args, config, scale_dir, mode="scale_inspection"):
    model_kwargs = dict(
        torch_dtype=resolve_model_dtype(config, args.device),
        device_map="auto",
        trust_remote_code=True,
    )

    model_family = config.get("quantization", {}).get("model_family", "opt").lower()

    if config.get("model", {}).get("attn_implementation") is not None:
        model_kwargs["attn_implementation"] = config["model"]["attn_implementation"]
    elif model_family in {"qwen3.5", "qwen3_5"}:
        # Qwen3.5 的 qk/pv matmul patch 需要 eager attention
        model_kwargs["attn_implementation"] = "eager"

    # Qwen3.5 是 VL 多模态模型（Qwen3_5ForConditionalGeneration）
    if model_family in {"qwen3.5", "qwen3_5"}:
        from transformers import AutoModelForImageTextToText as _AutoModelClass
    else:
        _AutoModelClass = AutoModelForCausalLM

    model = _AutoModelClass.from_pretrained(args.model_path, **model_kwargs)

    stat_manager = QuantStatManager(scale_dir, nmacro=args.nmacro, as_l=args.as_l)
    model = wrap_model_by_family(
        model,
        config["quantization"],
        mode=mode,
        stat_manager=stat_manager,
    )

    return model, stat_manager


def calibrate(args, config):
    print_header("STEP 1: CALIBRATION")

    scale_dir = config["quantization"]["scale_dir"]
    os.makedirs(scale_dir, exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    model, stat_manager = build_wrapped_model(
        args=args,
        config=config,
        scale_dir=scale_dir,
        mode="scale_inspection",
    )

    # validate_reuse_layers_have_scales(model)
    summary = calibration_action_summary(model)
    print(f"Calibration action summary: {summary}")

    # 只有所有层都是 reuse，才允许真正跳过 calibration。
    if args.skip_calibration:
        if summary["recalibrate"] == 0:
            print("✓ All quantized layers are reuse; skipping calibration.")
            return model
        else:
            print(
                "⚠ --skip-calibration is set, but some layers still require recalibration. "
                "Continuing calibration."
            )

    if summary["recalibrate"] == 0:
        print("All quantized layers are reuse; skip calibration dataloader.")
        return model

    print("\nPreparing calibration data...")
    calib_cfg = config["calibration"]

    calib_loader = CalibrationDataLoader(
        dataset_name=calib_cfg["dataset"],
        dataset_config=calib_cfg.get("dataset_config", None),
        split=calib_cfg.get("split", "train"),
        model_name_or_path=args.model_path,
        seq_length=calib_cfg.get("seq_length", 2048),
        batch_size=calib_cfg.get("batch_size", 1),
        num_samples=calib_cfg.get("num_samples", None),
        seed=calib_cfg.get("seed", 42),
        text_column=calib_cfg.get("text_column", "text"),
        streaming=calib_cfg.get("streaming", False),
        min_text_tokens=calib_cfg.get("min_text_tokens", 32),
    )

    print(f"\nRunning calibration on {len(calib_loader)} batches...")
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(calib_loader, desc="Calibrating")):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)

            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    calibration_time = time.time() - start_time

    print("\n" + "-" * 80)
    stat_manager.print_summary()

    print("Saving quantization scales...")
    stat_manager.save_all_scales()

    print(f"\n✓ Calibration completed in {calibration_time:.2f}s")
    print(f"✓ Scales saved to: {scale_dir}")

    return model

def _bind_stat_manager(model, stat_manager):
    """Bind stat_manager to all quantized modules and attention wrappers."""
    for module in model.modules():
        if isinstance(module, (QuantizedLinear, QuantizedMatMul)):
            module._stat_manager = stat_manager

        # Qwen/OPT attention wrapper stores stat_manager on attention_module.
        # Only rebind modules that actually own quantized attention matmuls.
        if hasattr(module, "qk_matmul") or hasattr(module, "pv_matmul"):
            if hasattr(module, "stat_manager"):
                module.stat_manager = stat_manager
            if hasattr(module, "qk_matmul"):
                module.qk_matmul._stat_manager = stat_manager
            if hasattr(module, "pv_matmul"):
                module.pv_matmul._stat_manager = stat_manager


def evaluate(args, config, model):
    print_header("STEP 2: QUANTIZATION & EVALUATION")

    stat_manager_prefill = QuantStatManager(config["quantization"]["scale_dir"], nmacro=args.nmacro, as_l=args.as_l)
    stat_manager_decode = QuantStatManager(config["quantization"]["scale_dir"], nmacro=args.nmacro, as_l=args.as_l)

    _bind_stat_manager(model, stat_manager_prefill)

    model = switch_quantization_mode_all(model, "quant_forward")

    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nApplying quantization (quant_forward mode)...")
    print("✓ Quantization applied successfully")

    if args.skip_evaluation:
        print("\n✓ Skipping evaluation (--skip-evaluation flag set)")
        return

    print("\n" + "-" * 80)
    print("STEP 3: PERPLEXITY EVALUATION")
    print("-" * 80)

    results = {}

    # ============================================================
    # 从 config 的 prefill_decode_profile 加载数据做推理
    # ============================================================
    pd_config = config.get("prefill_decode_profile", {})
    prefill_length = pd_config.get("prefill_length", 1024)
    decode_steps = pd_config.get("decode_steps", 64)
    pd_dataset_name = pd_config.get("dataset", "HuggingFaceFW/fineweb")
    pd_dataset_config = pd_config.get("dataset_config", "sample-10BT")
    pd_split = pd_config.get("split", "train")
    pd_streaming = pd_config.get("streaming", True)
    pd_text_column = pd_config.get("text_column", "text")
    pd_min_doc_tokens = pd_config.get("min_doc_tokens", 4096)
    pd_sample_offset = pd_config.get("sample_offset", 0)   # 跳过前 N 篇文档
    pd_start_offset  = pd_config.get("start_offset", 0)    # 在文档内跳过前 N 个 token 再开始

    print(f"\nLoading data from {pd_dataset_name} ({pd_dataset_config}, {pd_split})...")
    print(f"prefill_length={prefill_length}, decode_steps={decode_steps}, min_doc_tokens={pd_min_doc_tokens}")
    print(f"sample_offset={pd_sample_offset}, start_offset={pd_start_offset}")

    from datasets import load_dataset as _load_dataset

    pd_ds = _load_dataset(
        pd_dataset_name,
        pd_dataset_config,
        split=pd_split,
        streaming=pd_streaming,
    )

    # 取第 pd_sample_offset 篇满足 min_doc_tokens 的文档，默认 0（第一篇）
    input_ids_full = None
    skip_count = 0
    needed_tokens = pd_start_offset + prefill_length + decode_steps
    for sample in pd_ds:
        raw_text = sample[pd_text_column]
        ids = tokenizer(raw_text, return_tensors="pt").input_ids[0]
        if ids.shape[0] >= max(pd_min_doc_tokens, needed_tokens):
            if skip_count < pd_sample_offset:
                skip_count += 1
                continue
            input_ids_full = ids
            break

    if input_ids_full is None:
        raise RuntimeError(
            f"未能找到 >= {max(pd_min_doc_tokens, needed_tokens)} tokens 的文档 "
            f"(sample_offset={pd_sample_offset}, start_offset={pd_start_offset})"
        )

    total_len = input_ids_full.shape[0]
    print(f"Found document with {total_len} tokens")

    # 从 start_offset 开始截取 prefill + decode 所需 token
    start = pd_start_offset
    prefill_ids  = input_ids_full[start:start + prefill_length].unsqueeze(0).to(args.device)
    decode_gt_ids = input_ids_full[start + prefill_length:start + prefill_length + decode_steps]

    start_time = time.time()

    # ---- prefill 阶段: 统计激活稀疏度 ----
    stat_manager_prefill.set_phase("prefill")
    with torch.no_grad():
        outputs = model(prefill_ids, use_cache=True)

    # ---- decode 阶段: teacher-forced, 统计 decode 稀疏度 ----
    _bind_stat_manager(model, stat_manager_decode)
    stat_manager_decode.set_phase("decode")
    past_key_values = outputs.past_key_values
    generated_ids = [prefill_ids[0, -1].item()]

    with torch.no_grad():
        for step in range(decode_steps):
            if step < decode_gt_ids.shape[0]:
                next_token = decode_gt_ids[step].unsqueeze(0).to(args.device)
            else:
                logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else None
                if logits is None:
                    break
                next_token = torch.argmax(logits, dim=-1)

            step_input = next_token.unsqueeze(0)
            outputs = model(step_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values

    eval_time = time.time() - start_time

    eval_key = f"{pd_dataset_name}:{pd_dataset_config}:{pd_split}"
    results[eval_key] = {
        "time": eval_time,
        "tokens": prefill_length + decode_steps,
    }

    # ============================================================
    # Nop 表: 根据 config.model.name 自动选择模型结构,
    # S / S_kv 由实际 prefill_length / decode_steps 决定
    #
    # 结构库: L, h, num_heads, head_dim, KV_dim (GQA), inter, has_gate
    #   has_gate=False: FFN = fc1 + fc2 (OPT, 非 SwiGLU)
    #   has_gate=True:  FFN = gate + up + down (Qwen / BitNet, SwiGLU)
    # ============================================================
    MODEL_SHAPES = {
        # kv_heads: GQA 的 num_kv_heads; MHA (OPT) = heads
        # KV_dim = kv_heads × head_dim
        "opt-1.3b":      dict(L=24, h=2048, heads=32, kv_heads=32, head_dim=64,  inter=8192,  has_gate=False),
        "opt-6.7b":      dict(L=32, h=4096, heads=32, kv_heads=32, head_dim=128, inter=16384, has_gate=False),
        "qwen2.5-1.5b":  dict(L=28, h=1536, heads=12, kv_heads=2,  head_dim=128, inter=8960,  has_gate=True),
        "qwen2.5-7b":    dict(L=28, h=3584, heads=28, kv_heads=4,  head_dim=128, inter=18944, has_gate=True),
        "qwen2.5-14b":   dict(L=48, h=5120, heads=40, kv_heads=8,  head_dim=128, inter=13824, has_gate=True),
        "bitnet-b1.58-2b": dict(L=30, h=2560, heads=20, kv_heads=5, head_dim=128, inter=6912, has_gate=True),
    }

    model_name = str(config.get("model", {}).get("name", "")).lower()
    shape = None
    for key, cand in MODEL_SHAPES.items():
        if key in model_name or model_name in key:
            shape = cand
            print(f"Nop table: using model shape '{key}' for '{model_name}'")
            break
    if shape is None:
        raise ValueError(
            f"config model.name='{model_name}' 不在 MODEL_SHAPES 中, "
            f"支持: {list(MODEL_SHAPES.keys())}"
        )

    L   = shape["L"]
    h   = shape["h"]
    H   = shape["heads"]
    hd  = shape["head_dim"]
    kvd = shape["kv_heads"] * shape["head_dim"]   # KV_dim = num_kv_heads × head_dim
    FFN = shape["inter"]
    has_gate_model = shape["has_gate"]

    # 实际序列长度: prefill 一次性前向; decode 每步 S_decode=1, cache=S_kv
    S_pre = prefill_length
    S_kv = prefill_length + decode_steps   # decode 时 cache 里已有的 token 数
    print(f"Nop table: S_prefill={S_pre}, S_kv(decode)={S_kv}, L={L}, h={h}, heads={H}, head_dim={hd}")

    def _build_nop(S_query, S_kv_val):
        """按 query 长度和 kv cache 长度生成各层算力表 (FLOPs)。
        prefill: S_query=S_kv_val=S; decode: S_query=1, S_kv_val=cache。
        布局 (has_gate): [q, qk, pv, o, gate, up, down]
                  (else):  [q, qk, pv, o, up, down]
        """
        nop = []
        nop.append(L*2*S_query*(h**2 + 2*kvd*h))   # q/k/v_proj 合并 (GQA: h² + 2×KV_dim×h)
        nop.append(L*2*H*S_query*hd*S_kv_val)      # qk_matmul
        nop.append(L*2*H*S_query*S_kv_val*hd)      # pv_matmul
        nop.append(L*2*S_query*(h**2))             # o_proj
        if has_gate_model:
            nop.append(L*2*S_query*h*FFN)          # gate_proj
        nop.append(L*2*S_query*h*FFN)              # up_proj / fc1
        nop.append(L*2*S_query*FFN*h)              # down_proj / fc2
        return nop

    Nop_prefill_list = _build_nop(S_pre, S_pre)
    Nop_decode_list  = _build_nop(1, S_kv)

    Nop = [n / 1e12 for n in Nop_prefill_list]  # 当前打印块使用 prefill 半段, 转换为 TFLOPS


    print(f"\nInference completed: prefill={prefill_length} tokens, decode={decode_steps} steps")
    print("\n" + "-" * 80)

    # ============================================================
    # 每层固定总算力 (TFLOPS)
    # 通用 Nop 解析: 自动适配有/无 gate_proj 的模型
    # 每阶段布局:
    #   无 gate: [q_proj, qk_matmul, pv_matmul, o_proj, up_proj, down_proj]
    #   有 gate: [q_proj, qk_matmul, pv_matmul, o_proj, gate_proj, up_proj, down_proj]
    # ============================================================
    has_gate = (len(Nop) == 7)
    n_per_phase = 7 if has_gate else 6
    Nop_decode = [n / 1e12 for n in Nop_decode_list]  # decode 半段 (TFLOPS)

    # prefill 阶段算力
    Nop_q_proj_prefill    = Nop[0]
    Nop_qk_matmul_prefill = Nop[1]
    Nop_pv_matmul_prefill = Nop[2]
    Nop_o_proj_prefill    = Nop[3]
    if has_gate:
        Nop_gate_proj_prefill = Nop[4]
        Nop_up_proj_prefill   = Nop[5]
        Nop_down_proj_prefill = Nop[6]
    else:
        Nop_gate_proj_prefill = 0
        Nop_up_proj_prefill   = Nop[4]
        Nop_down_proj_prefill = Nop[5]
    Nop_all_linear_prefill = Nop_q_proj_prefill + Nop_o_proj_prefill + Nop_gate_proj_prefill + Nop_up_proj_prefill + Nop_down_proj_prefill
    Nop_all_matmul_prefill = Nop_qk_matmul_prefill + Nop_pv_matmul_prefill
    Nop_all_prefill = Nop_all_linear_prefill + Nop_all_matmul_prefill

    # decode 阶段算力 (S_decode=1, S_kv=prefill+decode_steps)
    Nop_q_proj_decode    = Nop_decode[0]
    Nop_qk_matmul_decode = Nop_decode[1]
    Nop_pv_matmul_decode = Nop_decode[2]
    Nop_o_proj_decode    = Nop_decode[3]
    if has_gate:
        Nop_gate_proj_decode = Nop_decode[4]
        Nop_up_proj_decode   = Nop_decode[5]
        Nop_down_proj_decode = Nop_decode[6]
    else:
        Nop_gate_proj_decode = 0
        Nop_up_proj_decode   = Nop_decode[4]
        Nop_down_proj_decode = Nop_decode[5]
    Nop_all_linear_decode = Nop_q_proj_decode + Nop_o_proj_decode + Nop_gate_proj_decode + Nop_up_proj_decode + Nop_down_proj_decode
    Nop_all_matmul_decode = Nop_qk_matmul_decode + Nop_pv_matmul_decode
    Nop_all_decode = Nop_all_linear_decode + Nop_all_matmul_decode

    # prefill + decode 合并算力
    Nop_q_proj_merged    = Nop_q_proj_prefill    + Nop_q_proj_decode
    Nop_qk_matmul_merged = Nop_qk_matmul_prefill + Nop_qk_matmul_decode
    Nop_pv_matmul_merged = Nop_pv_matmul_prefill + Nop_pv_matmul_decode
    Nop_o_proj_merged    = Nop_o_proj_prefill    + Nop_o_proj_decode
    Nop_gate_proj_merged = Nop_gate_proj_prefill + Nop_gate_proj_decode
    Nop_up_proj_merged   = Nop_up_proj_prefill   + Nop_up_proj_decode
    Nop_down_proj_merged = Nop_down_proj_prefill + Nop_down_proj_decode
    Nop_all_linear_merged = Nop_q_proj_merged + Nop_o_proj_merged + Nop_gate_proj_merged + Nop_up_proj_merged + Nop_down_proj_merged
    Nop_all_matmul_merged = Nop_qk_matmul_merged + Nop_pv_matmul_merged
    Nop_all_merged = Nop_all_linear_merged + Nop_all_matmul_merged

    # import csv as _csv

    # def _write_profile_csv(filename, label, sm, nop_map):
    #     """Generate a CSV from a single stat_manager instance.
        
    #     sm: stat_manager instance
    #     nop_map: dict mapping layer prefix to Nop value for this CSV
    #     """
    #     _rows = []
    #     for prefix in ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
    #         base_lat   = getattr(sm, f"{prefix}_baseline_latency_stat")
    #         sacim_lat  = getattr(sm, f"{prefix}_SACIM_latency_stat")
    #         util       = sacim_lat / base_lat if base_lat > 0 else 0
    #         speedup    = base_lat / sacim_lat if sacim_lat > 0 else 0
    #         nop_val    = nop_map[prefix]
    #         base_tput  = nop_val / base_lat if base_lat > 0 else 0
    #         sacim_tput = nop_val / sacim_lat if sacim_lat > 0 else 0
    #         _rows.append([prefix, util, speedup, base_lat, sacim_lat, nop_val, base_tput, sacim_tput])

    #     # all_linear
    #     base_lat   = sm.baseline_latency_stat
    #     sacim_lat  = sm.latency
    #     util       = sacim_lat / base_lat if base_lat > 0 else 0
    #     speedup    = base_lat / sacim_lat if sacim_lat > 0 else 0
    #     nop_all    = nop_map["all_linear"]
    #     base_tput  = nop_all / base_lat if base_lat > 0 else 0
    #     sacim_tput = nop_all / sacim_lat if sacim_lat > 0 else 0
    #     _rows.append(["all_linear", util, speedup, base_lat, sacim_lat, nop_all, base_tput, sacim_tput])

    #     _csv_path = f"/{filename}"
    #     with open(_csv_path, "w", newline="") as f:
    #         writer = _csv.writer(f)
    #         writer.writerow(["layer", "utilization", "speedup", "baseline_latency", "SACIM_latency", "Nop", "baseline_throughput", "SACIM_throughput"])
    #         writer.writerows(_rows)
    #     print(f"{label} CSV written to {_csv_path}")

    # _nop_prefill = {
    #     "q_proj": Nop_q_proj_prefill,
    #     "o_proj": Nop_o_proj_prefill,
    #     "up_proj": Nop_up_proj_prefill,
    #     "down_proj": Nop_down_proj_prefill,
    #     "all_linear": Nop_all_linear_prefill,
    # }
    # _nop_decode = {
    #     "q_proj": Nop_q_proj_decode,
    #     "o_proj": Nop_o_proj_decode,
    #     "up_proj": Nop_up_proj_decode,
    #     "down_proj": Nop_down_proj_decode,
    #     "all_linear": Nop_all_linear_decode,
    # }
    # _nop_merged = {
    #     "q_proj": Nop_q_proj_merged,
    #     "o_proj": Nop_o_proj_merged,
    #     "up_proj": Nop_up_proj_merged,
    #     "down_proj": Nop_down_proj_merged,
    #     "all_linear": Nop_all_linear_merged,
    # }

    # # --- prefill ---
    # _write_profile_csv("profile_prefill.csv", "prefill", stat_manager_prefill, _nop_prefill)

    # # --- decode ---
    # _write_profile_csv("profile_decode.csv", "decode", stat_manager_decode, _nop_decode)

    # # --- prefill + decode (simple sum) ---
    # class _MergedSM:
    #     pass
    # _sm_merged = _MergedSM()
    # for _attr in ["q_proj_baseline_latency_stat", "q_proj_SACIM_latency_stat",
    #                "o_proj_baseline_latency_stat", "o_proj_SACIM_latency_stat",
    #                "up_proj_baseline_latency_stat", "up_proj_SACIM_latency_stat",
    #                "down_proj_baseline_latency_stat", "down_proj_SACIM_latency_stat",
    #                "baseline_latency_stat", "latency"]:
    #     setattr(_sm_merged, _attr,
    #             getattr(stat_manager_prefill, _attr, 0) + getattr(stat_manager_decode, _attr, 0))
    # _write_profile_csv("profile_prefill_decode.csv", "prefill+decode", _sm_merged, _nop_merged)



    # _dc = stat_manager_decode
    # if _dc.latency>0:
    #     # print(f"utilization: {_sm.ideal_sparsity_latency_stat/_sm.latency:.12f}")
    #     # print(
    #     #     f"IA value sparsity: "
    #     #     f"{_sm.activation_zero_count / _sm.activation_element_count:.4%}"
    #     # )
    #     # print(
    #     #     f"IA bit sparsity: "
    #     #     f"{_sm.activation_0bit_count / _sm.activation_bit_count:.4%}"
    #     #     f"{_sm.activation_sparsebit_count / _sm.activation_bit_count:.4%}"
    #     #     f"{_sm.activation_amplitude_zero_bits_total / _sm.activation_bit_count:.4%}"
    #     # )

    #     # print(
    #     #     f"W value sparsity: "
    #     #     f"{_sm.weight_zero_count / _sm.weight_element_count:.4%}"
    #     # )
    #     # print(
    #     #     f"W bit sparsity: "
    #     #     f"{_sm.weight_sparsebit_count / _sm.weight_bit_count:.4%}"
    #     # )

    #     # print(
    #     #     f"KV value sparsity: "
    #     #     f"{_sm.dynamic_weight_zero_count / _sm.dynamic_weight_element_count:.4%}"
    #     # )
    #     # print(
    #     #     f"KV bit sparsity: "
    #     #     f"{_sm.dynamic_weight_sparsebit_count / _sm.dynamic_weight_bit_count:.4%}"
    #     # )


    #     # 保存 pv_matmul per-token effective_one_bits 平均值到 pickle
    #     # if _sm.pv_count > 0 and _sm.pv_per_token_effective_one_bits_all is not None:
    #     #     avg_per_token = _sm.pv_per_token_effective_one_bits_all.float() / _sm.pv_count
    #     #     pv_pickle_path = os.path.join(config['quantization']['scale_dir'], 'pv_per_token_effective_one_bits_avg.p')
    #     #     os.makedirs(os.path.dirname(pv_pickle_path), exist_ok=True)
    #     #     with open(pv_pickle_path, 'wb') as f:
    #     #         pickle.dump(avg_per_token.cpu(), f)
    #     #     print(f"pv_per_token_effective_one_bits_avg saved to {pv_pickle_path}")
    #     #     print(f"  shape: {avg_per_token.shape}, pv_count: {_sm.pv_count}")

    #     # print(
    #     #     f"pv_amplitude_zero_ratio: "
    #     #     f"{_sm.pv_amplitude_zero_bits_all / _sm.pv_total_bits:.6f}"
    #     # )

    #     # print(
    #     #     f"pv_amplitude_zero_ratio_no_causal: "
    #     #     f"{_sm.pv_amplitude_zero_bits_no_causal / _sm.pv_total_bits:.6f}"
    #     # )

    #     matmul_latency = _dc.qk_matmul_SACIM_latency_stat + _dc.pv_matmul_SACIM_latency_stat
    #     linear_latency = _dc.latency - matmul_latency

    #     matmul_ideal_latency = _dc.qk_matmul_sparsity_speedup + _dc.pv_matmul_sparsity_speedup
    #     linear_ideal_latency = _dc.ideal_sparsity_latency_stat - matmul_ideal_latency

    #     print(f"qkv SACIM: {_dc.q_proj_SACIM_latency_stat:.12f} s")
    #     print(f"qkv baseline: {_dc.q_proj_baseline_latency_stat:.12f} s")
    #     print(f"qkv speedup: {_dc.q_proj_baseline_latency_stat/_dc.q_proj_SACIM_latency_stat:.12f}")
    #     print(f"qkv TOPS: {Nop_q_proj_decode/_dc.q_proj_SACIM_latency_stat:.12f}")

    #     print(f"qk SACIM: {_dc.qk_matmul_SACIM_latency_stat:.12f} s")
    #     print(f"qk baseline: {_dc.qk_matmul_baseline_latency_stat:.12f} s")
    #     print(f"qk speedup: {_dc.qk_matmul_baseline_latency_stat/_dc.qk_matmul_SACIM_latency_stat:.12f}")
    #     print(f"qk TOPS: {Nop_qk_matmul_decode/_dc.qk_matmul_SACIM_latency_stat:.12f}")

    #     print(f"pv SACIM: {_dc.pv_matmul_SACIM_latency_stat:.12f} s")
    #     print(f"pv baseline: {_dc.pv_matmul_baseline_latency_stat:.12f} s")
    #     print(f"pv speedup: {_dc.pv_matmul_baseline_latency_stat/_dc.pv_matmul_SACIM_latency_stat:.12f}")
    #     print(f"pv TOPS: {Nop_pv_matmul_decode/_dc.pv_matmul_SACIM_latency_stat:.12f}")


    #     print(f"out SACIM: {_dc.o_proj_SACIM_latency_stat:.12f} s")
    #     print(f"out baseline: {_dc.o_proj_baseline_latency_stat:.12f} s")
    #     print(f"out speedup: {_dc.o_proj_baseline_latency_stat/_dc.o_proj_SACIM_latency_stat:.12f}")
    #     print(f"out TOPS: {Nop_o_proj_decode/_dc.o_proj_SACIM_latency_stat:.12f}")
    #     if has_gate:
    #         print(f"gate_proj SACIM: {_dc.gate_proj_SACIM_latency_stat:.12f} s")
    #         print(f"gate_proj baseline: {_dc.gate_proj_baseline_latency_stat:.12f} s")
    #         print(f"gate_proj speedup: {_dc.gate_proj_baseline_latency_stat/_dc.gate_proj_SACIM_latency_stat:.12f}")
    #         print(f"gate_proj TOPS: {Nop_gate_proj_decode/_dc.gate_proj_SACIM_latency_stat:.12f}")

    #     print(f"up_proj SACIM: {_dc.up_proj_SACIM_latency_stat:.12f} s")
    #     print(f"up_proj baseline: {_dc.up_proj_baseline_latency_stat:.12f} s")
    #     print(f"up_proj speedup: {_dc.up_proj_baseline_latency_stat/_dc.up_proj_SACIM_latency_stat:.12f}")
    #     print(f"up_proj TOPS: {Nop_up_proj_decode / _dc.up_proj_SACIM_latency_stat:.12f} TOPS" if _dc.up_proj_SACIM_latency_stat > 0 else "up_proj TOPS: N/A")

    #     print(f"down_proj SACIM: {_dc.down_proj_SACIM_latency_stat:.12f} s")
    #     print(f"down_proj baseline: {_dc.down_proj_baseline_latency_stat:.12f} s")
    #     print(f"down_proj speedup: {_dc.down_proj_baseline_latency_stat/_dc.down_proj_SACIM_latency_stat:.12f}")
    #     print(f"down_proj TOPS: {Nop_down_proj_decode / _dc.down_proj_SACIM_latency_stat:.12f} TOPS" if _dc.down_proj_SACIM_latency_stat > 0 else "down_proj TOPS: N/A")

    #     print(f"-----------------all----------------------")

    #     print(f"all SACIM: {_dc.latency:.12f} s")
    #     print(f"ideal latency: {_dc.ideal_sparsity_latency_stat:.12f} s")
    #     print(f"allflops baseline: {Nop_all_decode:.12f} ")
    #     print(f"TOPS: {Nop_all_decode / _dc.latency:.12f} TOPS" if _dc.latency > 0 else "TOPS: N/A")
    #     print(f"TOPS (ideal): {Nop_all_decode / _dc.ideal_sparsity_latency_stat:.12f} TOPS" if _dc.ideal_sparsity_latency_stat > 0 else "TOPS (ideal): N/A")

    #     print(f"-----------------linear----------------------")

    #     print(f"linear SACIM: {linear_latency:.12f} s")
    #     print(f"allflops baseline: {Nop_all_linear_decode:.12f} ")
    #     print(f"TOPS: {Nop_all_linear_decode /linear_latency:.12f} TOPS" if linear_latency > 0 else "TOPS: N/A")
    #     print(f"TOPS (ideal): {Nop_all_linear_decode / linear_ideal_latency:.12f} TOPS" if linear_ideal_latency > 0 else "TOPS (ideal): N/A")

    #     print(f"-----------------matmul----------------------")

    #     print(f"matmul SACIM: {matmul_latency:.12f} s")
    #     print(f"allflops baseline: {Nop_all_matmul_decode:.12f} ")
    #     print(f"TOPS: {Nop_all_matmul_decode / matmul_latency:.12f} TOPS" if matmul_latency > 0 else "TOPS: N/A")
    #     print(f"TOPS (ideal): {Nop_all_matmul_decode / matmul_ideal_latency:.12f} TOPS" if matmul_ideal_latency > 0 else "TOPS (ideal): N/A")


    # else:
    #     print("未收集到decode激活统计数据")



    _sm = stat_manager_prefill
    destination=args.stats_output_dir or os.path.join(config['quantization']['scale_dir'],'cim_stats')
    os.makedirs(destination,exist_ok=True)
    for phase,manager in [('prefill',stat_manager_prefill),('decode',stat_manager_decode)]:
        manager.export_cim_stats(os.path.join(destination,f'{phase}.json'))
        sparse=manager.SACIM_latency_stat;dense=manager.baseline_latency_stat
        speed=dense/sparse if sparse>0 else None
        print(f"{phase}: counted-bit compute={sparse:.12f}s, dense={dense:.12f}s, mapped speedup={speed}")


    # # --------------------------------------------------------
    # # 导出每层 per-layer 统计到 JSON 文件 (保留原有合并输出不变)
    # # --------------------------------------------------------
    # import json as _json
    # import os as _os

    # _per_layer_report = {
    #     "model_path": getattr(args, "model_path", ""),
    #     "config": getattr(args, "config", ""),
    #     "phase": "prefill",
    #     "layers": {},
    #     "overall": {
    #         "baseline_latency": float(stat_manager_prefill.baseline_latency_stat),
    #         "ideal_sparsity_latency": float(stat_manager_prefill.ideal_sparsity_latency_stat),
    #         "macro_latency": float(stat_manager_prefill.latency),
    #     },
    # }

    # # 逐层注入: key = f"{layer_name}_{layer_idx}", e.g. q_proj_0, q_proj_1, ...
    # for _layer_key, _entry in stat_manager_prefill.per_layer_latency.items():
    #     _layer_info = {
    #         "layer_name": _entry["layer_name"],
    #         "layer_idx": _entry["layer_idx"],
    #         "actual_ideal_speedup": _entry["actual_ideal_speedup"],
    #         "actual_actual_speedup": _entry["actual_actul_speedup"],
    #         "SACIM_latency": _entry["SACIM_latency"],
    #         "baseline_latency": _entry["baseline_latency"],
    #         "ideal_sparsity_latency": _entry.get("ideal_sparsity_latency", 0.0),
    #         "actual_total_bits": _entry.get("actual_total_bits", 0.0),
    #         "actual_1_bits": _entry.get("actual_1_bits", 0.0),
    #         "ideal_1_bits": _entry.get("ideal_1_bits", 0.0),
    #     }
    #     # 逐层 sparsity (如果有)
    #     if "total_elements" in _entry:
    #         _layer_info["total_elements"] = _entry["total_elements"]
    #     if "total_bits" in _entry:
    #         _layer_info["total_bits"] = _entry["total_bits"]
    #         _layer_info["amplitude_zero_bits"] = _entry["amplitude_zero_bits"]
    #         _layer_info["sparse_bit_rate"] = _entry["sparse_bit_rate"]
    #         _layer_info["ideal_speed_up"] = _entry["ideal_speed_up"]
    #         _layer_info["1_bits"] = _entry["1_bits"]
    #     _per_layer_report["layers"][_layer_key] = _layer_info

    # if stat_manager_prefill.total_element_count > 0:
    #     _per_layer_report["global_sparsity"] = {
    #         "total_elements": int(stat_manager_prefill.total_element_count),
    #         "total_bits": int(stat_manager_prefill.total_element_count * 4),
    #         "zero_bit_rate": float(stat_manager_prefill.total_0bit_count / stat_manager_prefill.total_bit_count) if stat_manager_prefill.total_bit_count > 0 else 0.0,
    #     }

    # _json_out = "/per_layer_stats.json"
    # _os.makedirs(_os.path.dirname(_json_out), exist_ok=True)
    # with open(_json_out, "w", encoding="utf-8") as _jf:
    #     _json.dump(_per_layer_report, _jf, ensure_ascii=False, indent=2)
    # print(f"\nPer-layer stats written to {_json_out}")

    # print("\n" + "=" * 80)
    # print("FINAL RESULTS".center(80))
    # print("=" * 80)

    # for name, result in results.items():
    #     print(f"\n{name}:")
    #     print(f"  Time: {result['time']:.2f}s")

    # print("\n" + "=" * 80)

    # del model
    # torch.cuda.empty_cache()

def main():

    """Main entry point."""
    args = parse_args()
    _load_full_model_dependencies()
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Scale directory: {config['quantization']['scale_dir']}")
    
    try:
        # Step 1: Calibration
        model = calibrate(args, config)
        
        # Step 2 & 3: Quantization and Evaluation
        evaluate(args, config, model)
        
        print_header("✓ PIPELINE COMPLETED SUCCESSFULLY")
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

