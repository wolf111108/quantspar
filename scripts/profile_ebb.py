"""Local Qwen FP8/W4 collection; exports MulTCIM-inspired compute bounds."""
import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path

import torch
from quant import load_config, QuantStatManager, QuantizedLinear, QuantizedMatMul


LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# The Qwen2 attention patch relies on APIs that are stable in this window:
#   rotary_emb(x, seq_len=...)  ->  changed to rotary_emb(x, position_ids) in 4.45
#   Cache.get_usable_length()   ->  removed in 4.45
# 4.43.1 is the validated pin; 4.40.x is exercised by the test suite; 4.44 is
# signature-compatible but not executed here.
TRANSFORMERS_MIN = (4, 40)
TRANSFORMERS_MAX_EXCLUSIVE = (4, 45)


def transformers_version_tuple(raw):
    parts = []
    for chunk in raw.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/qwen2_14b_ebb_f8i4.yaml"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scale-dir", type=Path, help="Override scale directory, e.g. isolate smoke calibration")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--calibration-samples", type=int)
    parser.add_argument("--calibration-length", type=int,
                        help="Override calibration tokens; default caps YAML length at effective prefill")
    parser.add_argument("--prefill-length", type=int)
    parser.add_argument("--decode-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--token-file", type=Path,
                        help="torch.save LongTensor [tokens], or dict with input_ids; no dataset needed")
    parser.add_argument("--greedy-decode", action="store_true",
                        help="generate tokens; default uses teacher-forced tokens from the document")
    return parser.parse_args(argv)


def resolve_workload(config, args):
    """Keep calibration within the profiling memory budget unless explicitly overridden."""
    pd = config["prefill_decode_profile"]
    prefill = args.prefill_length if args.prefill_length is not None else pd["prefill_length"]
    decode = args.decode_steps if args.decode_steps is not None else pd["decode_steps"]
    if prefill <= 0 or decode < 0:
        raise ValueError("prefill must be positive and decode must be nonnegative")
    cc = config["calibration"]
    length = (args.calibration_length if args.calibration_length is not None
              else min(cc.get("seq_length", prefill), prefill))
    samples = args.calibration_samples if args.calibration_samples is not None else cc.get("num_samples", 64)
    if args.checkpoint_every <= 0 or length <= 0 or samples <= 0:
        raise ValueError("checkpoint-every, calibration-length and calibration-samples must be positive")
    pd.update(prefill_length=prefill, decode_steps=decode)
    cc.update(seq_length=length, num_samples=samples, batch_size=1,
              min_text_tokens=min(cc.get("min_text_tokens", length), length))
    return prefill, decode


def validate_config(config):
    if not config.get("ebb", {}).get("enabled", True):
        raise ValueError("profile_ebb requires the EBB backend to be enabled")
    quant = config["quantization"]
    if quant.get("mixed_precision", False):
        raise ValueError("EBB profiling requires mixed_precision=false")
    if not quant.get("quantize_matmul", False):
        raise ValueError("EBB profiling requires quantize_matmul=true")
    from quant.quant_spec import parse_quant_spec
    for name in LINEARS + ("qk_matmul", "pv_matmul"):
        layer = quant[name]
        if layer.get("outlier_ratio", quant.get("outlier_ratio", 0)) != 0:
            raise ValueError(f"{name}: outlier sidepath is not modeled; set outlier_ratio=0")
        if name in LINEARS:
            if parse_quant_spec(layer["a_bit"]).fmt != "e4m3" or parse_quant_spec(layer["w_bit"]).bits != 4:
                raise ValueError(f"{name}: expected E4M3 activation and INT4 weight")
        elif any(parse_quant_spec(layer[key]).fmt != "e4m3" for key in ("A_bit", "B_bit")):
            raise ValueError(f"{name}: expected E4M3 operands")
    if not quant.get("kv_cache", {}).get("fp8_static", False):
        raise ValueError("Set kv_cache.fp8_static=true for FP8 cache write semantics")
    from quant.ebb import EBBConfig
    EBBConfig.from_dict(config["ebb"])


def bind_manager(model, manager):
    for module in model.modules():
        if isinstance(module, (QuantizedLinear, QuantizedMatMul)):
            module._stat_manager = manager
        if hasattr(module, "qk_matmul") and hasattr(module, "pv_matmul"):
            module.stat_manager = manager


def profile_tokens(args, config, tokenizer, prefill, decode):
    needed = prefill if args.greedy_decode else prefill + decode
    if args.token_file:
        data = torch.load(args.token_file, map_location="cpu", weights_only=True)
        data = data["input_ids"] if isinstance(data, dict) else data
        if data.dtype != torch.long or data.ndim not in (1, 2):
            raise ValueError("token-file must contain torch.long token IDs")
        if data.ndim == 2 and data.shape[0] != 1:
            raise ValueError("EBB profiling uses batch size 1")
        ids = data.reshape(-1)
    else:
        from datasets import load_dataset
        pd = config["prefill_decode_profile"]
        dataset = load_dataset(pd.get("dataset", "HuggingFaceFW/fineweb"),
                               pd.get("dataset_config", "sample-10BT"),
                               split=pd.get("split", "train"),
                               streaming=pd.get("streaming", True))
        offset = pd.get("sample_offset", 0)
        ids = None
        for sample in dataset:
            candidate = tokenizer(sample[pd.get("text_column", "text")],
                                  add_special_tokens=False, return_tensors="pt").input_ids[0]
            start = pd.get("start_offset", 0)
            if candidate.numel() >= start + needed:
                if offset:
                    offset -= 1
                    continue
                ids = candidate[start:]
                break
        if ids is None:
            raise RuntimeError(f"No document with at least {needed} tokens")
    if ids.numel() < needed:
        raise ValueError(f"Need {needed} tokens; received {ids.numel()}")
    return ids[:needed]


def main(argv=None):
    args = parse_args(argv)
    config = deepcopy(load_config(args.config))
    if args.scale_dir is not None:
        config["quantization"]["scale_dir"] = str(args.scale_dir)
    validate_config(config)
    prefill, decode = resolve_workload(config, args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu for a small-model smoke run")
    installed = importlib.metadata.version("transformers")
    if not TRANSFORMERS_MIN <= transformers_version_tuple(installed) < TRANSFORMERS_MAX_EXCLUSIVE:
        raise RuntimeError(
            f"transformers {installed} is outside the supported range "
            f">=4.{TRANSFORMERS_MIN[1]},<4.{TRANSFORMERS_MAX_EXCLUSIVE[1]} "
            "(4.43.1 is the validated version; 4.45 moved RoPE to position_ids and dropped "
            "Cache.get_usable_length). Install requirements-model.txt")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from quant.model_wrapper import wrap_model_by_family
    from quant.qwen_wrapper import switch_quantization_mode_all
    from others.data import CalibrationDataLoader
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokens = profile_tokens(args, config, tokenizer, prefill, decode)
    dtype = torch.float32 if args.device == "cpu" else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype,
                device_map="auto" if args.device == "cuda" else None, attn_implementation="eager")
    if model.config.model_type != "qwen2":
        raise ValueError("profile_ebb currently supports Qwen2/Qwen2.5")
    quant = config["quantization"]
    if args.skip_calibration:
        quant["calibration_policy"] = dict(default="reuse")
    scale_dir = quant["scale_dir"]
    calibration_manager = QuantStatManager(scale_dir)
    model = wrap_model_by_family(model, quant, mode="scale_inspection",
                                 stat_manager=calibration_manager).eval()
    modules = [m for m in model.modules() if isinstance(m, (QuantizedLinear, QuantizedMatMul))]
    expected = len(model.model.layers) * 9
    if len(modules) != expected:
        raise RuntimeError(f"Expected {expected} quantized operators, found {len(modules)}")
    actions = [m._resolve_calibration_action() for m in modules]
    cc = config["calibration"]
    calibration_batches = 0
    if "recalibrate" in actions:
        loader = CalibrationDataLoader(cc["dataset"], cc.get("dataset_config"), cc.get("split", "train"),
                    args.model_path, seq_length=cc["seq_length"], batch_size=1,
                    num_samples=cc["num_samples"], seed=cc.get("seed", 23), text_column=cc.get("text_column", "text"),
                    streaming=cc.get("streaming", True), min_text_tokens=cc["min_text_tokens"])
        from tqdm import tqdm
        with torch.no_grad():
            for batch in tqdm(loader, desc="Calibrating"):
                # LM head is outside this collection's scope; avoid materializing
                # a [sequence_length,vocabulary] logits tensor during calibration/prefill.
                model.model(input_ids=batch["input_ids"].to(args.device),
                            attention_mask=batch["attention_mask"].to(args.device), use_cache=False)
                calibration_batches += 1
        calibration_manager.save_all_scales()
    switch_quantization_mode_all(model, "quant_forward")
    ebb = dict(config["ebb"], trace_path=str(args.output_dir / "ebb_trace.jsonl"))
    manager = QuantStatManager(scale_dir, ebb_config=ebb)
    bind_manager(model, manager)
    workload = dict(model_type=model.config.model_type, layers=len(model.model.layers),
                    d_model=model.config.hidden_size, ffn_dim=model.config.intermediate_size,
                    q_heads=model.config.num_attention_heads, kv_heads=model.config.num_key_value_heads,
                    head_dim=model.config.hidden_size // model.config.num_attention_heads,
                    batch_size=1, prefill_length=prefill, decode_steps=decode,
                    decode_mode="greedy" if args.greedy_decode else "teacher_forced",
                    transport=dict(linear_weight="packed_int4", kv_cache="fp8"),
                    torch_version=torch.__version__, transformers_version=importlib.metadata.version("transformers"),
                    calibration=dict(cc, performed=calibration_batches > 0,
                                     completed_batches=calibration_batches,
                                     operators_recalibrated=actions.count("recalibrate"),
                                     operators_reused=actions.count("reuse")),
                    quantization=quant)
    workload.update(status="running", completed_decode_steps=0)
    (args.output_dir / "run_config.json").write_text(json.dumps(
        dict(workload=workload, ebb=config["ebb"]), indent=2) + "\n")
    try:
        with torch.no_grad():
            manager.set_phase("prefill")
            manager.set_step(0, query_length=prefill)
            output = model.model(tokens[:prefill].unsqueeze(0).to(args.device),
                                 past_key_values=DynamicCache(), use_cache=True)
            past = output.past_key_values
            if manager.ebb_stats.phases["prefill"]["calls"] != expected:
                raise RuntimeError("Incomplete prefill operator coverage")
            manager.export_ebb_stats(args.output_dir / "ebb_summary.json", workload)
            manager.set_phase("decode")
            from tqdm import trange
            for step in trange(decode, desc="Collecting EBB decode"):
                if args.greedy_decode:
                    token = model.lm_head(output.last_hidden_state[:, -1]).argmax(-1).reshape(1, 1)
                else:
                    token = tokens[prefill + step].reshape(1, 1).to(args.device)
                manager.set_step(step, cache_length_before=prefill + step, query_length=1)
                output = model.model(token, past_key_values=past, use_cache=True)
                past = output.past_key_values
                if past.get_seq_length() != prefill + step + 1:
                    raise RuntimeError("KV cache did not grow by one token")
                workload["completed_decode_steps"] = step + 1
                if (step + 1) % args.checkpoint_every == 0:
                    manager.export_ebb_stats(args.output_dir / "ebb_summary.json", workload)
            if decode and manager.ebb_stats.phases["decode"]["calls"] != expected * decode:
                raise RuntimeError("Incomplete decode operator coverage")
            workload["status"] = "complete"
            doc = manager.export_ebb_stats(args.output_dir / "ebb_summary.json", workload)
    except BaseException:
        workload["status"] = "interrupted"
        manager.export_ebb_stats(args.output_dir / "ebb_summary.json", workload)
        raise
    finally:
        manager.close()
    for phase, result in doc["phases"].items():
        seconds = result["compute_seconds"]
        print(f"{phase}: conditional compute bounds "
              f"{seconds['ideal_balanced']:.6f}..{seconds['leading_zero']:.6f} s; "
              f"word coverage complete={result['configured_word_coverage_complete']}")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
