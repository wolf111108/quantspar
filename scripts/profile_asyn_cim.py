"""Qwen FP8/W4 Asyn-CIM collection and strict LLMCompass manifest export."""
import argparse
from collections import Counter
from copy import deepcopy
import importlib.metadata
import json
import math
from pathlib import Path

import torch
from quant import load_config, QuantStatManager, QuantizedLinear, QuantizedMatMul
from .profile_ebb import (LINEARS, bind_manager, profile_tokens, resolve_workload,
                          source_git_commit, transformers_version_tuple,
                          TRANSFORMERS_MIN, TRANSFORMERS_MAX_EXCLUSIVE)

OPERATORS = LINEARS + ("qk_matmul", "pv_matmul")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config/qwen2_14b_asyn_cim_f8i4_256_32.yaml"))
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--scale-dir", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--token-file", type=Path)
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--calibration-samples", type=int)
    p.add_argument("--calibration-length", type=int)
    p.add_argument("--prefill-length", type=int)
    p.add_argument("--decode-steps", type=int)
    p.add_argument("--checkpoint-every", type=int, default=32)
    p.add_argument("--greedy-decode", action="store_true")
    return p.parse_args(argv)


def validate_config(config):
    from quant.quant_spec import parse_quant_spec
    q = config["quantization"]
    if not q.get("enabled") or not q.get("quantize_linear") or not q.get("quantize_matmul") or q.get("mixed_precision", False):
        raise ValueError("Enable Linear/MatMul quantization with mixed_precision=false")
    if q.get("lm_head", {}).get("enabled", False):
        raise ValueError("LM head is outside the nine-GEMM collection scope")
    for name in OPERATORS:
        s = q[name]
        if s.get("outlier_ratio", q.get("outlier_ratio", 0)) != 0 or s.get("mixed_precision", q.get("mixed_precision", False)):
            raise ValueError(f"{name}: outlier and mixed-precision sidepaths are unsupported")
        if name in LINEARS:
            a, w = parse_quant_spec(s["a_bit"]), parse_quant_spec(s["w_bit"])
            if a.fmt != "e4m3" or w.kind != "int" or w.bits != 4:
                raise ValueError(f"{name}: expected E4M3 activation and INT4 weight")
        elif any(parse_quant_spec(s[k]).fmt != "e4m3" for k in ("A_bit", "B_bit")):
            raise ValueError(f"{name}: both operands must be E4M3")
    if not q.get("kv_cache", {}).get("fp8_static", False):
        raise ValueError("Enable kv_cache.fp8_static")
    c = config["asyn_cim"]
    if any(type(c[k]) is not int or c[k] <= 0 for k in ("height", "width", "banks", "macros")):
        raise ValueError("CIM geometry must contain positive integers")
    if c.get("bit_scope", "mantissa") not in ("mantissa", "sign_mantissa"):
        raise ValueError("Bridge supports explicit mantissa or sign_mantissa compute; storage bits are not compute cycles")
    for k in ("frequency_hz", "cycles_per_effective_bit"):
        if not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError(f"Invalid {k}")


def validate_coverage(manager, layers, decode_steps):
    expected = {(phase, name, layer): (1 if phase == "prefill" else decode_steps)
                for phase in ("prefill", "decode") for name in OPERATORS for layer in range(layers)}
    actual = Counter((r["phase"], r["layer_name"], r["layer_idx"]) for r in manager.cim_records)
    if dict(actual) != expected:
        raise RuntimeError("Incomplete Asyn-CIM layer/operator/phase coverage")
    contexts = Counter((r["context"].get("step"), r["layer_name"], r["layer_idx"])
                       for r in manager.cim_records if r["phase"] == "decode")
    if dict(contexts) != {(step, name, layer): 1 for step in range(decode_steps)
                         for name in OPERATORS for layer in range(layers)}:
        raise RuntimeError("Incomplete per-step Asyn-CIM coverage")


def phase_summary(records):
    result = {}
    for phase in ("prefill", "decode"):
        selected = [r for r in records if r["phase"] == phase]
        def aggregate(items):
            dense = sum(r["dense_steps"] for r in items)
            sparse = sum(r["sparse_steps"] for r in items)
            return dict(calls=len(items), dense_steps=dense, sparse_steps=sparse,
                        mapped_compute_speedup=dense/sparse if sparse else None)
        result[phase] = aggregate(selected)
        result[phase]["operators"] = {name: aggregate([r for r in selected if r["layer_name"] == name])
                                       for name in OPERATORS}
    return result


def main(argv=None):
    args = parse_args(argv)
    config = deepcopy(load_config(args.config))
    validate_config(config)
    prefill, decode = resolve_workload(config, args)
    if decode < 1:
        raise ValueError("At least one decode forward is required for the two-phase bridge")
    source = source_git_commit()
    if source is None:
        raise RuntimeError("Run from a Git checkout to record source_commit")
    installed = importlib.metadata.version("transformers")
    if not TRANSFORMERS_MIN <= transformers_version_tuple(installed) < TRANSFORMERS_MAX_EXCLUSIVE:
        raise RuntimeError("Install requirements-model.txt: transformers >=4.40,<4.45; validated pin 4.43.1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use CPU only for small-model verification")
    if args.scale_dir:
        config["quantization"]["scale_dir"] = str(args.scale_dir)
    q = config["quantization"]
    if args.skip_calibration:
        q["calibration_policy"] = dict(default="reuse")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from quant.model_wrapper import wrap_model_by_family
    from quant.qwen_wrapper import switch_quantization_mode_all
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokens = profile_tokens(args, config, tokenizer, prefill, decode)
    dtype = torch.float32 if args.device == "cpu" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype,
             device_map="auto" if args.device == "cuda" else None, attn_implementation="eager")
    if model.config.model_type != "qwen2":
        raise ValueError("Only Qwen2/Qwen2.5 is supported")
    calibration = QuantStatManager(q["scale_dir"])
    model = wrap_model_by_family(model, q, mode="scale_inspection", stat_manager=calibration).eval()
    layers = len(model.model.layers)
    modules = [m for m in model.modules() if isinstance(m, (QuantizedLinear, QuantizedMatMul))]
    if len(modules) != layers * 9:
        raise RuntimeError("Expected nine quantized GEMMs per layer")
    actions = [m._resolve_calibration_action() for m in modules]
    batches = 0
    cc = config["calibration"]
    if "recalibrate" in actions:
        from others.data import CalibrationDataLoader
        from tqdm import tqdm
        loader = CalibrationDataLoader(cc["dataset"], cc.get("dataset_config"), cc.get("split", "train"),
            args.model_path, seq_length=cc["seq_length"], batch_size=1, num_samples=cc["num_samples"],
            seed=cc.get("seed", 23), text_column=cc.get("text_column", "text"),
            streaming=cc.get("streaming", True), min_text_tokens=cc["min_text_tokens"])
        with torch.no_grad():
            for batch in tqdm(loader, desc="Calibrating"):
                model.model(input_ids=batch["input_ids"].to(args.device),
                            attention_mask=batch["attention_mask"].to(args.device), use_cache=False)
                batches += 1
        if not batches:
            raise RuntimeError("Calibration produced no batches")
        calibration.save_all_scales()
    switch_quantization_mode_all(model, "quant_forward")
    c = config["asyn_cim"]
    manager = QuantStatManager(q["scale_dir"], nmacro=c["macros"], h=c["height"], w=c["width"],
        banks=c["banks"], bit_scope=c.get("bit_scope", "mantissa"), as_l=1,
        cycles_per_effective_bit=c["cycles_per_effective_bit"], cache_static_weight_counts=True)
    manager.clock_freq_hz = c["frequency_hz"]
    manager.configure_unit_sparsity(enable=False)
    bind_manager(model, manager)
    workload = dict(d_model=model.config.hidden_size, ffn_dim=model.config.intermediate_size,
        q_heads=model.config.num_attention_heads, kv_heads=model.config.num_key_value_heads,
        batch_size=1, shared_kv_gqa=q.get("shared_kv_gqa", True), layers=layers,
        prefill_lengths=[prefill], decode_cache_lengths=list(range(prefill, prefill+decode)),
        prefill_length=prefill, decode_steps=decode, completed_decode_steps=0, status="running",
        decode_mode="greedy" if args.greedy_decode else "teacher_forced", source_commit=source,
        torch_version=torch.__version__, transformers_version=installed,
        calibration=dict(performed=batches > 0, completed_batches=batches,
                         operators_reused=actions.count("reuse"), operators_recalibrated=actions.count("recalibrate"),
                         seq_length=cc["seq_length"], num_samples=cc["num_samples"]),
        quantization=q, token_convention="N decode forwards after prefill; LLMCompass output-length=N+1")
    def export():
        doc = manager.export_cim_stats(args.output_dir/"asyn_cim_summary.json")
        doc.update(workload=deepcopy(workload), phases=phase_summary(manager.cim_records))
        (args.output_dir/"asyn_cim_summary.json").write_text(json.dumps(doc, indent=2, allow_nan=False)+"\n")
        (args.output_dir/"run_config.json").write_text(json.dumps(dict(workload=workload, asyn_cim=c), indent=2, allow_nan=False)+"\n")
    try:
        export()
        with torch.no_grad():
            manager.set_phase("prefill"); manager.set_step(0, query_length=prefill)
            output = model.model(tokens[:prefill].unsqueeze(0).to(args.device), past_key_values=DynamicCache(), use_cache=True)
            past = output.past_key_values
            if len(manager.cim_records) != layers * 9:
                raise RuntimeError("Incomplete prefill coverage")
            export()
            manager.set_phase("decode")
            from tqdm import trange
            for step in trange(decode, desc="Collecting ASYN-CIM decode"):
                token = (model.lm_head(output.last_hidden_state[:, -1]).argmax(-1).reshape(1,1)
                         if args.greedy_decode else tokens[prefill+step].reshape(1,1).to(args.device))
                manager.set_step(step, cache_length_before=prefill+step, query_length=1)
                output = model.model(token, past_key_values=past, use_cache=True)
                past = output.past_key_values
                if past.get_seq_length() != prefill+step+1:
                    raise RuntimeError("KV cache did not grow by one token")
                workload["completed_decode_steps"] = step+1
                if (step+1) % args.checkpoint_every == 0:
                    export()
        validate_coverage(manager, layers, decode)
        workload["status"] = "complete"
        manager.export_llmcompass_manifest(args.output_dir/"llmcompass_speedups.json", workload, source)
        export()
    except BaseException:
        workload["status"] = "interrupted"
        (args.output_dir/"llmcompass_speedups.json").unlink(missing_ok=True)
        export()
        raise
    finally:
        manager.close()
    for phase, row in phase_summary(manager.cim_records).items():
        print(f"{phase}: calls={row['calls']}; mapped compute speedup={row['mapped_compute_speedup']}; bit_scope={manager.bit_scope}")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
