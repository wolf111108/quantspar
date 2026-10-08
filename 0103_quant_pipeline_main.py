"""
Main Entry Point for OPT Model Quantization Pipeline.

This script provides a complete pipeline for:
1. Calibration: Collect quantization scales
2. Quantization: Apply quantization to the model
3. Evaluation: Evaluate perplexity on test dataset

Usage:
    python scripts/0103_quant_pipeline_main.py --config config/Int8.yaml --model-path /path/to/opt-model
"""

import os
import sys
import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quant import load_config, QuantStatManager
from quant.model_wrapper import wrap_model_by_family
# from quant.bitnet_wrapper import switch_quantization_mode_all  # absent in this snapshot
from quant.qwen_wrapper import switch_quantization_mode_all
from quant.quant_spec import resolve_model_dtype
from others.data import CalibrationDataLoader
# from others.evaluation import evaluate_perplexity, profile_prefill_decode_sparsity  # absent
from perplexity import evaluate_perplexity, profile_prefill_decode_sparsity
from tqdm import tqdm
from quant.quant_linear import QuantizedLinear
from quant.quant_matmul import QuantizedMatMul
from scripts.evaluation_report import export_evaluation_report

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

    parser.add_argument(  #add
        "--eval-flow",  #add
        type=str,  #add
        default="all",  #add
        choices=["all", "ppl", "pd"],  #add
        help=(  #add
            "Which evaluation flow to run: "  #add
            "'all' runs PPL + prefill/decode, "  #add
            "'ppl' runs only PPL full-forward, "  #add
            "'pd' runs only prefill/decode profiling."  #add
        ),  #add
    )  #add
    parser.add_argument(  #add
        "--unit-bit-group-size",  #add
        type=int,  #add
        default=None,  #add
        help="Override unit_sparsity.bit_group_size from config",  #add
    )  #add

    parser.add_argument(  #add
        "--unit-dim-group-size",  #add
        type=int,  #add
        default=None,  #add
        help="Override unit_sparsity.dim_group_size from config",  #add
    )  #add
    parser.add_argument(  #add
        "--stats-output-dir",  #add
        type=str,  #add
        default=None,  #add
        help="Directory to save encoded bit sparsity JSON/CSV and layer coverage",  #add
    )  #add

    parser.add_argument(  #add
        "--run-name",  #add
        type=str,  #add
        default=None,  #add
        help="Run name used for output CSV filenames",  #add
    )  #add

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
        help="Path to pretrained OPT or Qwen2/Qwen2.5 model"
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
    parser.add_argument(  #add
        "--fp-baseline",  #add
        action="store_true",  #add
        help=(  #add
            "Floating-point baseline: skip calibration and run all quantized "  #add
            "modules in raw mode (pure FP forward, no quantization, no stats). "  #add
            "Recommended with --eval-flow ppl."  #add
        ),  #add
    )  #add
    parser.add_argument(  #add
        "--unit-sparsity",  #add
        action=argparse.BooleanOptionalAction,  #add
        default=False,
        help=(  #add
            "Unit sparsity is disabled by default; --unit-sparsity explicitly enables it."
        ),  #add
    )  #add
    parser.add_argument(  #add
        "--results-dir",  #add
        type=str,  #add
        default="log/legacy/results",  #add
        help=(  #add
            "Directory to save the .txt summary report "  #add
            "(default: log/legacy/results)"  #add
        ),  #add
    )  #add
    
    return parser.parse_args()


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def build_wrapped_model(args, config, scale_dir, mode="scale_inspection"):
    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # BitNet 模型：transformers 4.52 已内置支持，不需要 trust_remote_code
    # 本地模型目录缺少 modeling_bitnet.py / configuration_bitnet.py，
    # trust_remote_code=True 反而会尝试走 auto_map 路径而报错。
    model_family = config["quantization"].get("model_family", "opt").lower()
    if model_family not in {"opt", "qwen", "qwen2", "qwen2.5"}:
        raise ValueError("This snapshot supports OPT and Qwen2/Qwen2.5 only")
    if mode is None:
        mode = (
            "scale_inspection"
            if model_family == "bitnet"
            else "scale_inspection"
        )

    if model_family == "bitnet":
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": False,
            "attn_implementation": "eager",
        }
    else:
        model_kwargs = {
            "torch_dtype": (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            ),
            "device_map": "auto",
            "trust_remote_code": True,
        }

        attn_impl = (
            config
            .get("model", {})
            .get("attn_implementation")
        )

        # 35B-A3B 等大模型：device_map="auto" 会把放不下的模块 offload 到
        # CPU（权重留在 meta 设备），导致 wrapper 拷贝权重时报
        # "incompatible tensor type"。这里强制整卡加载（单卡 96G 足够放下
        # 67GiB 权重），避免 offload。
        if model_family in {"qwen3.5_moe", "qwen3_5_moe"}:
            model_kwargs["device_map"] = {"": 0}

        # Qwen3.5 的 qk/pv matmul patch 需要 eager attention
        if attn_impl is None and model_family in {
            "qwen3.5", "qwen3_5", "qwen3.5_moe", "qwen3_5_moe",
        }:
            attn_impl = "eager"

        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

    # Qwen3.5 是 VL 多模态模型（Qwen3_5ForConditionalGeneration /
    # Qwen3_5MoeForConditionalGeneration），需要用 AutoModelForImageTextToText 加载
    if model_family in {
        "qwen3.5", "qwen3_5", "qwen3.5_moe", "qwen3_5_moe",
    }:
        from transformers import AutoModelForImageTextToText as _AutoModelClass
    else:
        _AutoModelClass = AutoModelForCausalLM

    model_kwargs["torch_dtype"] = resolve_model_dtype(config, args.device)
    if args.device == "cpu":
        model_kwargs["device_map"] = None
    model = _AutoModelClass.from_pretrained(
        args.model_path,
        **model_kwargs,
    )

    model.eval()

    stat_manager = QuantStatManager(scale_dir)

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

    # 说明：此处原来会连续加载两次模型（第一次的结果直接被第二次覆盖），
    # 对小模型只是浪费时间，但对 35B 级别模型会因 2x67GB 超出显存而 OOM，
    # 因此只保留一次加载。bitnet 与其他 family 的 build_mode 相同。
    build_mode = "scale_inspection"

    model, stat_manager = build_wrapped_model(
        args=args,
        config=config,
        scale_dir=scale_dir,
        mode=build_mode,
    )
    # if model_family == "bitnet":
    #     print(
    #         "✓ Native BitNet uses its original W1.58A8 forward; "
    #         "project calibration is skipped."
    # )
    #     return model

    # validate_reuse_layers_have_scales(model)
    summary = calibration_action_summary(model)
    print(f"Calibration action summary: {summary}")

    # 浮点基线模式：raw forward 不量化、不需要 scale，直接跳过校准。  #add
    if getattr(args, "fp_baseline", False):  #add
        print(  #add
            "✓ --fp-baseline is set: skipping calibration "  #add
            "(raw FP forward needs no scales)."  #add
        )  #add
        return model  #add

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

def _export_bit_sparsity(args, stat_manager, suffix, phases):
    run_name = args.run_name or os.path.splitext(os.path.basename(args.config))[0]
    output_dir = args.stats_output_dir or args.results_dir
    json_path = os.path.join(output_dir, f"{run_name}_{suffix}_bit_sparsity.json")
    csv_path = os.path.join(output_dir, f"{run_name}_{suffix}_bit_sparsity.csv")
    doc = stat_manager.export_sparsity_stats(json_path, csv_path, phases)
    print(f"Bit sparsity saved to: {json_path} and {csv_path}")
    return doc


def _export_txt_summary(args, config, stat_manager, results, sparsity_summaries=None):  #add
    """把本次 run 的关键结果汇总导出为 .txt 报告。  #add

    输出路径：--results-dir/<run_name>_<时间戳>.txt
    内容：run 元信息、命令行开关状态、PPL 结果、
    encoded bit sparsity、可选 unit sparsity、collected layers 摘要。
    """  #add
    import datetime  #add

    results_dir = getattr(args, "results_dir", None)  #add
    if not results_dir:  #add
        return  #add

    os.makedirs(results_dir, exist_ok=True)  #add

    run_name = args.run_name  #add
    if run_name is None:  #add
        run_name = os.path.splitext(os.path.basename(args.config))[0]  #add

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  #add
    txt_path = os.path.join(results_dir, f"{run_name}_{ts}.txt")  #add

    lines = []  #add
    lines.append("=" * 80)  #add
    lines.append("RUN SUMMARY REPORT")  #add
    lines.append("=" * 80)  #add
    lines.append(  #add
        f"Time          : "  #add
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"  #add
    )  #add
    lines.append(f"Model path    : {args.model_path}")  #add
    lines.append(f"Config        : {args.config}")  #add
    lines.append(f"Run name      : {run_name}")  #add
    lines.append(f"Device        : {args.device}")  #add
    lines.append(f"Eval flow     : {args.eval_flow}")  #add
    lines.append(f"FP baseline   : {args.fp_baseline}")  #add
    lines.append(f"Unit sparsity : {args.unit_sparsity}")  #add
    lines.append("")  #add

    lines.append("BIT SPARSITY (all FP operands: explicit mantissa only)")
    for flow, summary in (sparsity_summaries or {}).items():
        lines.append(f"{flow}: sum zero bits / sum counted bits")
        if summary["outlier_sparsity"]["present"]:
            lines.append("  Outlier scope: masked quantized normal operands; mask zeros included; "
                         "high-precision sidepath operands excluded.")
        for row in summary["phase_operands"]:
            ratio = row["bit_zero_ratio"]
            value = f"{ratio:.4%}" if ratio is not None else "unavailable"
            scope = " (outlier masked)" if row["outlier_masked"] else ""
            lines.append(f"  {row['phase']} / {row['operand']}{scope}: "
                         f"{row['zero_bits']} / {row['bits']} = {value}")
    lines.append("")

    lines.append("-" * 80)  #add
    lines.append("PERPLEXITY")  #add
    lines.append("-" * 80)  #add
    if results:  #add
        for name, result in results.items():  #add
            lines.append(f"{name}:")  #add
            lines.append(f"  Perplexity: {result['perplexity']:.4f}")  #add
            lines.append(f"  Time     : {result['time']:.2f}s")  #add
    else:  #add
        lines.append("No PPL results in this run.")  #add
    lines.append("")  #add

    lines.append("-" * 80)  #add
    lines.append("UNIT SPARSITY (by phase x layer_type)")  #add
    lines.append("-" * 80)  #add
    unit_stats = getattr(stat_manager, "unit_sparsity", {}) or {}  #add
    has_unit_data = any(unit_stats.get(p) for p in unit_stats)  #add
    if not args.unit_sparsity:  #add
        lines.append("Unit sparsity collection disabled (--no-unit-sparsity).")  #add
    elif not has_unit_data:  #add
        lines.append(  #add
            "No unit sparsity collected in this run "  #add
            "(raw/FP-baseline mode collects no stats, or eval-flow=ppl)."  #add
        )  #add
    else:  #add
        lines.append(  #add
            f"{'phase':<14}{'layer_type':<16}{'num_layers':>10}"  #add
            f"{'zero_units':>16}{'total_units':>16}{'ratio':>10}"  #add
        )  #add
        for phase in sorted(unit_stats.keys()):  #add
            by_type = {}  #add
            for layer_key, counter in unit_stats[phase].items():  #add
                layer_type, _ = stat_manager._split_unit_layer_key(layer_key)  #add
                entry = by_type.setdefault(  #add
                    layer_type,  #add
                    {"n": 0, "zero": 0, "total": 0},  #add
                )  #add
                entry["n"] += 1  #add
                entry["zero"] += counter["zero_units"]  #add
                entry["total"] += counter["total_units"]  #add
            for layer_type in sorted(by_type):  #add
                e = by_type[layer_type]  #add
                ratio = e["zero"] / e["total"] if e["total"] else 0.0  #add
                lines.append(  #add
                    f"{phase:<14}{layer_type:<16}{e['n']:>10}"  #add
                    f"{e['zero']:>16}{e['total']:>16}{ratio:>10.4%}"  #add
                )  #add
    lines.append("")  #add

    lines.append("-" * 80)  #add
    lines.append("COLLECTED LAYERS (by phase)")  #add
    lines.append("-" * 80)  #add
    collected = getattr(stat_manager, "collected_layer_names_by_phase", {}) or {}  #add
    if collected:  #add
        for phase in sorted(collected.keys()):  #add
            lines.append(  #add
                f"{phase}: {len(collected[phase])} layers collected"  #add
            )  #add
    else:  #add
        lines.append(  #add
            "No collected layer stats (raw/FP-baseline mode collects none)."  #add
        )  #add
    lines.append("")  #add

    if args.stats_output_dir is not None:  #add
        lines.append("-" * 80)  #add
        lines.append("CSV OUTPUTS")  #add
        lines.append("-" * 80)  #add
        lines.append(f"stats_output_dir: {args.stats_output_dir}")  #add
        lines.append("")  #add

    lines.append("=" * 80)  #add

    with open(txt_path, "w", encoding="utf-8") as f:  #add
        f.write("\n".join(lines) + "\n")  #add

    print(f"\nTXT summary report saved to: {txt_path}")  #add


def evaluate(args, config, model):
    print_header("STEP 2: QUANTIZATION & EVALUATION")

    stat_manager = QuantStatManager(config["quantization"]["scale_dir"],
                                   bit_scope="mantissa", collect_mapping=False,
                                   cache_static_weight_counts=True)


    # from quant.quant_moe_experts import QuantizedMoEExperts  # absent in this snapshot

    for module in model.modules():  #add
        if isinstance(module, (QuantizedLinear, QuantizedMatMul)):
            module._stat_manager = stat_manager  #add

        # Qwen attention wrapper stores stat_manager on attention_module.  #add
        # Only rebind modules that actually own quantized attention matmuls.  #add
        if hasattr(module, "qk_matmul") or hasattr(module, "pv_matmul"):  #add
            if hasattr(module, "stat_manager"):  #add
                module.stat_manager = stat_manager  #add
            if hasattr(module, "qk_matmul"):  #add
                module.qk_matmul._stat_manager = stat_manager  #add
            if hasattr(module, "pv_matmul"):  #add
                module.pv_matmul._stat_manager = stat_manager  #add


    # ------------------------------------------------------------
    # Unit/block sparsity config  #add
    #
    # PPL 阶段先关闭 unit sparsity，避免 PPL full-forward 额外做 unit 统计。 #add
    # 后面 prefill/decode profiling 前会重新打开。 #add
    # ------------------------------------------------------------
    unit_cfg = config.get("unit_sparsity", {})  #add

    unit_bit_group_size = (  #add
        args.unit_bit_group_size  #add
        if args.unit_bit_group_size is not None  #add
        else unit_cfg.get("bit_group_size", 2)  #add
    )  #add

    unit_dim_group_size = (  #add
        args.unit_dim_group_size  #add
        if args.unit_dim_group_size is not None  #add
        else unit_cfg.get("dim_group_size", 2)  #add
    )  #add

    print(  #add
        f"Unit sparsity config: "  #add
        f"enabled={args.unit_sparsity}, "  #add
        f"bit_group_size={unit_bit_group_size}, "  #add
        f"dim_group_size={unit_dim_group_size}"  #add
    )  #add

    stat_manager.configure_unit_sparsity(  #add
        enable=args.unit_sparsity,  #add
        bit_group_size=unit_bit_group_size,  #add
        dim_group_size=unit_dim_group_size,  #add
    )  #add

    model_family = config["quantization"].get("model_family", "opt").lower()

    # 浮点基线：全部量化模块切 raw 模式（纯浮点 forward）。  #add
    if getattr(args, "fp_baseline", False):  #add
        execution_mode = "raw"  #add
    else:  #add
        execution_mode = "quant_forward"  #add

    model = switch_quantization_mode_all(
        model,
        execution_mode,
    )

    print(f"Execution mode: {execution_mode}")

    if execution_mode == "raw":  #add
        print(  #add
            "⚠ FP-baseline (raw) mode: quantized layers run pure FP forward; "  #add
            "no sparsity/unit stats are collected. "  #add
            "Use --eval-flow ppl for a pure PPL baseline."  #add
        )  #add

    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer_kwargs = {}

    if model_family != "bitnet":
        tokenizer_kwargs["trust_remote_code"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nApplying quantization (quant_forward mode)...")
    print("✓ Quantization applied successfully")

    if args.skip_evaluation:
        print("\n✓ Skipping evaluation (--skip-evaluation flag set)")
        return
    
    run_ppl = args.eval_flow in ["all", "ppl"]  #add
    run_pd = args.eval_flow in ["all", "pd"]  #add

    print(f"Evaluation flow: {args.eval_flow}")  #add
    
    results = {}  #add
    sparsity_summaries = {}

    if run_ppl:  #add
        print("\n" + "-" * 80)  #add
        print("STEP 3: PERPLEXITY EVALUATION")  #add
        print("-" * 80)  #add

        for eval_ds in config["evaluation"]["datasets"]:
            dataset_name = eval_ds["name"]
            dataset_cfg = eval_ds.get("config", None)
            split = eval_ds.get("split", "test")

            print(f"\nEvaluating on {dataset_name} ({dataset_cfg}, {split})...")

            start_time = time.time()

            ppl = evaluate_perplexity(
                model,
                tokenizer,
                dataset_name=dataset_name,
                dataset_config=dataset_cfg,
                split=split,
                seq_length=eval_ds.get(
                    "seq_length",
                    config["evaluation"].get("seq_length", 2048),
                ),
                device=args.device,
                text_column=eval_ds.get("text_column", "text"),
                streaming=eval_ds.get("streaming", False),
                max_eval_tokens=eval_ds.get(
                    "max_eval_tokens",
                    config["evaluation"].get("max_eval_tokens", None),
                ),
                min_doc_tokens=eval_ds.get(
                    "min_doc_tokens",
                    config["evaluation"].get("min_doc_tokens", 128),
                ),
            )

            eval_time = time.time() - start_time

            eval_key = eval_ds.get(
                "alias",
                f"{dataset_name}:{dataset_cfg}:{split}",
            )

            results[eval_key] = {
                "perplexity": ppl,
                "time": eval_time,
            }
    else:  #add
        print("\nSkipping PPL evaluation because --eval-flow=pd")  #add
        """
        dataset_name = eval_ds["name"]
        dataset_cfg = eval_ds.get("config", None)
        split = eval_ds.get("split", "test")

        print(f"\nEvaluating on {dataset_name} ({dataset_cfg}, {split})...")

        start_time = time.time()

        ppl = evaluate_perplexity(
            model,
            tokenizer,
            dataset_name=dataset_name,
            dataset_config=dataset_cfg,
            split=split,
            seq_length=eval_ds.get(
                "seq_length",
                config["evaluation"].get("seq_length", 2048),
            ),
            device=args.device,
            text_column=eval_ds.get("text_column", "text"),
            streaming=eval_ds.get("streaming", False),
            max_eval_tokens=eval_ds.get(
                "max_eval_tokens",
                config["evaluation"].get("max_eval_tokens", None),
            ),
            min_doc_tokens=eval_ds.get(
                "min_doc_tokens",
                config["evaluation"].get("min_doc_tokens", 128),
            ),
        )

        eval_time = time.time() - start_time

        eval_key = eval_ds.get(
            "alias",
            f"{dataset_name}:{dataset_cfg}:{split}",
        )

        results[eval_key] = {
            "perplexity": ppl,
            "time": eval_time,
        }
        """
        # ------------------------------------------------------------
    # PPL full-forward sparsity before reset_sparsity()  #add
    # 这里打印的是原始 PPL evaluation 路径 model(batch).logits 下的统计。 #add
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # PPL full-forward sparsity before reset_sparsity()  #add
    # Only run when eval_flow is "all" or "ppl".  #add
    # In eval_flow="pd", PPL is skipped, so full_forward stats are empty.  #add
    # ------------------------------------------------------------
    original_use_cache = getattr(
        model.config,
        "use_cache",
        None,
    )

    if run_ppl:  #add
        print("\n" + "-" * 80)  #add
        print("PPL FULL-FORWARD SPARSITY")  #add
        print("-" * 80)  #add

        model.config.use_cache = False

        stat_manager.print_global_sparsity("PPL FULL-FORWARD OPERAND SPARSITY")
        stat_manager.print_operand_sparsity(["full_forward"])
        sparsity_summaries["full_forward"] = _export_bit_sparsity(
            args, stat_manager, "full_forward", ["full_forward"])

        stat_manager.print_collected_layer_names(  #add
            phase="full_forward",  #add
            title="PPL FULL-FORWARD COLLECTED LAYERS",  #add
        )  #add

        # ------------------------------------------------------------
        # Export PPL full-forward collected layers before reset_sparsity()  #add
        # ------------------------------------------------------------
        if args.stats_output_dir is not None:  #add
            os.makedirs(args.stats_output_dir, exist_ok=True)  #add

            run_name = args.run_name  #add
            if run_name is None:  #add
                run_name = os.path.splitext(os.path.basename(args.config))[0]  #add

            collected_dir = os.path.join(args.stats_output_dir, "collected_layers")  #add
            os.makedirs(collected_dir, exist_ok=True)  #add

            collected_csv = os.path.join(  #add
                collected_dir,  #add
                f"{run_name}_collected_layers.csv",  #add
            )  #add

            collected_summary_csv = os.path.join(  #add
                args.stats_output_dir,  #add
                "collected_layers_summary.csv",  #add
            )  #add

            stat_manager.export_collected_layers_csv(  #add
                collected_csv,  #add
                config_name=run_name,  #add
                model_path=args.model_path,  #add
                phases=["full_forward"],  #add
                append=False,  #add
            )  #add

            stat_manager.export_collected_layers_summary_csv(  #add
                collected_summary_csv,  #add
                config_name=run_name,  #add
                model_path=args.model_path,  #add
                phases=["full_forward"],  #add
                append=True,  #add
            )  #add

            print(f"Collected layer CSV saved to: {collected_csv}")  #add
            print(f"Collected layer summary CSV appended to: {collected_summary_csv}")  #add

    else:  #add
        print("\nSkipping PPL full-forward sparsity because --eval-flow=pd")  #add

    if original_use_cache is not None:
        model.config.use_cache = original_use_cache

    profile_cfg = config.get("prefill_decode_profile", {})  #add

    if run_pd and profile_cfg.get("enabled", True):  #add
        print("\n" + "-" * 80)  #add
        print("STEP 4: PREFILL / DECODE SPARSITY PROFILING")  #add
        print("-" * 80)  #add

        # PPL 的 model(batch).logits 是 full-forward 统计。  #add
        # 这里清空统计器，单独收集 teacher-forced prefill/decode 稀疏度。  #add
        stat_manager.reset_sparsity()  #add

        # ------------------------------------------------------------
        # Preserve the explicit CLI choice; old YAML cannot re-enable units.
        # ------------------------------------------------------------
        stat_manager.configure_unit_sparsity(  #add
            enable=args.unit_sparsity,
            bit_group_size=unit_bit_group_size,  #add
            dim_group_size=unit_dim_group_size,  #add
        )  #add

        profile_ds = config["evaluation"]["datasets"][0]  #add

        profile_prefill_decode_sparsity(  #add
            model=model,  #add
            tokenizer=tokenizer,  #add
            stat_manager=stat_manager,  #add
            dataset_name=profile_cfg.get(  #add
                "dataset_name",  #add
                profile_ds["name"],  #add
            ),  #add
            dataset_config=profile_cfg.get(  #add
                "dataset_config",  #add
                profile_ds.get("config", None),  #add
            ),  #add
            split=profile_cfg.get(  #add
                "split",  #add
                profile_ds.get("split", "test"),  #add
            ),  #add
            prefill_length=profile_cfg.get(  #add
                "prefill_length",  #add
                profile_ds.get(  #add
                    "prefill_length",  #add
                    config["evaluation"].get("seq_length", 2048),  #add
                ),  #add
            ),  #add
            decode_steps=profile_cfg.get(  #add
                "decode_steps",  #add
                128,  #add
            ),  #add
            num_samples=profile_cfg.get(  #add
                "num_samples",  #add
                1,  #add
            ),  #add
            device=args.device,  #add
            text_column=profile_cfg.get(  #add
                "text_column",  #add
                profile_ds.get("text_column", "text"),  #add
            ),  #add
            streaming=profile_cfg.get(  #add
                "streaming",  #add
                profile_ds.get("streaming", False),  #add
            ),  #add
            max_eval_tokens=profile_cfg.get(  #add
                "max_eval_tokens",  #add
                None,  #add
            ),  #add
            min_doc_tokens=profile_cfg.get(  #add
                "min_doc_tokens",  #add
                profile_ds.get(  #add
                    "min_doc_tokens",  #add
                    config["evaluation"].get("min_doc_tokens", 128),  #add
                ),  #add
            ),  #add
        )  #add

        stat_manager.print_prefill_decode_sparsity()  #add
        stat_manager.print_global_sparsity("PREFILL + DECODE TOTAL SPARSITY")  #add
        stat_manager.print_operand_sparsity(["prefill", "decode"])
        sparsity_summaries["prefill_decode"] = _export_bit_sparsity(
            args, stat_manager, "prefill_decode", ["prefill", "decode"])
        if args.unit_sparsity:
            stat_manager.print_unit_sparsity_by_phase()

        # ------------------------------------------------------------
        # Export prefill/decode collected layers after profiling  #add
        # Append to the same per-config collected_layers CSV.  #add
        # ------------------------------------------------------------
        if args.stats_output_dir is not None:  #add
            os.makedirs(args.stats_output_dir, exist_ok=True)  #add

            run_name = args.run_name  #add
            if run_name is None:  #add
                run_name = os.path.splitext(os.path.basename(args.config))[0]  #add

            collected_dir = os.path.join(args.stats_output_dir, "collected_layers")  #add
            os.makedirs(collected_dir, exist_ok=True)  #add

            collected_csv = os.path.join(  #add
                collected_dir,  #add
                f"{run_name}_collected_layers.csv",  #add
            )  #add

            collected_summary_csv = os.path.join(  #add
                args.stats_output_dir,  #add
                "collected_layers_summary.csv",  #add
            )  #add

            stat_manager.export_collected_layers_csv(  #add
                collected_csv,  #add
                config_name=run_name,  #add
                model_path=args.model_path,  #add
                phases=["prefill", "decode"],  #add
                append=True,  #add
            )  #add

            stat_manager.export_collected_layers_summary_csv(  #add
                collected_summary_csv,  #add
                config_name=run_name,  #add
                model_path=args.model_path,  #add
                phases=["prefill", "decode"],  #add
                append=True,  #add
            )  #add

            print(f"Collected layer CSV appended to: {collected_csv}")  #add
            print(f"Collected layer summary CSV appended to: {collected_summary_csv}")  #add

        if args.unit_sparsity and args.stats_output_dir is not None:  #add
            os.makedirs(args.stats_output_dir, exist_ok=True)  #add

            run_name = args.run_name  #add
            if run_name is None:  #add
                run_name = os.path.splitext(os.path.basename(args.config))[0]  #add

            unit_dir = os.path.join(args.stats_output_dir, "unit_sparsity")  #add
            os.makedirs(unit_dir, exist_ok=True)  #add

            unit_csv = os.path.join(  #add
                unit_dir,  #add
                f"{run_name}_unit_sparsity.csv",  #add
            )  #add

            unit_summary_csv = os.path.join(  #add
                args.stats_output_dir,  #add
                "unit_sparsity_summary.csv",  #add
            )  #add

            stat_manager.export_unit_sparsity_csv(  #add
                unit_csv,  #add
                config_name=run_name,  #add
                model_path=args.model_path,  #add
            )  #add

            stat_manager.export_unit_sparsity_summary_csv(  #add
                unit_summary_csv,  #add
                config_name=run_name,  #add
                model_path=args.model_path,  #add
                append=True,  #add
            )  #add

            print(f"Unit sparsity CSV saved to: {unit_csv}")  #add
            print(f"Unit sparsity summary CSV appended to: {unit_summary_csv}")  #add

    elif not run_pd:  #add
        print("\nSkipping prefill/decode profiling because --eval-flow=ppl")  #add
    else:  #add
        print("\nPrefill/decode sparsity profiling disabled by config.")  #add
    """
    print("\n" + "-" * 80)
    print("ZERO ACTIVATION RATIO")
    print("-" * 80)

    if stat_manager.total_element_count > 0:
        print(
            f"全模型量化激活零值比例: "
            f"{stat_manager.total_zero_count / stat_manager.total_element_count:.4%}"
        )
        print(
            f"零值元素: "
            f"{stat_manager.total_zero_count:,} / {stat_manager.total_element_count:,}"
        )

        if stat_manager.total_bit_count > 0:
            print(
                f"全模型稀疏比特比例: "
                f"{stat_manager.total_sparsebit_count / stat_manager.total_bit_count:.4%}"
            )
            print(
                f"全模型Sign Magnitude编码比例: "
                f"{stat_manager.total_amplitude_zero_bits_total / stat_manager.total_bit_count:.4%}"
            )
            print(
                f"零比特比例: "
                f"{stat_manager.total_0bit_count / stat_manager.total_bit_count:.4%}"
            )
    else:
        print("未收集到激活统计数据")
    """

    print("\n" + "=" * 80)
    print("FINAL RESULTS".center(80))
    print("=" * 80)

    if len(results) > 0:  #add
        for name, result in results.items():  #add
            print(f"\n{name}:")  #add
            print(f"  Perplexity: {result['perplexity']:.4f}")  #add
            print(f"  Time: {result['time']:.2f}s")  #add
    else:  #add
        print("\nNo PPL results in this run.")  #add

    print("\n" + "=" * 80)

    # 导出 .txt 汇总报告（--results-dir，默认根目录 results/）  #add
    _export_txt_summary(args, config, stat_manager, results, sparsity_summaries)
    report_path = export_evaluation_report(args, config, results, sparsity_summaries)
    print(f"JSON evaluation report saved to: {report_path}")

    del model
    torch.cuda.empty_cache()

def main():
    """Main entry point."""
    args = parse_args()
    
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

