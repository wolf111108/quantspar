from .opt_wrapper import wrap_opt_model
from .qwen_wrapper import wrap_qwen_model


def wrap_model_by_family(
    model,
    quant_config,
    mode="scale_inspection",
    stat_manager=None,
):
    family = quant_config.get("model_family", "opt").lower()

    if family == "opt":
        return wrap_opt_model(
            model,
            quant_config,
            mode=mode,
            stat_manager=stat_manager,
        )

    if family == "bitnet":
        from .bitnet_wrapper import wrap_bitnet_model

        return wrap_bitnet_model(
            model,
            quant_config,
            mode=mode,
            stat_manager=stat_manager,
        )

    if family in {"qwen", "qwen2", "qwen2.5"}:
        return wrap_qwen_model(
            model,
            quant_config,
            mode=mode,
            stat_manager=stat_manager,
        )

    if family in {"qwen3.5", "qwen3_5", "qwen3.5_moe", "qwen3_5_moe"}:
        from .qwen35_wrapper import wrap_qwen35_model

        return wrap_qwen35_model(
            model,
            quant_config,
            mode=mode,
            stat_manager=stat_manager,
        )

    raise ValueError(f"Unsupported model_family: {family}")