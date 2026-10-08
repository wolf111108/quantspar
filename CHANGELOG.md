# 修改记录

每次提交在同一提交中增加一条，记录目的、内容、验证和限制。当前提交用与 commit message 一致的标题标识；提交前不填自身 SHA。规则见 [AGENTS.md](AGENTS.md)。

## 2026-10-08 — Fix evaluation report flow key mismatch failing matrix validation

- 目的：修复重跑矩阵把成功流水线误判为 failed 的问题。真实运行中流水线退出码 0、PPL 正常、stats 文件齐全，但 runner 校验 `evaluation.json` 时报 "Missing or incompatible full_forward bit statistics"。
- 内容：`0103_quant_pipeline_main.py` 将 PPL full-forward 快照写入 `sparsity_summaries` 的键从 `"ppl"` 改为 `"full_forward"`，与 stats 文件名后缀、`scripts/evaluation_report.py` 的 artifacts 指纹查找及 `scripts/run_sparsity_ppl_matrix.py` 的 `read_report` 校验三方对齐。旧键还导致 `artifacts.ppl` 指纹查找落空（找 `..._ppl_bit_sparsity.*` 不存在），即使 runner 认键也无法通过 SHA256 校验。同步更新 `tests/test_bit_sparsity_pipeline.py` 中 TXT 摘要的键名断言。单测原先用合成报告（直接写 `full_forward` 键）故未覆盖真实流水线的键名。
- 验证：smtqt 环境（PyTorch 2.6.0+cpu、Transformers 4.43.1）下 `tests/test_sparsity_ppl_matrix.py` 与 `tests/test_bit_sparsity_pipeline.py` 共 18 项测试加 31 子测试全部通过；py_compile 通过。base 环境因 torch 过旧无 `float8_e4m3fn` 与本次修改无关。
- 限制与旧结果影响：`source_digest` 含主入口文件，fingerprint 变化使 `--resume` 无法复用修复前的 attempt；`outputs/sparsity_ppl_rerun` 下修复前的 attempt（含 PPL=14.4414 的 opt_1.3b_bf16_bf16 等）报告键名错误且内嵌旧指纹，不进入新 summary。实测重跑 4 个 opt-1.3b 任务 PPL 与旧 attempt bit 级一致（14.441374778747559 / 14.9683837890625 / 131.02838134765625 / 277.5075988769531），可复现。运行注意事项：矩阵 fingerprint 含 `runtime_versions()`（torch/transformers 版本），必须始终用同一环境（smtqt）启动，`${PYTHON:-python}` 在错误 conda 环境下会因缺 transformers qwen2 模块秒失败，且会把 manifest 指纹覆盖为错误环境的值导致 resume 失效；误跑后需以正确环境重算指纹并恢复 completed 状态。



- 目的：按截图中 GPT-2 以外的三个模型、五种 A/W 格式创建可本地重跑的配置与脚本，重新采集比特稀疏度和 PPL，包括旧 NaN 项。
- 内容：新增 OPT-1.3B、OPT-6.7B、Qwen2.5-7B × BF16/BF16、E4M3FN/E4M3FN、INT8/INT8、INT8/INT4、E4M3FN/INT4 共 15 份 YAML；全部显式 BF16 加载、eager、关闭 mixed precision/unit，使用 FineWeb sample-10BT/train、64 校准样本/seed23、统一 65536 输入 token PPL 预算，OPT 长度 2048、Qwen 长度 8192。量化组 Linear outlier=0.0001，BF16 与 QK/PV 为零；K/V 跟随 A 格式。新增逐组子进程 Python/Shell 脚本、独立 attempt/scales、dry-run/子集/resume/outlier 覆盖、失败继续与退出码；主入口新增完整数值 PPL/阶段快照 JSON，脚本导出 summary JSON/CSV 与操作数原始计数 CSV，比例按分子/分母加权，NaN 保留 null/status。更新 README 和专门运行说明。
- 验证：新增 9 项无模型依赖 unittest 全部通过，覆盖 15 配置实际 wrapper 层名/格式/上下文/数据预算与 scale 隔离、dry-run 无模型调用/无文件写入、完整 PPL 数值、加权计数与非有限 JSON、失败后继续/空值、resume 指纹变化及文件缺失/错误计数/详细文件 SHA256 损坏重跑、中断状态、PPL-only 和跨工作目录 Shell 入口。Python compileall、Shell bash -n 与 diff 检查通过。
- 限制与旧结果影响：只使用合成报告和调度回调验证，没有在这里运行完整 checkpoint、FineWeb、CUDA 或真实 PPL，当前环境无 PyTorch/Transformers。FP 显式尾数、INT 补码、mask 零计入及 protected codes 排除口径保持不变；截图的指数对齐尾数脚注与本次不同，旧上传配置的 FP16/BF16、零 outlier、Qwen token 预算差异均在说明中明确，新结果不可直接混称旧设置。resume 校验路径/配置/代码/依赖，不计算 checkpoint 权重哈希；数值仿真不含原生低位 kernel 或硬件时延。既有 FP16 配置、默认 14B 配置与硬件 collector 未改。

## 2026-10-08 — Add QK/PV outlier ratio override to sparsity PPL matrix

- 目的：支持对 PPL 恶化组做加严 outlier 旁路对比实验——原 runner 的 `--outlier-ratio` 只覆盖 Linear，QK/PV MatMul 固定使用模板值，无法按用户要求同时把两者设为 0.0002。
- 内容：`scripts/run_sparsity_ppl_matrix.py` 新增 `--qk-pv-outlier-ratio` 参数（复用 `outlier_ratio` 校验，仅覆盖 `qk_matmul`/`pv_matmul`，不触碰 Linear），覆盖值进入任务指纹；manifest entry 与 `summary.csv` 新增 `qk_pv_outlier_ratio` 列，与 `linear_outlier_ratio` 并列记录有效设置。`docs/sparsity_ppl_rerun.md` 补充该参数说明。
- 验证：smtqt 环境下 `tests/test_sparsity_ppl_matrix.py`（新增覆盖测试，验证配置覆盖、指纹变化、manifest 只保留最新 attempt、summary 两列取值）与 `tests/test_bit_sparsity_pipeline.py` 全部通过。
- 限制与旧结果影响：默认不传该参数时行为与旧版完全一致（QK/PV 保持模板值），旧结果不受影响。加严实验使用独立输出目录（Linear 与 QK/PV 均 0.0002，9 组：opt_1.3b×3、opt_6.7b×2、qwen2.5_7b×4），与主矩阵（0.0001/0）不可混表，对比时需注明两组旁路设置不同。
