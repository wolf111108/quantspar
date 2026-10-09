# 修改记录

每次提交在同一提交中增加一条，记录目的、内容、验证和限制。当前提交用与 commit message 一致的标题标识；提交前不填自身 SHA。规则见 [AGENTS.md](AGENTS.md)。

## 2026-10-09 — Union activation and weight outlier channels

- 目的：按相同 outlier ratio 独立寻找乘法两侧的大幅值归约通道，并在两侧使用同一个并集 mask。
- 内容：Linear 分别从激活和权重的输入列选通道；QK/PV MatMul 分别从 A 的末维和 B 的倒数第二维选通道。两侧各选 `max(1, int(K * ratio))`，取并集后共同分离 normal/protected；Linear 的普通校准和 BitNet 校准也使用该 mask。删除按元素权重 mask 函数与不再生效的 outliermore 开关，更新相关断言和统计说明。输出通道自身的 mask 仍按输出选择。
- 验证：按用户要求，本次未运行测试或模型实验；仅核对远端最新代码与改动范围。
- 限制与旧结果影响：并集最多包含两侧各自选中的通道数，通常多于原先仅由激活选中的通道；权重按元素 top-k 的保护取消。normal 张量的校准范围、量化结果、mask 人为零及 PPL 均可能改变，旧 scales 和稀疏度/PPL 结果应重新校准和采集。高精度 protected 部分仍不计入比特稀疏度。

## 2026-10-09 — Aggregate calibration scales and couple outlier ratio overrides

- 目的：修复量化校准只保留最后一个 batch 的 A/W/O scale，并避免矩阵脚本单独覆盖 Linear 旁路时误以为 QK/PV 也跟随。
- 内容：普通 Linear 和 QK/PV MatMul 校准按 batch 对各自标量或逐输出通道 scale 取最大值；`--outlier-ratio` 默认同时覆盖 Linear 与 QK/PV，显式 `--qk-pv-outlier-ratio` 优先。翻倍驱动器原本已同时覆盖两者，保持其调度语义。更新回归测试与实验说明。
- 验证：矩阵参数传递的无模型回归测试及 Python 编译检查；当前执行环境缺少 PyTorch，新增校准数值测试留待用户的 PyTorch 环境运行，未运行真实 Qwen/FineWeb PPL。
- 限制与旧结果影响：默认模板的 QK/PV 仍为 0；显式传入 `--outlier-ratio` 的实验从现在起改变 QK/PV 行为，旧输出不能混用。普通 Linear/QK/PV 模式的旧 scale 仅对应最后 batch，必须使用新目录重新校准并重跑 PPL/稀疏结果；跨批最大值不能保证 Qwen PPL 达标。

## 2026-10-09 — Promote outliermore to a default-on instance flag

- 目的：把 outlier 掩码中硬编码在函数内的 `outliermore = True` 显式化为实例属性，允许按层关闭 weight 自身 element-wise top-k 并集（仅保留激活 channel mask 掩 weight 列），与用户参考代码的结构对齐；行为保持不变。
- 内容：`quant/quant_linear.py` 新增 `self.outliermore = True`（`__init__`），`scale_inspection_bitnet` 与 `_split_outlier_operands` 改读该属性；`quant/quant_matmul.py` 的 `_split_outlier_operands` 中 B 侧 element-wise 并集同样受 `getattr(self, "outliermore", True)` 控制。掩码语义（weight = 激活 channel mask ∪ element-wise top-k）、数值、统计口径均无变化。
- 验证：smtqt 环境下 `tests/test_outlier_sparsity.py`（含逐 bit W codes 断言）、`tests/test_bit_sparsity_pipeline.py`、`tests/test_sparsity_ppl_matrix.py` 共 29 项测试加 71 子测试全部通过；新旧实现双路径数值对照确认正常/小 scale 下输出一致。
- 限制与旧结果影响：默认行为与既有实验完全一致，所有已产出结果（主矩阵、三轮翻倍、doubling）不受影响，无需重跑。若手动设 `outliermore = False` 则改变掩码语义，属新实验口径，需重新校准并独立目录。

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
## 2026-10-08 — Add iterative outlier-ratio doubling driver for sparsity PPL matrix

- 目的：按用户要求自动化"每轮跑完检测 PPL，把大于阈值的组 outlier ratio 翻倍重跑，直到全部低于阈值"的实验流程，免去手工多轮调度。
- 内容：新增 `scripts/run_sparsity_ppl_doubling.py`：第一轮按模板默认跑选中矩阵；每轮结束读取该轮 `summary.csv`，PPL 超过 `--ppl-threshold`（默认 20）的组在下轮单独调度并把 Linear 与 QK/PV outlier 同时翻倍（默认/零值起步用 `--seed-ratio`=0.0001，上限 `--max-ratio`=0.1，轮数上限 `--max-rounds`=8）；到上限仍不达标标记 exhausted 并停止。每轮独立 `round_NN` 目录，组级调度按单模型单格式调用底层 runner。最终输出 `doubling_summary.csv`：每组最终 PPL、达标状态、轮数、最终两 ratio、PPL 与 ratio 历史、exhausted 标记。退出码 0=全部达标，1=仍有超阈值组。
- 验证：smtqt 环境下 `tests/test_sparsity_ppl_matrix.py` 新增 2 项测试（固定失败下的逐轮翻倍与 max-rounds 终止、第二轮收敛场景、cap 场景 exhausted 终止与封顶值），全套 21 项测试加 31 子测试通过。逻辑用合成 PPL 与调度桩验证，未跑真实模型。
- 限制与旧结果影响：纯新增脚本，不改变既有 runner 行为与已产出结果。真实运行时 Qwen 组在 0.0002 已观测到 PPL 恶化（+6%~+26%），翻倍循环可能对其无效并最终以 exhausted/max-rounds 终止，属预期实验结论而非脚本故障；每轮全量校准+评测，15 组满矩阵多轮总耗时可观。
