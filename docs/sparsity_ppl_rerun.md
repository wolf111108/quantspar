# 三个模型 × 五种格式：FineWeb 稀疏度与 PPL 重跑

`config/sparsity_ppl_rerun/` 提供 15 份独立 YAML。模型为 OPT-1.3B、OPT-6.7B、**Qwen2.5-7B**，不包括 GPT-2，也不使用 Qwen2-7B 或 Qwen2.5-14B 替代。每个模型都运行下面五组，包括旧表中 PPL 为 NaN 的组。

| 文件后缀 / `--formats` | Linear A | Linear W | Linear O | QK/PV A、B、O | 统计位宽 A / W |
|---|---|---|---|---|---|
| `bf16_bf16` | BF16 | BF16 | BF16 | BF16 | 7 / 7 |
| `fp8_fp8` | FP8 E4M3FN | FP8 E4M3FN | FP8 E4M3FN | FP8 E4M3FN | 3 / 3 |
| `int8_int8` | INT8 | INT8 | INT8 | INT8 | 8 / 8 |
| `int8_int4` | INT8 | INT4 | INT8 | INT8 | 8 / 4 |
| `fp8_int4` | FP8 E4M3FN | INT4 | FP8 E4M3FN | FP8 E4M3FN | 3 / 4 |

列名 A×W 描述 Linear 的输入与权重。QK/PV 的 B 是动态 K/V，不是静态 Linear W；混合组的 K/V 跟随 A 格式。FP8 在 YAML 中明确写 `e4m3`，数字 `8` 表示 INT8。

所有组都显式以 `model.dtype: bf16` 加载 checkpoint，attention 为 eager，关闭 mixed precision 和 unit，LM head 不替换。BF16 组使用 BF16 A/W/B/O、原生 scale=1，通过 `quant_forward` 采集尾数；**不能加 `--fp-baseline`**，该开关走 raw 并关闭统计。数值计算沿用当前 FP32 codes/累加与输出舍入的仿真通路，不代表原生 BF16/FP8/INT4 kernel 性能。

## 本地运行

在仓库根目录更新代码并安装模型依赖，把三个路径改成本地 checkpoint 目录：

```bash
git pull --ff-only
python -m pip install -r requirements-model.txt

bash scripts/run_sparsity_ppl_matrix.sh \
  --opt-1-3b-path /path/to/opt-1.3b \
  --opt-6-7b-path /path/to/opt-6.7b \
  --qwen-7b-path /path/to/Qwen2.5-7B \
  --output-dir outputs/sparsity_ppl_rerun
```

也可以使用 `python -u -m scripts.run_sparsity_ppl_matrix`，参数相同。Shell 入口默认使用 `python`，可用 `PYTHON=/path/to/environment/bin/python` 指定环境；选择单张卡可在命令前设置 `CUDA_VISIBLE_DEVICES=0`。**必须始终在同一个 conda 环境中启动矩阵（含 `--resume`）**：任务指纹包含运行环境依赖版本，混用环境会因缺 transformers Qwen2 模块立即失败，并把 manifest 指纹覆盖为错误环境的值，导致 `--resume` 无法识别已完成的任务。

默认 `--eval-flow all`：每组重新校准，再运行 PPL full-forward 与独立的 teacher-forced prefill/decode，顺序启动一个模型子进程，退出后开始下一组。每组完整日志写入自己的 `run.log`，终端输出进度。某一组非零退出、缺报告或 PPL 非有限时，记录状态并继续后续组；最终有失败时脚本退出码为 1。Ctrl-C 保留已完成结果与当前 attempt，退出码为 130。

先检查矩阵，不加载模型也不写结果：

```bash
bash scripts/run_sparsity_ppl_matrix.sh --dry-run
```

只运行一组；若只需表格的 PPL 和 full-forward 比特比例，可加 `--eval-flow ppl`，此时不运行 PD：

```bash
bash scripts/run_sparsity_ppl_matrix.sh \
  --models qwen2.5-7b --formats fp8_int4 \
  --qwen-7b-path /path/to/Qwen2.5-7B \
  --eval-flow ppl \
  --output-dir outputs/qwen7b_fp8_int4
```

中断后原命令加 `--resume`。只跳过 PPL 有限、所需阶段/六类操作数均有计数、详细 JSON/CSV 完整且 SHA256 校验一致、checkpoint 路径/配置/代码/依赖版本指纹一致的 completed 项；失败、非有限、损坏或设置不同的项重新校准。默认不加 `--resume` 会重新运行全部选中项，每次创建 `attempt_0001`、`attempt_0002` 等新目录，保留历史文件。指纹不计算 checkpoint 权重文件哈希；如果在原路径替换权重，去掉 `--resume` 或使用新输出目录。

也可以用翻倍驱动器自动执行"PPL 超阈值即把 Linear 与 QK/PV outlier 同时翻倍重跑"的循环，直到全部低于阈值或触及 `--max-ratio`/`--max-rounds`（届时该组标记 exhausted）：

```bash
PYTHON=/path/to/python bash scripts/run_sparsity_ppl_matrix.sh --help  # 环境要求同上
python -m scripts.run_sparsity_ppl_doubling \
  --opt-1-3b-path /path/to/opt-1.3b --opt-6-7b-path /path/to/opt-6.7b \
  --qwen-7b-path /path/to/Qwen2.5-7B --ppl-threshold 20
```

每轮写入独立 `round_NN` 子目录（含该轮 manifest/summary），结束后顶层生成 `doubling_summary.csv`（每组最终 PPL、轮数、最终两 ratio、历史与 exhausted 标记）。注意 Qwen 在加严旁路下 PPL 反而恶化，翻倍循环对其可能无效并以上限终止。

## 数据与旁路设置

统一使用 `HuggingFaceFW/fineweb`、`sample-10BT`、`train`、streaming、text 字段，校准 seed=23、64 样本、batch=1。PPL 对同一模型的五组使用相同数据筛选和 **65,536 输入 tokens 预算**；数据加载器从固定顺序流取前缀，校准 loader 使用带 seed 的 shuffle。

| 模型 | 校准 / PPL 序列长度 | 校准 / PPL 最短文档 tokens | PD prefill＋decode | PD 样本数 / 最短文档 tokens |
|---|---:|---:|---:|---:|
| OPT-1.3B | 2048 / 2048 | 128 / 128 | 1024＋64 | 1 / 4096 |
| OPT-6.7B | 2048 / 2048 | 128 / 128 | 1024＋64 | 1 / 4096 |
| Qwen2.5-7B | 8192 / 8192 | 4096 / 4096 | 2048＋64 | 1 / 4096 |

PPL 沿用分段评测，每段首 token 不计算 loss，末尾不足整段的 tokens 丢弃；预算不等于预测 token 数。不同模型的 tokenizer、序列长度和文档筛选不同，不能把跨模型数值归因为量化格式的单一影响。

四个量化组的全部 Linear 默认 `outlier_ratio: 0.0001`，BF16 基线为 0，QK/PV 均为 0。所有格式都关闭 mixed precision。normal codes 按完整形状统计，**计入 mask 人为产生的零**，高精度 protected sidepath 数值参与 PPL forward，其 codes 不计入比特比例。带输入相关 mask 的 W 每次 forward 重计，无 mask 的静态 W 每阶段计一次。

需要对照无旁路的实验，在原命令中加 `--outlier-ratio 0`，建议配合新的 `--output-dir`。该参数默认同时覆盖全部 Linear（包括 BF16）及 QK/PV；若需两者使用不同的比例，用 `--qk-pv-outlier-ratio` 显式覆盖 QK/PV（进入指纹，优先级高于 `--outlier-ratio`）；`summary.csv` 的 `linear_outlier_ratio` 与 `qk_pv_outlier_ratio` 两列分别记录两者的有效值。

模板使用独立 `quant/scales/sparsity_ppl_rerun/<模型>_<格式>/` 并强制 `calibration_policy.default: recalibrate`。批量脚本进一步将有效 YAML、scales、日志与结果放进每个 attempt 内，避免跨模型、跨格式、跨重跑复用旧 scales；不添加 `--skip-calibration`。标准 Linear/QK/PV 校准 scale 按每批观测值逐元素取最大值（标量或逐输出通道），不再只使用最后一个 batch；需重新校准后重跑量化 PPL/稀疏统计。

## 输出和统计口径

输出目录包含：

- `summary.csv`、`summary.json`：每组状态、完整数值 PPL、预算与 outlier 设置，full-forward/prefill/decode 的总比特比例和六类操作数比例。
- `operand_sparsity.csv`：按 run/phase/operand/outlier_masked 保留 `zero_bits`、`bits`、比特比例，以及独立的元素零值计数。
- `manifest.json`：任务状态、设置指纹、attempt 路径与退出码。
- `jobs/<run>/attempt_NNNN/config.yaml`：实际执行配置；同目录的 `scales/`、`run.log`、`results/` 和 `stats/` 保留该次校准/评测结果。
- `results/<run>_evaluation.json`：主入口新增的机器可读报告，保留完整 PPL、有效配置、依赖版本和各阶段快照；原 TXT 报告继续输出。
- `stats/<run>_full_forward_bit_sparsity.{json,csv}`、`stats/<run>_prefill_decode_bit_sparsity.{json,csv}`：现有逐层/格式计数；`--eval-flow ppl` 只生成前一组。

CSV 中比例是 **0–1**，显示百分数时乘以 100。汇总始终为 `sum(zero_bits) / sum(bits)`，包括 masked/unmasked 组，不平均各层或各组百分数。空统计、失败或 NaN PPL 保留空值/JSON null 和状态，不能解释为零；`nonfinite_ppl` 项如果统计采集完整，仍保留它的稀疏度，但 PPL 不作为有效结果。

`activation` 是 Linear A；`Q` 与 `attention_probs` 是 attention A；`weight` 是 Linear W；`K`、`V` 分开统计。若表格仅报告 Linear 输入激活稀疏度，取 `full_forward_activation_bit_zero_ratio`，而不是 `full_forward_total_bit_zero_ratio`。后者合并所有已包装乘法输入，W 的采样频率还受旁路 mask 影响，不能当作唯一张量的存储压缩率。PD 的 K/V 描述各次 attention 乘法操作数，历史 cache 可重复参与统计，不意味着实际 cache 按 INT8/FP8 打包存储。

所有 FP 操作数只计原生编码的**显式尾数**：BF16 为 7 位，E4M3FN 为 3 位，不含 sign、exponent、hidden one；所有 INT 使用完整补码。截图脚注写的是“指数对齐后的尾数”，与本次口径不同。旧上传配置也包含 FP16 而非 BF16、outlier=0、Qwen PPL 预算不一致等设置；本次结果应作为新实验填写，不能声明直接复现或覆盖旧脚注口径。

这次只验证 YAML、报告与调度器逻辑，使用合成计数/结果测试继续运行、断点恢复、NaN 和加权汇总；未在这里运行真实 checkpoint、FineWeb、CUDA 或完整 PPL 实验。
