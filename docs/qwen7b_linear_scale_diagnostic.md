# Qwen2.5-7B Linear weight scale PPL 定位

在相同 FineWeb 样本与 QK/PV 配置下，分别隔离 Linear 的 weight scale、output quantization、activation 和 weight quantization。入口配置为 [`config/diagnostics/qwen7b_linear_scale_fast.yaml`](../config/diagnostics/qwen7b_linear_scale_fast.yaml)，脚本为 [`scripts/diagnose_qwen7b_linear_scales.py`](../scripts/diagnose_qwen7b_linear_scales.py)。默认只取 8 段、每段 1024 token 校准，评测约 8192 token；数值只能用于初筛，不能直接与原来长序列的 PPL 比较。

```bash
python scripts/diagnose_qwen7b_linear_scales.py \
  --model-path /path/to/Qwen2.5-7B \
  --output-dir outputs/qwen7b_linear_scale_fast
```

`--dry-run` 只展示实际参数；`--resume` 仅复用模型路径、配置指纹、入口代码以及成功且有限的 PPL 相符的旧报告。每次重新执行会新建 `attempt_NNNN`，保存配置、独立的校准 scale、完整日志及原始评测报告。根目录 `summary.csv` 和 `summary.json` 包含 PPL 及相对 `scalar_awo_fp8` 的差值；某组失败会记录错误并继续其他组，最后返回非零状态。默认关闭评测期的逐层操作数统计，仍执行正常校准和相同的 FP8 前向。BF16 raw 行跳过校准，走相同的包装模型。

| Case | Linear weight scale | Linear A/W/O | 用途 |
| --- | --- | --- | --- |
| `bf16_raw` | 不使用 | BF16 / BF16 / raw | 快速数据与模型基线 |
| `scalar_awo_fp8` | scalar | FP8 / FP8 / FP8 | 当前标量对照 |
| `channel_awo_fp8` | output_channel | FP8 / FP8 / FP8 | 每输出通道对照 |
| `scalar_aw_fp8_o_raw` | scalar | FP8 / FP8 / raw | 消除 Linear output 量化 |
| `channel_aw_fp8_o_raw` | output_channel | FP8 / FP8 / raw | 每通道并消除 output 量化 |
| `channel_a_raw_w_fp8` | output_channel | BF16 / FP8 / raw | 可选，定位 weight 路径 |
| `channel_a_fp8_w_raw` | output_channel | FP8 / BF16 / raw | 可选，定位 activation 路径 |

所有量化组固定 Linear outlier 3%、QK A/B FP8 且 output raw、PV A/B/O FP8、QK/PV outlier 8%，逐层 q/k/v/o/gate/up/down 使用相同设置。可选组：

```bash
python scripts/diagnose_qwen7b_linear_scales.py \
  --model-path /path/to/Qwen2.5-7B \
  --cases channel_a_raw_w_fp8 channel_a_fp8_w_raw
```

先比较 `scalar_awo_fp8` 与 `channel_awo_fp8`，再比较两个 `*_o_raw`。若去掉 Linear O 量化后差距消失，重点检查 output scale 的校准和与 weight scale 的耦合；若仍存在，再比较可选 A/raw 与 W/raw 组以及对应层的 `w_interval` 形状、极值与零值比例。短预算排名稳定后，用相同两组重新跑更接近原实验的预算，并检查基线是否仍接近预期：

```bash
python scripts/diagnose_qwen7b_linear_scales.py \
  --model-path /path/to/Qwen2.5-7B \
  --cases scalar_awo_fp8 channel_awo_fp8 \
  --calibration-samples 64 --calibration-seq-length 8192 \
  --eval-tokens 65536 --eval-seq-length 8192 --min-doc-tokens 4096 \
  --output-dir outputs/qwen7b_linear_scale_full
```

脚本本身不会保证 PPL 降到 10 以下；若快速 BF16 基线也明显偏离约 6，应先检查 checkpoint、数据段和评测口径。不同序列长度、样本量及其 calibration scale 不可当作同一次严格对照。
