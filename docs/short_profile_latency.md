# 短序列在线统计与长序列离线估算

在线入口默认采集 256 prefill＋32 decode；离线入口不加载 checkpoint，不需要 PyTorch，按目标长度重新计算工作量与最低搬运量。这是短序列校准后的条件估计，不是长序列实测或完整微架构仿真。

## 运行

只采集 BitWave：

```bash
python -m scripts.profile_bit_arches \
  --model-path /path/to/Qwen2.5-14B \
  --architectures bitwave \
  --output-dir outputs/bitwave_256_32
```

不指定 `--architectures` 时仍共同采集 Bitlet、BitWave、Slim-Llama。选择其中任意子集即可减少统计开销。`--architectures ebb` 走原有独立 EBB collector，不能与其他三者同时选择；新短配置的 EBB 为 275 MHz，Bitlet/BitWave/Slim-Llama 分别为 1 GHz/250 MHz/50 MHz。旧独立入口/YAML 不改频率或长度；需要短配置时用共同入口或显式 `--config config/qwen2_14b_bit_arches_f8i4_256_32.yaml`。

新短配置采用独立 scale 目录。64 个校准样本的默认数量不变，长度为 256；快速流程验证可显式 `--calibration-samples 4`，但这不验证量化精度。已有**匹配量化配置**的 scales 可通过 `--scale-dir /path/to/scales --skip-calibration` 复用，这样更能隔离缩短序列的影响；使用短校准重新得到的 scale 可能改变稀疏分布。`--frequency-mhz ebb=275` 可显式覆盖所选架构的时钟，不能通过时钟覆盖改变 PE/存储几何。

完整运行会生成原有 `*_summary.json`、`*_trace.jsonl` 及新增 `*_profile.json`。新增文件保存逐层、逐算子、逐阶段的 cycles/MAC，源周期、源 MAC、观察样本与映射信息、硬件/量化配置和来源。离线入口可直接读取以前完成的 `*_summary.json`，无需为增加 profile 文件重跑模型。

外推到 2048＋256：

```bash
python -m scripts.estimate_profile_latency \
  --profile-dir outputs/bitwave_256_32 \
  --architectures bitwave \
  --prefill-length 2048 --decode-steps 256 \
  --external-bandwidth-gbps bitwave=1000 \
  --output-dir outputs/bitwave_estimate_2048_256
```

这里的 **1000 GB/s 是显式系统假设**，不是 BitWave 原论文确认值；省略带宽仍输出 compute，GEMM/IO 保持 null。`--shared-external-bandwidth-gbps` 是统一比较的显式假设，`--external-bandwidth-gbps` 可逐架构覆盖。默认只复用 summary 中明确的 `dram_bytes_per_second`；不把 Bitlet 的双 DMA、片内带宽或 buffer 大小自动当作单一片外通道。频率缺省继承源 profile，也可 `--frequency-mhz architecture=MHz`，只缩放 compute。

输出 `profile_latency_summary.json`、`profile_latency_comparison.csv`，包含 dense、online_mapped、逐阶段/算子和逐 decode step 的估计。EBB 额外包含 online_leading_zero。不接受中断、遗漏层/算子或调用数不完整的运行；EBB word coverage 不完整时默认拒绝离线外推，只有明确添加 `--allow-incomplete-word-coverage` 才输出仍带有该标记的条件情景，不解决位宽超限的硬件路径。

## 在线数据如何进入离线计算

对架构 a、阶段 p、算子 o、层 l 分别计算：

\[
c_{a,p,o,l}=\frac{C_{\mathrm{short},a,p,o,l}}{\mathrm{MAC}_{\mathrm{short},p,o,l}},\qquad
T_{\mathrm{compute},a,p,o,l}=\frac{c_{a,p,o,l}\,\mathrm{MAC}_{\mathrm{target},p,o,l}}{f_a}
\]

稠密与映射周期各保存一套系数。decode 跨采集 step 按 MAC 加权；不混合 prefill 与 decode，不使用全模型统一 sparsity ratio。系数继承源 FP8 对齐位宽、多 pass、分组同步、映射利用率和控制占位值；这是对当前分析映射的校准，不证明与原芯片等价。

如果希望采用“在线稠密基准＋论文加速比”，显式添加 `--paper-speedup ebb=2.71` 或 `--paper-speedup bitwave=1.58`；新增 paper_speedup case 使用 `c_dense / S_paper`，不会再次乘在线 sparse 收益。Fig.16 没有 Bitlet/Slim-Llama，入口不为它们捏造默认倍率。论文倍率是否可外推到 decode/不同长度仍是条件假设。

目标形状以 Qwen/GQA 维度重新计算：

| 工作量（单层） | Prefill 长度 S | D 次 decode forward |
|---|---:|---:|
| Linear，权重 K×N | S K N | D K N |
| QK，H 个 query heads、head_dim h | H h S² | H h [D S＋D(D＋1)/2] |
| PV | 同 QK | 同 QK |
| 紧凑 W4 权重读量 | KN/2 bytes | D KN/2 bytes |
| 每路物理 FP8 KV 的历史读量 | Hkv h S | Hkv h [D S＋D(D−1)/2] |
| 每路 KV 写量 | Hkv h S | Hkv h D |

Prefill 采用完整 S×S attention GEMM；不因 causal mask 自动减半。D 指 prefill 后 D 次 decode **forward**，包含最后一次，不等同于某些生成接口的 D 个输出 token 所需 forward 次数。目标每个 decode step 的 context 为 S＋j，历史 cache 为 S＋j−1。计算用 query heads，搬运用实际 KV heads，避免 GQA 重复读取。

最低搬运口径 `weights_kv` 每次 forward 读一次权重，KV 每路读写一次；`materialized_once` 再计每 GEMM 一次 activation 读取与 output 写入。每层/算子分别按 `max(Tcompute,Tio)` 或 `Tcompute＋Tio` 合并，然后顺序相加。与原在线 collector 的 SRAM/压缩/容量窗口情景不是同一 IO 模型；不会直接复制或放大其旧 latency。

## 与 LLMCompass 配合及 E2E 范围

LLMCompass 可按**目标长度和目标系统**提供剩余成本 JSON，再用：

```bash
python -m scripts.estimate_profile_latency \
  --profile-dir outputs/bitwave_256_32 --architectures bitwave \
  --prefill-length 2048 --decode-steps 256 \
  --external-bandwidth-gbps bitwave=1000 \
  --other-latency-json bitwave=other_bitwave_2048_256.json \
  --output-dir outputs/bitwave_conditional_e2e_2048_256
```

JSON 延用现有 remaining-cost schema，并**必须明确目标长度**：

```json
{
  "schema_version": 1,
  "includes_lm_head": true,
  "workload": {"prefill_length": 2048, "decode_steps": 256},
  "prefill_seconds": 0.1,
  "decode_step_seconds": []
}
```

上例只是字段示意：空数组不能用于 256 decode；必须提供 256 个有限非负的实际估计值，不要填示例数字当结果。成本包括 LM head、Softmax/RMSNorm/RoPE/SiLU/residual、未纳入在线系数的转换/控制及其他必要成本。每个架构分别供给；避免再次添加已由本入口统计的 GEMM 或相同权重/KV 搬运。这次没有自动改造 LLMCompass 的其他架构后端，也没有执行其系统仿真。

当前离线搬运**没有**容量驱动的 tiling 重读、CIM refill 复制、SRAM/NoC/bank 冲突、metadata 和 partial-sum spill。要达到之前完整 quantspar＋LLMCompass 的系统级建模，需要让对应架构的 memory/scheduler 后端接收目标 shape 与 profile 数据，并代替当前最低搬运模型；不是导入一个 sparsity ratio 就具备这些成本。没有带宽或其他成本时 E2E 为 null；补成本后字段仍为 conditional_e2e_seconds，不声称物理芯片实测。

## 外推精度

静态权重统计最容易复用；激活、QK/PV 的动态稀疏和指数分布会随数据/上下文变化。当前假设短 run 的逐算子 cycles/MAC 在长序列上保持不变，包括 padding 利用率、BitWave dataflow 和 Slim-Llama 中心复用的摊销；目标 shape 可能改变这些值。建议用额外文档或 512＋64 的短验证点检查系数变化，论文报告为 calibrated extrapolation，并保留来源和长度。不能称为“等价的长序列实测”。
