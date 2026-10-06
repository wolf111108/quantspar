# EBB-CIM / MulTCIM 统计与本地运行

目标：Qwen2.5-14B，batch=1，Linear IA E4M3FN、W INT4；prefill=8192，decode=1024；片外 Linear 权重按紧凑 W4，KV 按 FP8。LM head、Softmax、RMSNorm、RoPE、SiLU 的系统时延继续使用 LLMCompass 的配置。

本后端采集真实量化 codes，输出有效位宽分布、分组失衡、位宽溢出、条件 GEMM 周期范围和最低必需搬运字节数。它不调用原 Asyn-CIM 的尾数 popcount 映射，也不修改原实验 YAML。

## 本地运行

在仓库根目录使用已有 CUDA PyTorch 环境：

```bash
python -m pip install -r requirements-model.txt
python -m scripts.profile_ebb \
  --config config/qwen2_14b_ebb_f8i4.yaml \
  --model-path /path/to/Qwen2.5-14B \
  --output-dir outputs/ebb_qwen14b_8192_1024
```

输出目录必须是新目录。默认校准 FineWeb 的 64 个 8192-token 样本（seed=23），量化 scale 保存到新的 `quant/scales/qwen2_14b_ebb_f8i4/`。第一次不要传 `--skip-calibration`；再次运行同一量化/校准配置时可传该参数复用 scale。旧 outlier、INT8 或不同粒度的 scale 不能复制到新目录。

先核对接口和缓存增长，可运行较短的 profiling：

```bash
python -m scripts.profile_ebb \
  --model-path /path/to/Qwen2.5-14B \
  --output-dir outputs/ebb_smoke \
  --scale-dir outputs/ebb_smoke_scales \
  --prefill-length 128 --decode-steps 4 --calibration-samples 1
```

这仍使用 YAML 的 8192-token 校准长度；1 个校准样本只用于接口检查。测试 scale 使用独立目录，正式命令使用默认新目录并按 64 样本重校准。

默认 decode 是 teacher-forced：从同一文档读取后续 1024 个 token，逐个带 cache 推理。输入文档需至少 9216 tokens。`--greedy-decode` 改为模型自身生成；此时只需 8192-token prompt，会调用 LM head 选择 token，LM head 不进入 EBB GEMM 统计。

已有固定 token 输入可使用 `--token-file tokens.pt`，文件为 `torch.save(torch.long_tensor)`，形状 `[tokens]` 或 `[1,tokens]`，也支持 `{"input_ids": tensor}`。这可以绕过 profiling 的在线文档加载；首次校准仍使用配置的校准数据集。

## 输出及核对

| 文件 | 内容 |
|---|---|
| `ebb_summary.json` | 每 layer/operator/phase 的计数与位宽直方图、phase 周期/秒数、逐 step 合计、原生位宽覆盖检查、建模假设与缺失系统参数 |
| `ebb_trace.jsonl` | 每次算子调用的 shape、宏布局、周期、GQA/cache 信息、最低搬运量及溢出计数；不保存大张量 |
| `run_config.json` | 运行设置、模型维度和软件版本 |

每 32 个 decode step 写一次汇总检查点；`--checkpoint-every` 可调整。异常退出时保存已完成计数，`workload.status` 为 `interrupted`，不能作为完成实验。成功时是 `complete`，`completed_decode_steps=1024`。

Qwen2.5-14B 应有 48 层，每层 7 Linear＋QK＋PV：prefill 432 次调用，decode 442368 次调用。完整 trace 共 442800 行。decode `cache_length_before` 为 8192…9215，`cache_length_after` 为 8193…9216。

`phases.<phase>.compute_seconds` 有三个字段：

- `dense`：相同布局、相同 word width 下的稠密串行计算。
- `leading_zero`：只去掉前导零，仍由组内最长有效位宽决定。
- `ideal_balanced`：理想组内平衡的周期下界。

全部是条件计算模型，单位为秒；不是 Python/GPU 程序运行时间，也没有包含片外搬运、CIM 写入或非 GEMM。要获取整个工作负载的条件 GEMM 范围，分别相加 prefill 与 decode 的 `ideal_balanced`、`leading_zero`。不要把倍率乘在已有稀疏 TOPS 上。

## 对照论文的统计定义

依据 MulTCIM（JSSC 2024，DOI `10.1109/JSSC.2023.3305663`），Sec. VI、Fig. 13–15：EB detector 每组 8 输入；前导零去除后，普通 bit-serial 时间由最大 EB 决定；平衡器向短 EB 的位置搬移长 EB 的高位，同时跨移/移位权重。

本模型按 K 维分组，不跨 token/head 拼组，保留 K 尾部有效元素计数。对于正整数 `v`，EB 是 `bit_length(v)`，不是 popcount。例如 `1000` 的 EB 为 4，内部三个零不能当作三次 EBB 跳过。

对一组宽度 `b_i`，下界为 `ceil(sum(b_i)/8)`，仅前导零上界为 `max(b_i)`。Fig.13 的四元素 EB `[4,1,4,7]` 对应 8→7→4 个周期；测试使用两份该例填满八输入，仍得到 8→7→4。论文没有给出足以复现全部交叉网络限制与匹配控制器的细节，因此理想平衡只报告下界。

默认 `signed_encoding=twos_complement`：负值保留完整补码 word width，不假定可免费去除符号扩展。可选 `sign_magnitude` 把 magnitude EB 加一个串行符号槽；这是另外的显式假设，结果不可混用。输出同时记录负值比例及 required signed width。

## FP8 对齐与位宽覆盖

论文原生支持 INT8/INT16，没有发布 FP8×INT4 的硬件时序。本软件使用 `fp8_alignment=exact_dyadic`：

1. E4M3 codes 乘 `2^9`，E5M2 codes 乘 `2^16`，得到精确整数。
2. 选组内非零元素最小的**编码 LSB 指数**作为共同指数，整数相应右移。
3. 保留完整 significand（含隐含位）、指数跨度和符号；不会删除尾数内部零或统一的尾数末尾零。例如 E4M3 的 `1.0` 仍是 `1000`，不是 `1`。
4. 检查共同指数下的整数是否可放入配置的 8/16-bit signed word；不裁剪数据、不改变前向运算。

例如一组同时含 `448` 和 E4M3 最小 subnormal `2^-9`，需要 19-bit signed integer。它会被记录为 INT16 溢出，而不是假装仍需 16 个稠密周期。

`configured_word_coverage_complete=false` 表示至少一个 A 或 B 组超过配置的 word width。A 溢出会保留扩展串行位宽；B 溢出表示配置的权重存储无法容纳其对齐整数，当前周期仍以配置的 bank pass 为条件。两者都不能用于声称原生 MulTCIM 延迟，需另选转换/多次计算策略。即使覆盖完整，FP8 对齐、partial-sum 指数对齐等转换开销也仍未计入。

## 宏映射假设

默认 128 宏、每宏 32 arrays、8 banks、4 个复用 row，160 MHz，对应论文 0.7 V 测试频率。275 MHz 是另一工作点，可修改 `frequency_hz`；不可同时借用 160 MHz 的时延和 275 MHz 的吞吐。

紧凑 W4 是片外格式，CIM 内部按 8-bit word 保存 Linear 权重；attention 的 FP8 B 对齐后按 16-bit word 预算。INT16 使用两个 8-bit bank pass、驻留 K 容量减半，这是显式的分析假设。内部打包、FP8 转换控制器和跨组指数累加没有声称是论文公布的实现。

布局按 dense 条件选择 K/token/N 并行份额。每个 macro 累加其分配的 K 组周期，每 token wave 等待最慢 macro，再串行执行 N rounds；独立 KV-head operand 串行执行。三个周期模型使用同一布局。可用 `group_setup_cycles` 加每组统一设置开销，默认 0；未知开销也记录在 scope 中。

该布局的原生 dense 参考在 275 MHz 下为 INT8 2.2528 TOPS、INT16 0.5632 TOPS。论文公布的 3.55/0.89 TOPS 已包含其 bit-sparsity 收益，不能作为本后端的 dense 基准再乘 EBB 加速比。

## KV、GQA 与最低搬运量

新配置在 RoPE 后、`cache.update` 前，用已校准的 QK-B / PV-B scale 将新 K/V 量化至 FP8。HF Cache 内仍保存反量化浮点值，以模拟 FP8 缓存语义；不是 CUDA FP8 cache kernel，也不是测量 GPU 的实际 FP8 存储带宽。

对共享 KV 的 GQA，把同一 KV head 对应的 5 个 query head 合并为 M 行供映射；实际 MAC 不减少。B 统计和物理 cache 字节数按 8 个 KV head，而不是 `repeat_kv` 后的 40 个 head。

动态 KV 权重直方图使用精确增量更新：QK 对新 token 增加完整 head_dim 分组；PV 复用已完成的 K 分组，重新统计末尾不满 8 token 的组。不会对每个 decode step 重扫全部历史 KV，也没有抽样省略。前提是新配置使用固定 B scale、不可变历史 FP8 cache。

`packed_weight_read_bytes_minimum` 按每次 Linear 调用的 `ceil(numel*4/8)`；`fp8_kv_read_bytes_minimum` 按调用前缓存长度，每个 K/V 元素 1 byte；新 token 的 KV 写入另列 `fp8_kv_append_bytes`。这不包含量化 scale/metadata，也不含 tiling 重复读、CIM 重复写入和其他 activation IO。

## 后续端到端计算仍需的数据

从本地实验可以补齐实际有效位宽、组失衡、溢出、层/阶段/上下文依赖及条件计算范围。要形成端到端结果，仍需：片外带宽；CIM 写入带宽/时间及 refill schedule；搬运与计算的重叠规则；FP8/partial-sum 转换开销；以及你沿用的 LLMCompass 非 GEMM 时延。精确 EBB 值还需要控制器匹配/路由细节，可在获得这些资料后替换当前平衡下界。

`export_cim_stats` 和原 Asyn-CIM 的 `export_llmcompass_manifest` 会拒绝 EBB manager，避免使用错误的后端基准。EBB 专用入口是 `export_ebb_stats`；导出中明确包含剩余系统输入。

## 验证

```bash
python -m unittest discover -s tests -v
```

核心依赖下运行数学/编码与原 Asyn 回归；安装 `requirements-model.txt` 后，额外运行真实小型 Qwen 的 prefill/decode/FP8-cache 测试，以及保存模型后通过 CLI 的 teacher-forced/greedy 冒烟验证。完整 14B、8192＋1024 和 CUDA 性能由本地实验验证。
