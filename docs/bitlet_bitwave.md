# 一次 Qwen 推理同时收集 Bitlet 与 BitWave

scripts.profile_bit_arches 共用一次模型加载、一次校准和一条 prefill/decode 输入序列，将每次实际量化得到的 A/B 张量交给两个独立 collector。默认 batch=1、2048 prefill＋256 次 decode forward，IA FP8 E4M3FN、Linear W INT4、FP8 KV；与 Bitlet 单独入口使用相同量化配置和 scale 目录。

~~~bash
python -m scripts.profile_bit_arches \
  --model-path /path/to/Qwen2.5-14B \
  --output-dir outputs/bitlet_bitwave_2048_256
~~~

依赖仍为 requirements-model.txt。已生成并确认匹配的 Bitlet scales 可加 --skip-calibration 复用。首次校准需要 YAML 中的文本数据集；--token-file 只替代 profiling 输入。可先用独立 smoke scales 检查本地环境：

~~~bash
python -m scripts.profile_bit_arches \
  --model-path /path/to/Qwen2.5-14B \
  --prefill-length 256 --decode-steps 8 --calibration-samples 4 \
  --scale-dir quant/scales/bit_arches_smoke/ \
  --output-dir outputs/bit_arches_smoke
~~~

命令行缩短 prefill 时仍会限制校准长度。--greedy-decode 改为贪心生成；默认 teacher-forced。256 次 forward 与恰好生成 256 个输出 tokens 的定义见 [Bitlet 说明](bitlet.md)。

## 输出与一致性

| 文件 | 内容 |
|---|---|
| run_config.json | 共同 workload、校准/复用信息和两套硬件参数 |
| bitlet_summary.json / bitlet_trace.jsonl | Bitlet 相位/层/步汇总和逐算子记录 |
| bitwave_summary.json / bitwave_trace.jsonl | BitWave 相位/层/步汇总和逐算子记录 |
| bit_arch_comparison.json | 相同输入的调用数、各自计算加速比与时延情景 |

48 层完整实验中，每个 collector 都应收集 432 个 prefill 调用和 110592 个 decode 调用。runner 逐 collector 检查所有 layer/operator 的覆盖和共同 cache 增长；默认每 32 步更新汇总。中断或某个 collector 失败时，两份结果标为 interrupted，关闭两个 trace，不能用于完整 E2E。

额外 collector 会增加统计耗时和 trace 磁盘占用，但不会重复校准或重复数值推理。默认保持可复现抽样；--prefill-sample-waves / --decode-sample-waves 同时设置两者的预算，BitWave 的预算应用于每个候选 SU 的独立操作数。--exact 同时全量枚举两者，14B 下可能很慢。

comparison 中的 compute 加速比使用各自的 dense 布局作基准。比较两种架构的总性能时应使用前提一致的总时延；这些计算比率只反映各自跳过工作量的收益。

也可分别运行 scripts.profile_bitlet 或 scripts.profile_bitwave；其 quantization、seed、step 和配置一致时，单独运行与共同运行的统计一致。Bitlet 既有默认入口和映射公式保持不变，EBB 仍走独立入口。

## BitWave 的论文默认参数

来源：[BitWave, HPCA 2024 作者公开稿](https://arxiv.org/abs/2507.12444)，Sec.III、Sec.IV、Table I、Fig.11 与 Sec.V-A/V-B。

| 项目 | 默认值 |
|---|---:|
| BCE 数 | 512 |
| 1-bit × 8-bit SMM 数 | 4096 |
| 时钟 | 250 MHz |
| Activation SRAM | 256 KiB |
| Weight SRAM | 256 KiB |
| 每块 SRAM | 16 banks × 64 bits |
| SRAM 最大读宽 | 1024 bits/cycle，即 32 GB/s @ 250 MHz |
| 支持的 K 组大小 Cu | 8 / 16 / 32 |
| DRAM 速率 | null；论文未指定具体传输速率 |

Table I 的 SU 映射到 GEMM：Cu→K-group，OXu→M-tile，Ku→N-tile。读取带宽采用各自 SU 的数值，不能在所有 SU 中直接用最大 SRAM 带宽。

| SU | K-group | M-tile | N-tile | Weight bits/cycle | Activation bits/cycle |
|---|---:|---:|---:|---:|---:|
| SU1 | 8 | 16 | 32 | 256 | 1024 |
| SU2 | 16 | 8 | 32 | 512 | 1024 |
| SU3 | 32 | 4 | 32 | 1024 | 1024 |
| SU4 | 8 | 1 | 128 | 1024 | 64 |
| SU5 | 16 | 1 | 64 | 1024 | 128 |
| SU6 | 32 | 1 | 32 | 1024 | 256 |

SU7 专用于 depthwise convolution，不用于这九种 Qwen GEMM。auto 评估六个候选，按估计 compute＋streaming SRAM 输入读耗时选择；这不是论文的 ZigZag 全层次数据流搜索。可用 --bitwave-dataflow SU1 等固定一个 SU。SRAM 容量检查只验证一个扩展输入/输出 tile，不保证跨 tile 的全部复用、partial sum 或双缓冲可行。

## 独立的位列规则与 FP8 扩展

BitWave 以组内相同 significance 的 B bits 为一列，整列全零时跳过；一个有效 magnitude 列需要一个 bit-serial 周期。与 Bitlet 的 max(column population) 规则不同，不能复用其稀疏倍率。例如八个相同的整数 1：BitWave 只有一个 magnitude 列；Bitlet 的该列有八个有效 bits。

符号列用于存储/加载和加减控制，不算一个 magnitude MAC 周期。补码存储的 INT4 -8 需要四个 magnitude bits 加符号，不能截为三位。诊断包含原存储编码位数、零值、负值、非零列/列 population、输入/权重整数位宽、超出 native INT8 的组数及 BCS 压缩估计。

论文计算引擎面向整数操作数，未测量 FP8×INT4 Qwen。本适配器对 FP8 数值采用以下无损分析扩展：

1. E4M3FN / E5M2 分别从精确量子 2^-9 / 2^-16 得到整数代码；组内移除共同 trailing zero，保存共同 power-of-two scale。保留 hidden one 的数值，不将 FP8 exponent/mantissa 原始编码当作整数 magnitude。
2. A 的对齐整数若超过 signed INT8 范围，拆成若干 signed 7-bit-magnitude 分片，每片能以原生 INT8 输入表示；-128 在原生范围内。需要额外分片时计入 serial passes。
3. 较宽的 B 分成若干至多七个 magnitude bits 加符号的 index words，统计每列和重复符号存储。每 word 按八位 ZCIP index 计开销；FP8 对齐额外按两个 metadata bytes/group 建模。

因此，BitWave 的周期是整数分片分析情景，不是声称原论文硬件原生支持 FP8。分片结果累加、组指数转换、格式转换和控制停顿需额外成本。对齐保持数值且不截断，但负零在整数表达中归为零；数值 forward 始终使用原有量化模型。

Bit-Flip 会修改 weights 并要求独立精度验证，joint 模式关闭它；YAML 设置 bit_flip=true 会拒绝运行。两种架构采集相同量化操作数，也不自动跳过 A 的全零组。

## 压缩、流量与 E2E

BCS 压缩诊断包含索引、符号、K 尾组 padding 和 FP8 metadata，既给出相对 dense fixed-point 的比率，也给出相对原 W4/FP8 存储的比率；有开销时可能膨胀，不能只报告有效列数量。片外默认仍是紧凑 W4 与 FP8 KV；不会把尚未实现的 BCS cache 编码当作免费片外压缩收益。

GQA cache 读量按实际 KV heads 计算，QK/PV 分别计历史 K/V 与新增 cache 写入。local B 分 resident 和跨 M-tile streaming 两个情景；同一 B group 广播给多个 M 行。输出按 output_storage_bytes=2，group_setup_cycles=0 是显式假设。

未填 BitWave DRAM 速率时，summary 仍有周期、SRAM 时间和最低片外字节量；gemm_and_io_seconds 与 conditional_e2e_seconds 为 null。需要绝对搬运情景时显式设置速率，例如：

~~~bash
python -m scripts.profile_bit_arches \
  --model-path /path/to/Qwen2.5-14B \
  --bitwave-dram-bandwidth-gbps 12.8 \
  --output-dir outputs/bit_arches_bw_assumption
~~~

这里的 12.8 是使用者指定的比较条件，不是 BitWave 论文参数；GB/s 为十进制，不会改变 Bitlet 的两条独立 DMA 默认值。BitWave 按 Eq.5 的结构逐 GEMM 加 DRAM 与 output writes，再取 compute 和 SRAM input reads 的 max；另外给 streaming 无重叠情景。未模拟全部 register traffic、重排、spill 或控制，不能作为真实物理上下界。

--other-latency-json 可同时补两者的 LM head 和其他算子耗时；缺这些成本时 E2E 保持 null。采集后 scripts.estimate_bitlet_latency 也接受 BitWave summary（需已填 DRAM 速率），schema 与操作见 [Bitlet 说明](bitlet.md)。FP8 转换和分片累加成本需明确包含在剩余耗时中；不能把统计脚本的 GPU 墙钟时间作为 Bitlet/BitWave 时延。

本次仅在真实 CPU PyTorch/Transformers、小型 Qwen 上检查数值操作数共享、单次校准/推理、独立参考调度、单独/共同结果相同、两种 decode、cache、流式输出与失败关闭；完整 14B、CUDA、PPL 与 RTL 由本地实验补充。
