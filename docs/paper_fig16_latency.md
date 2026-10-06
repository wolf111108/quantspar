# Fig.16 倍率、原生计算容量与带宽的离线估算

无需运行 Qwen、加载 checkpoint、校准或安装 PyTorch。此方法按固定模型尺寸计算工作量，使用论文 Fig.16 的 Qwen2.5-14B 倍率和各架构原生资源推导计算容量，叠加紧凑 W4 / FP8 KV 的搬运情景。完整 E2E 仍需提供沿用 LLMCompass 设置、对应目标 shape 的剩余算子成本。

## 直接运行

在仓库根目录：

```bash
python -m scripts.estimate_paper_latency \
  --output-dir outputs/paper_fig16_2048_256
```

默认 batch=1，2048 prefill 后 **256 次 decode forward**，Qwen2.5-14B 的 48 层、d=5120、FFN=13824、40 Q heads、8 KV heads、head_dim=128。长度可用 `--prefill-length`、`--decode-steps` 改变；倍率仍是假定可以外推，不能成为新长度的实测倍率。

结果：`paper_latency_summary.json` 包含原始资源、来源、假设、覆盖参数、逐阶段/算子工作量、dense 与 Fig.16 两套结果及逐 decode step 时延；`paper_latency_comparison.csv` 并列五种架构。目录必须不存在，防止覆盖之前的结果。

## Fig.16 与硬件基准

| 架构 | Qwen14B Fig.16 倍率 | 原生资源及频率 | 推导的 dense TOPS | 片外带宽 |
|---|---:|---|---:|---:|
| SIGMA | 1.58 | 128×128 BF16 MAC，500 MHz | 16.384 | 1024 GB/s，Fig.8 |
| BitWave | 1.58 | 512 BCE×8 SMM，1b×8b，250 MHz；8 串行位等价 512 完整 MAC/cycle | 0.256 | 未查到明确 DRAM 速率，null |
| EBB-CIM / MulTCIM | 2.71 | 16 cores×8 macros×32 arrays；每 array 8 个 bit×byte 乘法，8 cycles/MAC，275 MHz | 2.2528 | FPGA DDR3/FMC 不等于有效芯片带宽，null |
| Bit-Pragmatic | 3.33 | 16 tiles×256 PIPs，每 PIP 16 terms/cycle，原生 16b 基准 | `0.008192 × frequency_MHz` | inspected paper 未给出可用绝对频率/DRAM 速率，均 null |
| Asyn-CIM | 4.01 | 16 macros×16 banks×48 byte-weight lanes，BI=3，1 GHz | 8.192 | 1000 GB/s |

Bitlet 和 Slim-Llama 不在 Fig.16，不能把 Bit-Pragmatic 的 3.33 倍分配给 Bitlet，也不能给 Slim-Llama 编造对应倍率。它们继续使用已有实际操作数 collector。Fig.16 数字按图中柱标读取；附近正文中的部分范围与柱标不一致，本入口保留图的数字。

配置集中在 `config/qwen14b_fig16_capacity.json`。其中 PE 的含义各不相同，所以同时保存 `products_per_pe_per_cycle` 和 `dense_cycles_per_mac`，使用

`P_dense = 2 × PE_count × products_per_PE_per_cycle × frequency / dense_cycles_per_MAC`。

不能只把 4096 个 BitWave SMM 当成 4096 个完整 INT8 MAC；不能把 EBB 3.55 TOPS、BitWave 0.22 TOPS 等已包含稀疏/利用率收益的数字再乘 Fig.16 倍率。W4 的片外打包也不自动给计算吞吐翻倍。

SIGMA 原文 Fig.8 给出 500 MHz、16K MAC 和 1024 GB/s HBM；表中 16 TFLOPS 为取整，代码根据 MAC 数推导 16.384。MulTCIM 275 MHz 与已有 collector 的默认 160 MHz 属于不同工作点。Bit-Pragmatic 使用原始 16b 布局，不混用 Sec.6.7 的缩减 PIP 列数的 8b 变体，也不借用 Bitlet 时钟。

片内带宽、buffer 容量和来源已记录在配置中。当前模型不模拟内部驻留、NoC、bank conflict、CIM refill 或 buffer spill，因此这些容量不能被解释成已经验证了 residency。BitWave Table I 的 SRAM feed 不是片外 DRAM 带宽。

## 在缺少硬件参数时运行条件情景

显式给缺失参数，示例中的 1 GHz 和 1000 GB/s 是**分析假设**，不是论文报告的 Bit-Pragmatic/BitWave/EBB 工作点：

```bash
python -m scripts.estimate_paper_latency \
  --output-dir outputs/paper_fig16_shared_1TBps \
  --shared-external-bandwidth-gbps 1000 \
  --frequency-mhz bitpragmatic=1000
```

共同带宽会覆盖所有架构，包括 SIGMA 原生 1024 GB/s。要保留 SIGMA 原生值，再追加 `--external-bandwidth-gbps sigma=1024`。针对不同架构可重复使用 `--external-bandwidth-gbps ARCH=VALUE`，频率同样用 `--frequency-mhz ARCH=VALUE`。所有覆盖都写入 JSON provenance，单位 GB/TB 为十进制。

默认 `prefill_utilization=decode_utilization=1`，表示乐观容量估计，**不代表 batch-one decode 可以用满所有 PE**。例如 BitWave 的 SU4–SU6、Bit-Pragmatic 的 16 token/window 并行都会影响 decode。可做显式敏感性分析：

```bash
python -m scripts.estimate_paper_latency \
  --output-dir outputs/paper_fig16_decode_utilization \
  --shared-external-bandwidth-gbps 1000 \
  --frequency-mhz bitpragmatic=1000 \
  --decode-utilization bitwave=0.25 \
  --decode-utilization bitpragmatic=0.0625
```

这些比例是映射敏感性情景，不能冒充 Fig.16 已公布的 decode 利用率。也不能把 Fig.16 已包含的平均同步损失再当成同一损失重复扣除。

## 工作量和访存

每层七个 Linear 的参数量：

`2d² + 2d×(KV_heads×head_dim) + 3d×FFN`。

48 层共有 13,212,057,600 个 Linear 权重，紧凑 W4 为 **6,606,028,800 bytes**，不含 LM head、embedding、scale 和 metadata。

prefill 完整 QK/PV 矩阵的 FLOPs：`2P×N_linear_parameters + 4L×d×P²`。沿用当前 eager 路径，因果 mask 不缩小 GEMM shape。decode 第 j 次计算包含本步 token，FLOPs 为 `2N_linear_parameters + 4L×d×(P+j)`，j=1…D。

访存按物理 KV heads 计，计算按 query heads 计。prefill KV 写一次、读一次；decode 的历史 KV 读取为 `2L×KV_heads×head_dim×(P+j-1)` bytes，新 KV 写入 `2L×KV_heads×head_dim` bytes，不把本步新 KV 当成片外历史再读一次。每个 Linear 的权重在 prefill 和每个 decode forward 各读取一次，不假定整个 6.606 GB 权重驻留。

默认 `--traffic-mode weights_kv` 只计紧凑权重与物理 KV，适合作为乐观的搬运情景。`--traffic-mode materialized_once` 额外计每个 GEMM 一次 FP8 A 读取和 FP16 output 写入。两者都没有 tiling 重读、内部 FP8 定点扩宽、多 pass、scale/indices、partial-sum spill、CIM 写入停顿或命令启动成本。

这不是对所有架构完全相同精度的原生性能预测。SIGMA 为 BF16 MAC；BitWave/EBB/Pragmatic 的整数计算容量用于 FP8 **代理估算**，不是已验证的无损转换时序。此方法不采集实际 FP8 指数跨度，因此无法验证整数 word width 覆盖。原 FP8 collector 的多 pass/对齐代价也不移植到本倍率模型，否则很容易混合不同基准。

## 时延公式与 E2E

每个算子分别计算，再依照串行的 Transformer 调用顺序相加：

`T_compute_i = FLOPs_i / (P_dense × Fig16_speedup × phase_utilization)`。

`T_full_overlap_i = max(T_compute_i, transfer_bytes_i / external_bandwidth)`。

`T_no_overlap_i = T_compute_i + transfer_bytes_i / external_bandwidth`。

对 decode 逐 j 求和；不能用 `max(整个请求计算时间, 整个请求搬运时间)` 代替逐算子求和，也不能用 `dense_E2E / Fig16_speedup`。稠密对照只把 speedup 换成 1，带宽和流量保持相同。

完整 `T_E2E = T_prefill + Σ_j T_decode_j + T_other_prefill + Σ_j T_other_decode_j`。利用已有剩余成本 JSON，无需运行模型：

```bash
python -m scripts.estimate_paper_latency \
  --output-dir outputs/paper_fig16_e2e \
  --shared-external-bandwidth-gbps 1000 \
  --frequency-mhz bitpragmatic=1000 \
  --other-latency-json /path/to/other_latency_2048_256.json
```

沿用 `schema_version=1`、`includes_lm_head=true`、`prefill_seconds` 和长度等于 decode_steps 的 `decode_step_seconds` 列表；数值必须有限且非负。建议另外保存 `workload: {"prefill_length": 2048, "decode_steps": 256}`，入口会拒绝与目标长度不符的成本。旧 JSON 没有 workload 字段时仍可使用，但 `other_workload_verified=false`，需自行确认来源。这里的 other 不应重复包含已建模的七 Linear 和 QK/PV。

按照用户当前比较政策，LM head、Softmax、RMSNorm、RoPE、SiLU 等沿用 LLMCompass 设置；需要针对 2048＋256 **重新计算**，不能复制 8192 的非 GEMM 时间。相同 other 可作为共同系统假设，但不能声称每个原生芯片都有相同 vector engine。未知格式转换/控制成本也应补入 other 或另行披露。

没有 other 时 E2E 为 null；缺少 frequency 或外部 bandwidth 时 GEMM/IO 和 E2E 都为 null，同时保留可求出的计算/流量。没有将缺失成本设为零，也没有使用 Fig.14 的 16.90 ms TPOT 替代当前 W4/KV-FP8 访存。

倍率来自论文 8192-token 评估，而本目标为 2048＋256。**同一个平均倍率用于不同算子和两个阶段**是此简化方法最大的外推假设。建议把结果称为“基于 Fig.16 的条件容量/访存估计”，不要称作芯片测得的 E2E 或 RTL 重现。两种重叠情景没有模拟 buffer，不能保证实际物理上下界。

Asyn 的 4.01 与 Fig.14 的 after-I/O prefill 吞吐比 `32.84/8.19` 一致，说明该倍率也可能保留来源系统的 I/O 损失。把它当成计算容量校准，再对新访存取 roofline，是近似，会残留部分 I/O 惩罚；不能声称它严格等于纯计算加速比。

作为入口的数值核对，默认 `weights_kv`、full overlap、utilization=1 时：原生 SIGMA 1024 GB/s 得到 3.954790 s GEMM/IO，Asyn 1000 GB/s 得到 3.518821 s。若**人为统一所有外部带宽到 1000 GB/s 并假设 Bit-Pragmatic 为 1 GHz**，五者依次得到 3.995710 / 162.065032 / 11.320405 / 3.880855 / 3.518821 s。这些是标准库计算器的输出，不是完整模型运行或芯片测量；完整 E2E 还须加同一目标 workload 的 other。

## 验证

```bash
python -S -m unittest discover -s tests -p test_paper_latency.py -v
```

独立公式核对 Qwen14B 参数、GQA 字节数、2048＋256 工作量、decode cache 长度边界；用小尺寸标量参考核对逐算子重叠、带宽不被倍率放大、dense/fig16 两套结果；检查缺失参数、剩余成本、零 decode、非法数值、输出保护及无 site-packages CLI。无需 GPU 或模型依赖。
