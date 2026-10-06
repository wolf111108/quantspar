# Slim-Llama：FP8 / INT4 的条件时延估计

入口 scripts.profile_slimllama 单独采集，scripts.profile_bit_arches 默认一次推理同时采集 Bitlet、BitWave 和 Slim-Llama。全模型依赖和量化配置与既有 runner 相同，batch=1、2048 prefill＋256 次 decode forward。collector 不修改权重、激活、量化 scale 或数值 forward；这里只生成统计与分析时延，完整 14B/CUDA 由本地运行。

~~~bash
python -m scripts.profile_bit_arches \
  --model-path /path/to/Qwen2.5-14B \
  --output-dir outputs/bit_arches_2048_256
~~~

匹配已有量化配置的 scales 可加 --skip-calibration。独立入口将同一默认 YAML 中的 Slim-Llama 段交给 manager，不启用另两个 collector。显式选择 --architectures slimllama 也可单独运行。

## 论文参数与实现假设

来源：[ISSCC 2025 Slim-Llama](https://doi.org/10.1109/ISSCC49661.2025.10904761)，Fig.23.9.2/3/4/7。默认使用用户提供的 Asyn-CIM 论文 Table V 中的 50 MHz 对比工作点，不使用 25 MHz 下的 4.69 mW 或 200 MHz 峰值吞吐推导这个工作点的时延。

| 项目 | 默认值与依据 |
|---|---|
| SBC | 8 clusters × 8 SBC = 64 |
| 每个 SBC | 8 BMM columns；每 column 8 S-LUT |
| S-LUT | 8 × 7-bit register file，2 个读口 |
| 时钟 | 50 MHz，与 Asyn-CIM Table V 一致；原论文范围 25–200 MHz |
| 片上 SRAM | 论文写 500 KB，代码明确按 500×1024=512000 bytes 解释 |
| 外部带宽 | 默认 null：50 MHz 下未确认；原文仅报告 1.6 GB/s @200 MHz |
| 内部 SRAM/NoC 带宽 | 未公开；SRAM 带宽默认 null，可显式提供 |
| 激活 | 原文 INT4/8/16；SBC 以 4-bit 单元结合 aggregation |
| 权重 | Fig.23.9.7：INT1–16 或 ternary；当前量化框架/collector 使用 INT2–16，默认 INT4 |
| 聚类数量 | 128，来自原文 binary/ternary 基准；不意味着 Qwen W4 的最优值 |
| 输出/累加存储 | 2/4 bytes，显式假设 |
| LUT 初始化、切换、group setup | 原文没有周期数，默认 0 占位；可在 YAML 设置 |
| 中心结果存储/输出复用吞吐 | 默认 512 vectors/cycle，按 column 数作分析假设，可配置 |

INT4 位于论文支持范围内；原文没有 FP8×INT4 Qwen 实测，不能将 4.92–13.1 TOPS 或 Llama3B 的归一化时延当作这一 workload 的实测性能。

本次只调整默认硬件时钟及带宽缺失处理，不改变数值 forward、校准、量化 scale、采样、聚类、周期或字节计算。同样操作数下，50 MHz 计算时间为旧 200 MHz 的四倍；旧结果保留其原配置，不会自动变成新工作点。200 MHz 峰值、25 MHz 功耗及 50 MHz benchmark 不可混作同一工作点。

## 静态权重输出复用

每个 Linear 的量化权重按输出向量聚类：W_i = W_center(i) + delta_i，输出 A·W_i = A·W_center(i) + A·delta_i。中心输出对同一激活行只计算一次，随后计算差量，读取对应中心输出并相加。当前 runner 要求 weight_scale_granularity=scalar：共享 scale 下整数代码的分解才直接对应实际权重。channel scale 需要进一步推导，暂时拒绝。

短论文没有完整给出聚类算法。本实现明确采用独立的可复现近似：从 K 随机选至多 64 个坐标，从 N 无放回选至多 128 个实际向量作为中心，再按 feature 上相同代码数量的 Hamming distance 分配全部向量，平局选最低 ID；中心指向自身。assignment 和实际中心索引有 SHA256，可修改 clustering_features、weight_clusters 和 seed 做敏感性实验。

聚类 feature 只影响分组质量，不改变数值。每个采集 tile 都用完整原代码计算差量；INT4 [-8,7] 的差量范围 [-15,15]，用 INT5 验证并保留四个 magnitude bits 与符号，绝不裁回 INT4。原文 67.4% binary / 62.3% ternary 的稀疏率没有写入模型。

weight_preprocessing 按层导出 feature 的差量零比例、两种 residual mode 的向量数量及 host 预处理耗时。该比例不是全权重稀疏率；trace 的 observed_B_coefficients / observed_B_zero_coefficients 来自真正 center/delta tile 的样本，也不能当作全量数量。需要全量周期时用 --exact，聚类仍按配置的 feature 方法。

中心和 assignment 首次遇到该层时创建，后续 prefill/decode 复用，缓存不保留全量权重/残差副本。权重假定在采集期间不变，每次调用以 64 个分散坐标做变化探测；探测不等于完整 hash，改变权重或 scale 后应使用新 manager。host 预处理耗时只作诊断，不加入硬件推理时延。

## FP8 与 S-LUT 周期

FP8 E4M3FN/E5M2 先按精确量子 2^-9/2^-16 转为整数，再移除同一 S-LUT 组的共同 trailing zero，保留共同 power-of-two scale。保留 hidden one 和指数范围；不把 FP8 原始编码直接当作整数，也不截断。

能以 signed INT4 表示的组使用一个 activation pass，包括 -8；更宽的 A 用 signed 三位 magnitude 分片（每片 ±7）处理，再按位移和符号合并。这是一个保守的 FP8 扩展，不等同于原生 INT8/INT16 的最优 bit 分配。B 逐 magnitude plane 计算，符号控制加减；宽 FP8 K/V 保留全部对齐位。转换、指数/分片累加的额外逻辑及停顿未公开，需要单独成本。

Buffer mode 使用 8 个原始 activation registers，每 cycle 读取至多两个非零权重对应操作数，group work = ceil(nnz/2)。全零 B plane 跳过；全零 A 不额外免费跳过。

Mixed mode 对应 Fig.23.9.4 的 50/50 register 空间：四个 register 存三激活的 LUT 结果，另外四个存原始激活，所以一个加载组对应 7 个 K 位置。前三个权重都非零时，LUT 一次读取完成三项，再按两项/cycle 处理后四个 Buffer 操作数；前三个不全非零时切换为完整 Buffer，处理七项，并加配置的 mode_switch_cycles。只要任一 plane 使用 LUT，就每 activation pass 加一次初始 lut_setup_cycles；切换后需要重建 LUT 的成本须包含在 mode_switch_cycles 的假设中。这里假设各 plane 的 sign/shift 能正确组合，切换/控制粒度仍需 RTL 或作者细节确认。

中心使用 Mixed mode。非中心向量按 feature 差量零比例与 38% 阈值分流：高稀疏使用 8-position Buffer，其余使用 Mixed。每 SBC 八个 columns 映射 N，SBC 间分配 M/N，八个 S-LUT 映射 K 并行；每个 magnitude plane 以最慢的 M/N/S-LUT work 同步，再串行处理下一 plane 和 K tile。中心阶段先于 residual 阶段，额外加入中心结果 store 和输出 reuse/add 的假定周期。该同步布局及调度是分析选择，没有声称复现未公开控制器。

QK/PV 的 B 是动态 FP8 K/V，仅走 Buffer，不进行静态中心聚类或在线 similarity preprocessing。GQA 计算组织共享 KV，流量按物理 KV heads 计算历史 K/V 和新增 cache 写入。

原文 index reordering 主要针对 bit transitions/能耗，本实现不将其直接转换成免费周期收益，也不输出功耗估计。

## SRAM、片外流量与时延

片外保留原有紧凑 W4 / FP8 KV。预处理不是一个已经实现的 residual 压缩 kernel，程序不给未实现的差量压缩折扣。local_B 使用完整定点 magnitude、sign 和 FP8 metadata 的读量；local_A 包含额外 activation passes 和 FP8 metadata。

容量情景先保留扩展输入/权重 tile，再为一组激活行保留中心输出和输出列 tile 的 partial sums，计算 rows_per_weight_window。窗口中完成中心再处理 residual，中心输出不 spill；窗口不足时重新读取权重。每窗口额外读取中心系数和 cluster IDs，以从原始 W4 构造差量，并计入按 N tile 重读 FP8 A。这个固定策略可能比更好的编译调度慢；没有模拟 bank placement、双缓冲、所有寄存器转移或额外 spill。

| 输出时延字段 | 含义 |
|---|---|
| mapped_compute_seconds | center/delta/direct S-LUT 周期＋center store/reuse，除以频率 |
| minimum_traffic_full_overlap_seconds | max(compute, 最低原始字节/DRAM 带宽, 已提供的 local SRAM 时间) |
| capacity_window_no_overlap_seconds | compute＋容量窗口片外搬运＋已提供的 local SRAM 时间 |

未填片外带宽时，只导出周期、计算时间、流量和位宽诊断；per_phase_seconds、gemm_and_io_seconds 和 conditional_e2e_seconds 为 null，missing_costs 记录缺口。提供片外带宽后，前者采用乐观最低流量与重叠，不保证能在 500 KB 内实现；后者是所选容量窗口策略且无重叠，未保证包围真实硬件结果。没有内部 SRAM 带宽时，该项不加入数字，missing_costs 保留此缺口；不能把 null 理解为硬件无停顿。

使用 --slimllama-dram-bandwidth-gbps <value> 指定所选工作点的片外速率，使用 --slimllama-sram-bandwidth-gbps <value> 指定内部带宽，单位均为十进制 GB/s。50 MHz 下的速率必须来自明确来源或标为系统假设，不自动按时钟缩放，也不默认借用 200 MHz 的 1.6 GB/s。若要复现之前的 200 MHz 情景，在 YAML 副本的 slimllama 段显式设置 frequency_hz: 200000000.0 和 dram_bytes_per_second: 1600000000.0，并用 --config 选择该副本。

## 导出和剩余成本

slimllama_summary.json 包含独立的 phase/layer/step 周期、流量、mode/位宽观测、聚类与容量情景；slimllama_trace.jsonl 保留逐调用和分 stage 数据。bit_arch_comparison.json 收集所有选择的架构。dense baseline 是无 output reuse、所有 magnitude planes 占满的同布局工作；计算倍率可小于 1，不可当作三种硬件的 E2E 比率。

完整 E2E 仍要补 LM head、Softmax、RMSNorm、RoPE、SiLU 等剩余算子。可以沿用此前与 LLMCompass 相同的假设，并明确补入 Slim-Llama 格式转换、差量形成、累加和控制成本；不要用 host 统计脚本的墙钟时间代替硬件周期。未提供剩余成本时 conditional_e2e_seconds=null。

~~~bash
python -m scripts.estimate_bitlet_latency \
  --summary outputs/bit_arches_2048_256/slimllama_summary.json \
  --other-latency-json outputs/slimllama_other_latency.json \
  --output outputs/slimllama_e2e.json
~~~

剩余成本 schema 与 [Bitlet 说明](bitlet.md) 相同：schema_version=1、includes_lm_head=true、prefill_seconds 和覆盖每次 decode forward 的 decode_step_seconds。联合 --other-latency-json 表示三者共用剩余成本假设；架构专属成本可在采集后分别合并成三份文件，再调用离线估计，无需重跑模型。
