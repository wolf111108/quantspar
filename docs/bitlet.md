# Bitlet：Qwen2.5-14B FP8 / INT4 的统计与时延估计

需要一次推理同时收集 Bitlet 和 BitWave 时，使用 [双后端入口](bitlet_bitwave.md)。本页中的单独 Bitlet 命令与默认映射保留。

入口为 scripts.profile_bitlet，默认收集 batch=1、2048 prefill＋256 decode 的实际量化操作数。Linear 使用 IA FP8 E4M3FN、W INT4；QK/PV 两侧均为 FP8，K/V 在写 cache 前量化。关闭 outlier 和混精，使用独立的校准 scale 目录。

输出是 Bitlet BCE 列负载和带搬运假设的 GEMM 时延估计。FP8×INT4 是对论文浮点 BCE 的分析扩展，论文没有测量这个混合格式的 Qwen 性能；运行脚本的 CPU/GPU 墙钟时间也不是 Bitlet 时延。

## 论文参数与未公开数据

来源：[MICRO 2021 原论文](https://luhang-hpu.github.io/files/bitlet-MICRO21.pdf)的 Sec.4.2、Sec.5.5、Table 3 和 Fig.5，以及 [TCAD 2024 扩展论文](https://luhang-hpu.github.io/files/bitlet-TCAD24.pdf)。

| 项目 | 默认值 | 口径 |
|---|---:|---|
| PE 数 | 32 | 原论文 Table 3 |
| 每个 BCE 的元素组大小 N | 64 | 原论文默认配置；这里沿 GEMM 的 K 维分组 |
| 时钟 | 1 GHz | 原论文 Table 3 |
| 浮点 significand bit lanes | 24 | 含 FP32 hidden one |
| Activation DMA | 12.8 GB/s | 独立通道 |
| Weight/B DMA | 12.8 GB/s | 独立通道，不能把两条合成一条 W 通道 |
| Local buffer 到 PE 的总带宽 | 25.6 GB/s | 原论文描述的合计读带宽 |
| Local buffer 容量 | null | 两篇论文均未报告容量 |

GB/s 使用十进制。容量不能从另一架构 BitL 的 128 KB 借用，也不能由带宽反推。若有实现信息，可填写 bitlet.local_buffer_bytes；当前仅检查能否容纳一个按量化格式打包的最小输入/输出 tile，不会因此验证模型的常驻、partial sum 或双缓冲可行性。

output_storage_bytes=2、group_setup_cycles=0 是本项目显式建模假设，不是论文公布的硬件参数。前者按 FP16/BF16 中间结果计流量；后者可在已知额外组调度成本时修改。没有用论文峰值 TOPS 除整个模型工作量。

## 本地运行

在仓库根目录安装完整模型依赖后运行：

~~~bash
python -m pip install -r requirements-model.txt
python -m scripts.profile_bitlet \
  --config config/qwen2_14b_bitlet_f8i4_2048_256.yaml \
  --model-path /path/to/Qwen2.5-14B \
  --output-dir outputs/bitlet_2048_256
~~~

默认 CUDA、teacher-forced decode、64 个校准样本，每个 2048 tokens。output-dir 必须是新目录。需要先检查本地模型、依赖与显存时：

~~~bash
python -m scripts.profile_bitlet \
  --model-path /path/to/Qwen2.5-14B \
  --prefill-length 256 --decode-steps 8 --calibration-samples 4 \
  --scale-dir quant/scales/bitlet_smoke/ \
  --output-dir outputs/bitlet_smoke
~~~

缩短 prefill 会默认限制校准长度；可用 --calibration-length 显式设置。正式实验应使用正式长度和独立 scale 目录重新校准。已有 scales 不会因请求长度变化自动重算；run_config 记录本次请求和复用情况，不代表旧 scales 的生成历史。确认 scales 与量化配置匹配后可用 --skip-calibration。

--token-file 支持 torch.save 保存的一维 LongTensor 或含 input_ids 的字典，替代 profiling 文本数据下载；首次校准仍使用 YAML 中的数据集。--greedy-decode 使用 LM head 生成后续输入；默认模式使用文档中紧随 prompt 的 tokens，适合固定长度比较。贪心模式中执行的 LM head 耗时也不自动计入 Bitlet 估计。

这里的 256 decode 指 256 次单 token forward，缓存历史长度从 2048 增至 2303。若目标是恰好生成 256 个输出 tokens，prefill 已给出第一个 token 的 logits，通常只需要 255 次 decode；需要按最终实验定义调整。

## 收集内容与映射

每层独立收集七个 Linear 和 QK、PV 两个 attention GEMM：

1. 验证真实 FP8/INT4 网格和编码，记录完整存储位的 one-bit 数，包括 INT4 的补码。
2. 浮点/混合格式把量化代码值提升为论文 FP32 BCE 的 normalized significand/exponent 表示，保留 hidden one。EA＋EB 的组内最大值作为 Emax，按 Emax−Ei 对 B significand 右移，放入 24 bit lanes。
3. 每个 lane 每周期取一个有效 B bit；组周期为各列 population 的最大值。符号决定加减，不作为可跳过的 mantissa lane。纯整数模式绕过 exponent alignment，统计 B magnitude bits。
4. 记录超过 24-lane 窗口的截断、产品指数跨度、零产品和负产品等诊断。截断只影响这份硬件工作估计，模型数值 forward 不做对应截断；本项目没有据此验证精度。

Bitlet 在 B/weight 一侧做 bit interleaving。不能把 A 中的零值或零位直接换成免费跳过：即使 A 的整组全零，B 仍可能消耗 BCE 周期。这与 Asyn-CIM 的 activation 跳过口径不同，尤其影响有 causal 零区域的 PV。

布局将同一个 A 的 K-group 广播给至多 32 个输出列 PE，tile 周期为这些 PE 的最大组周期；不同 batch/KV-head 操作数顺序调度。同步 tile 的布局是分析假设，不声称复现未公开的控制器。还输出相同几何的 dense_tiles 和 ideal_pe_balance；后者是理想负载平衡的计算参考，不是另一项可直接相加的成本。

GQA 默认共享 K/V：Qwen2.5-14B 的 40 个 query heads 合并到 8 个独立 KV-head 操作数中，计算保留所有 query 的工作量，片外 cache 读量按实际 8 个 KV heads 计算。

## 默认抽样与精确模式

完整 14B 的逐元素、逐组统计非常昂贵。默认每个独立操作数/每次 GEMM 抽样 64 个 prefill waves 或 8 个 decode waves；一个 wave 包含至多 32 个 PE、各 64 个 K 元素。样本是可复现的均匀有放回抽样，seed 由配置、phase、operator、layer 和 decode step 共同确定。

周期估计用精确已知的 dense 工作量减去抽样得到的平均节省量，正确处理 K/N 的尾组；抽样得到的负周期会截为零，并保留截断前值。输出抽样标准误差和近似 95% 区间，单个 observation 不给区间。小样本正态近似不能保证覆盖率，也不反映硬件建模误差；若区间过宽或点估计触及边界，应增加样本。observed 与 histograms 只包含实际观察的样本，不能当作全张量数量。

~~~bash
# 增加统计样本；不会更改量化数值 forward
python -m scripts.profile_bitlet \
  --model-path /path/to/Qwen2.5-14B \
  --prefill-sample-waves 256 --decode-sample-waves 32 \
  --output-dir outputs/bitlet_more_samples

# 枚举全部 waves；14B 下可能非常慢
python -m scripts.profile_bitlet \
  --model-path /path/to/Qwen2.5-14B \
  --exact --output-dir outputs/bitlet_exact
~~~

也可将 YAML 的某个 phase 的 sample_waves 设置为 0。chunk_waves 限制临时张量大小；更改 chunk 不改变抽样或结果。mapped_compute_speedup 和汇总的 mapped_compute_speedups 都是相同布局的 dense/mapped 计算周期比，不是 E2E 加速比。

## 搬运与时延场景

Linear 的 B 最少片外读量按打包 W4 计算；QK/PV 的历史 K/V 分别按 FP8 读一次，新增 cache 片段记写入。片上 A 按输出 tile 广播次数计流量，B 分成跨 M 行保留一次的 resident 假设与每行重读的 streaming 假设。输出与 cache 写入在分析模型中共用 activation DMA。片上读写是否共享同一物理端口、重排/转换和 partial-sum spill 需要实现信息。

对于每个 GEMM，输出两种情景后逐算子相加：

| 输出字段 | 单个 GEMM 的计算方式 |
|---|---|
| resident_full_overlap_seconds | max(compute, A DMA, B DMA, resident local traffic) |
| streaming_no_overlap_seconds | compute＋A DMA＋B DMA＋streaming local traffic |

前者依赖未验证的 B 复用和充分重叠，后者假设无重叠；二者是条件场景，不是保证包围真实系统时延的物理上下界。buffer 容量、控制停顿、格式/scale 转换等未公开成本仍可能改变结果。项目没有实现原生 W4 打包 kernel 或 Bitlet RTL。

## 输出与 E2E 补全

输出目录包含：

- run_config.json：实际 workload、校准/复用信息、依赖版本、Git commit、硬件参数。
- bitlet_trace.jsonl：每个算子调用的布局、抽样、列直方图、周期、流量和 phase/step/cache provenance；流式写入。
- bitlet_summary.json：按 phase/layer/step 聚合，GEMM/IO 情景总时延和平均 decode 每步成本。

48 层完整运行应有 432 个 prefill 调用、110592 个 decode 调用，共 111024 条 trace。入口检查每个 layer/operator 的覆盖次数及 cache 每步增长，默认每 32 步更新汇总；中断时输出 status=interrupted，不能当完整实验。

latency.gemm_and_io_seconds 是所收集九种 body GEMM 与搬运的总和。完整 E2E 还需要 LM head（它本身也是 GEMM）、Softmax、RMSNorm、RoPE、SiLU、residual，以及 scale/format conversion 等剩余成本。没有提供这些成本时，conditional_e2e_seconds 保持 null，不能把 GEMM/IO 字段直接标为 E2E。

剩余耗时文件采用以下 schema。此例只示意格式；prefill_seconds 必须替换为真实测量或另一个明确模型的估计，decode_step_seconds 必须提供恰好 256 项，并包含 LM head 等剩余算子：

~~~json
{
  "schema_version": 1,
  "includes_lm_head": true,
  "prefill_seconds": 0.0,
  "decode_step_seconds": []
}
~~~

采集完成后可以补这些成本，无需重新运行 14B：

~~~bash
python -m scripts.estimate_bitlet_latency \
  --summary outputs/bitlet_2048_256/bitlet_summary.json \
  --other-latency-json /path/to/remaining_latency.json \
  --output outputs/bitlet_2048_256/bitlet_e2e.json
~~~

也可在 profiling 时直接传 --other-latency-json。当前按串行方式相加：

$$
T_{\mathrm{E2E}}=T_{\mathrm{prefill,GEMM+IO}}+
\sum_{t=0}^{D-1}T_{\mathrm{decode},t,\mathrm{GEMM+IO}}+
T_{\mathrm{prefill,other}}+\sum_{t=0}^{D-1}T_{\mathrm{decode},t,\mathrm{other}}.
$$

补成本后的结果依然是条件估计，不能替代目标硬件实测。迁移 GPU 的剩余算子耗时时，应明确该系统确实用此 GPU 执行这些算子并计入传输；不能直接将任意 GPU 时间作为 Bitlet 控制器耗时。

## 验证与现有结果

CPU 测试覆盖独立 Python 标量列/调度参考、hidden one、符号/补码、产品指数对齐与截断、K/N tails、抽样复现/误差、双 DMA/GQA 流量、导出隔离、剩余成本补全和中断拒绝。真实 PyTorch/Transformers 下保存并加载小型 Qwen，实际校准，验证 teacher-forced/greedy CLI、九种算子覆盖与 cache 增长。

这里没有运行完整 Qwen2.5-14B、CUDA、PPL 或 RTL。旧 EBB、Asyn-CIM 公式和导出入口保留，Bitlet 使用独立统计/schema/scale 目录，旧稀疏倍率不能直接套用。Bitlet 不接入现有的 Asyn-CIM LLMCompass manifest；需要其他算子的成本时使用上述补全入口。
