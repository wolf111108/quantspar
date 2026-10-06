# IA FP8 / W INT4 通路修复与使用

## 审计结论与修复

基线为 main `ca07eb0`，只有 5 个源文件。原始通路不能直接作为可信的 FP8/INT4 量化与架构倍率统计流程。

| 问题 | 原来的后果 | 本次处理 |
|---|---|---|
| 缺少 quant_spec、utils、包入口；无用硬件代理/causal 模块强制导入 | 核心模块也无法导入 | 补全核心依赖；硬件代理改为按需导入，移除未使用导入 |
| Linear/MatMul 默认开启混精 | 指定 FP8 仍可能执行 FP16/FP8/FP4 | 默认关闭；显式选择才开启混精 |
| Linear 关闭输出量化时回退原始 F.linear | IA/W 的量化误差被绕过 | 无论输出是否量化，都使用量化后操作数；bias 在真实数值域加一次 |
| 固定格式动态分支缺少 collector 的权重参数 | 运行时参数错误，不能统计 | 使用统一签名，传真实 FP8/INT4 codes |
| per-output-channel 选项未应用于 FP8/W4 通路 | 配置与实际 W scale 不符 | 校准保存通道 scale，按 K 维广播量化权重 |
| FP 统计混入额外指数计数；BF16 编码误用 FP16 | 位分母不一致，稀疏率失真 | 从实际编码提取所选位；元素零与编码零位分开 |
| INT4 仅保留 3 个 magnitude 位 | -8 可能被误判成零位 | INT4 存储统计使用完整 4-bit 补码 |
| 统计调用被注释、权重统计关闭 | 普通路径没有映射周期和 W 稀疏记录 | 两阶段都记录；静态 W 每阶段计一次，动态 B 每调用计一次 |
| prefill 用 INT8 bits，却用 3-bit FP8 dense 基准 | 把格式差异当成速度差异 | 两阶段使用实际 FP8 编码 |
| 小 K dense 高度固定 64 | 无位稀疏也可能得到 8x | dense/sparse 使用完全相同的有效元素与映射调度 |
| padding reshape 直接交换 token/K-round 维度 | token 和 K slice 混排 | 先按真实 token/K 顺序 reshape，再 permute 调度轴 |
| 独立 head/batch 先累加再取全局 max | 不同权重 operand 被错误跨 head 异步合并 | 每个独立 operand 取 macro 最大值，再串行相加 |
| as_l 不影响动态映射；几何调用使用默认值 | 配置与统计不一致 | 显式传入几何和同步模式 |
| 全零统计直接除周期、旧脚本打印未采集算子倍率 | 除零或 Infinity | 零成本记为 null；输出有限 JSON；导入不支持的零成本倍率时报错 |

本次测试使用真实 PyTorch 2.6.0+cpu，不使用 Torch 导入占位。19 项测试覆盖静态/动态、scalar/channel W scale、bias/输出量化、编码边界、权重计数、两个阶段、映射布局与独立标量调度器。混精数值路径、全模型包装器、GPU 和 Qwen checkpoint/PPL 不在已完成验证范围内。

## 严格 FP8/INT4 配置

```python
layer = QuantizedLinear(
    in_features=5120, out_features=5120,
    a_bit="e4m3", w_bit=4, o_bit="none",
    mixed_precision=False, outlier_ratio=0.0,
    dynamic_activation=False,
    weight_scale_granularity="scalar",
    mode="scale_inspection", scale_root_str="scales/run1",
)
```

INT4 codes 为 `[-8,7]`，对称 scale 使用 `max_abs/7.5`。FP8 先按 scale 归一化，饱和至 ±448，再转换到原生 E4M3FN 网格。`quant_awo` 返回 **codes**，调用者乘 scale 还原真实数值后完成 Linear/MatMul；不是直接返回反量化数据。

调用顺序是校准 → `save_scales()` → `mode="quant_forward"`。开启 `dynamic_activation=True` 时每 token 在线计算 IA scale；W 和输出 scale 仍校准。`output_channel` 将 W scale 按输出通道分别校准。改变格式、granularity 或混精策略后需要重新校准，不能直接复用旧实验 scale。

输出 `o_bit="none"` 表示输出不再量化，IA/W 仍量化。`"fp16"`/`"bf16"` 表示不增加输出量化，由输入/模型 dtype 决定实际返回 dtype。QK/PV 的动态 B 是 K/V 数据，不是 Linear 权重：通常 A/B 都设为 FP8，不应仅因 Linear W4 就把 K/V 改成 INT4。

## 稀疏统计口径

```python
stats = QuantStatManager(
    "scales/run1", nmacro=16, h=64, w=48, banks=16,
    as_l=1, bit_scope="mantissa", cycles_per_effective_bit=1,
)
stats.set_phase("prefill")
y = layer(x, stat_collector=stats)
stats.set_phase("decode")
y = layer(x[:, :1, :], stat_collector=stats)
stats.export_cim_stats("outputs/run1/cim.json")
```

| bit_scope | FP8 统计位 | 说明 |
|---|---:|---|
| mantissa（默认） | 3：0MMM | 与此前 quantspar 的 explicit-mantissa 口径对应 |
| sign_mantissa | 4：SMMM | 加上符号位，dense 计算和统计同时使用 4 位 |
| storage | 8：SEEEEMMM | 原始存储编码统计；不能自动解释为 8 个硬件串行周期 |

INT4 的存储稀疏度始终使用完整 4 位补码。元素级零值看实际 code 是否为零；非零值的尾数可以全零，例如 1.0。符号/存储口径保留有符号零的实际 sign 位。unit 稀疏使用同一编码和所选位宽，尾部 unit 补零，不删除尾部元素。

映射不会用 `1/(1-bit_sparsity)` 替代真实负载：先取 bank 最大工作量，再按同步/异步规则聚合 macro。Linear 共享一组 W，batch/token 合并为 M；attention 的独立 B/H operand 串行执行。统计分块处理 operand/token-round，避免对完整 attention tensor 生成额外的大型 bit 轴。

无可跳过位时，同一映射下的 sparse/dense 周期相等。例如 `K=128,M=16,N=4096` 全部 3 个尾数位为 1，dense 和 sparse 都是 2064 effective-bit steps，倍率 1；旧基准曾给出 8x。

## LLMCompass 对接

`cim_records` 每条记录包括形状、几何、布局、dense/sparse steps、实际所选位稀疏度、W 编码计数和 phase。

- `speedup`：相同有效元素与调度下的 `dense_steps/sparse_steps`，用来报告计算稀疏收益。
- `llmcompass_speedup`：当前 LLMCompass 的容量分母 / 此处映射 sparse steps，用来还原相同计算周期。它包含容量分母换算，不应直接当作论文中的纯稀疏加速比。

完整采集 Qwen 的 9 个 GEMM 后，可调用：

```python
stats.export_llmcompass_manifest(
    "outputs/run1/llmcompass_speedups.json",
    workload={
        "d_model": 5120, "ffn_dim": 13824, "q_heads": 40, "kv_heads": 8,
        "batch_size": 1, "shared_kv_gqa": False,
        "prefill_lengths": [8192],
        "decode_cache_lengths": list(range(8192, 9215)),
    },
    source_commit="实际采集版本的 git SHA",
)
```

该例按生成 1024 token 的约定，prefill 产生首 token，因此需要 1023 次 decode。若统计脚本实际执行了 1024 次 teacher-forced decode，则上下文列表应反映实际调用，不要用请求约定覆盖采集事实。例子不表示已经采集过这些张量。

导出使用 `baseline="effective"`，聚合 bridge ratio；要求两阶段的 9 个算子齐全、每阶段位宽一致、投影形状与模型配置匹配、观测上下文匹配及 GQA 分组匹配。缺失数据、同步映射或无法表示为有限倍率的零成本会报错。每 phase/operator 仍只有一个加权平均值，不能替代 per-layer/context 的完整轨迹。

LLMCompass 命令使用 `--speedups-json` 导入生成文件；采样上下文列表必须和 manifest 一致。旧 `source` 倍率和新 `effective` 文件不能混用。W INT4 的存储/更新参数还需要按真实硬件设置：当前 LLMCompass 默认 weight bytes 不能自动从本文件推导。仅在实际实现 4-bit 紧凑存储时，off-chip weight bytes 才应设为 0.5；本地阵列写入位宽按物理实现配置。

## 单工作负载脚本

```bash
python -m scripts.profile_fp8_int4 \
  --tensor-file /path/to/linear_workload.pt \
  --phase both --nmacro 16 --bit-scope sign_mantissa \
  --dynamic-activation --weight-scale-granularity output_channel \
  --output outputs/run1/linear_stats.json
```

输入字典：`activation` 为 `[M,K]` 或 `[B,M,K]`，`weight` 为 `[N,K]`，`bias` 可选。`both` 的 decode 是输入的首 token，用于接口核对；真实 decode 数据应单独提供并使用 `--phase decode`。这是 CPU 数值/统计脚本，不保存输入张量，不测硬件延迟。

## 仍需补全的范围

- 公开快照缺少 `quant.model_wrapper`、`quant.qwen_wrapper`、`others.data/evaluation`。旧全模型入口会给出明确依赖错误；它的帮助、阶段 JSON 导出和除零输出已修复，但完整推理依然需要原项目包装器/数据模块。本次没有将旧附件包装器当作当前实现。
- 全模型层覆盖、GQA/KV cache 接入、Qwen checkpoint 精度/PPL、CUDA 数值和运行内存尚未验证。
- explicit-mantissa 模型不包含 hidden-one、指数对齐及固定控制成本。1.0 的尾数是零，可能得到零“所选位成本”，但 FP8 乘法仍有实际工作；需要由硬件模型补充这些成本，不能据此声称无限物理加速。
- W 稀疏度单独统计，不默认乘入 activation 稀疏加速比。权重位串行、零权重跳过、权重写入和存储压缩必须按目标架构另外建模。
- 已启用 outlier sidepath 的计算不属于严格 FP8/INT4；统计器拒绝省略其成本的映射倍率采集。该数值 sidepath 和混精完整流程不在本次认证范围内。
