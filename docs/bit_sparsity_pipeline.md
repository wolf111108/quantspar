# OPT / Qwen 量化后的编码比特稀疏度

当前 `0103_quant_pipeline_main.py` 支持 OPT、Qwen2/Qwen2.5 的校准、量化、PPL 和 teacher-forced prefill/decode。统计在 `quant_forward` 中接收真正用于乘法的量化 codes，不在校准时统计原始浮点张量。主入口只采集编码稀疏度，不计算硬件映射周期；其他架构入口继续使用各自的 collector。

## 运行

先安装 `requirements-model.txt`。默认 FP8/INT4 配置保留 Linear outlier 旁路，统计量化 normal 部分并计入 mask 零：

```bash
python 0103_quant_pipeline_main.py \
  --config config/qwen2_14b_linear_matmul_f8i4.yaml \
  --model-path /path/to/Qwen2.5-14B \
  --eval-flow all \
  --stats-output-dir outputs/qwen_f8i4 \
  --results-dir outputs/qwen_f8i4
```

`all` 同时评估 PPL 和 prefill/decode；`ppl` 只评估 PPL，`pd` 只采集 prefill/decode。三者都会先按 calibration policy 处理 scale。默认 FP8/INT4 YAML 仍是 64 个校准样本、8192-token 校准长度，profiling 为 2048＋64；只选择 `pd` 不会缩短校准。

七类 Linear 的默认 `outlier_ratio` 为 `0.0001`，QK/PV 默认为零；QK/PV 也支持显式设置正比例后采集同一口径。若需要严格无旁路实验，把所有 outlier 设零并使用独立 scale 目录。已有架构映射入口继续使用它们自己的无 outlier 配置。

原生 FP16 配置：

```bash
python 0103_quant_pipeline_main.py \
  --config config/opt_1.3b_linear_matmul_fp16.yaml \
  --model-path /path/to/opt-1.3b \
  --eval-flow all \
  --stats-output-dir outputs/opt_fp16 \
  --results-dir outputs/opt_fp16
```

Qwen 使用 `config/qwen2_14b_linear_matmul_fp16.yaml` 和相应 checkpoint。两份新 FP16 YAML 显式设置 `model.dtype: fp16`、所有量化操作数及输出为 FP16、outlier 为零、独立的 native-v2 scale 目录。一个校准 forward 用于初始化和保存全为 1 的 scale，不进行动态范围拟合。OPT 示例的 1984＋64 上下文不超过 2048。

unit 默认关闭：统计器、CLI 和默认 YAML 都关闭，旧 YAML 的 `unit_sparsity.enabled: true` 也不会在 prefill/decode 阶段重新开启。保留 `--unit-sparsity` 作为显式选择。

## 输出和统计口径

每次运行导出以下文件，前缀默认为配置文件名，也可通过 `--run-name` 指定：

- `<run>_full_forward_bit_sparsity.json/.csv`：PPL full-forward 的统计。
- `<run>_prefill_decode_bit_sparsity.json/.csv`：prefill 和 decode 分别统计。
- `<run>_<timestamp>.txt`：PPL 结果和两个流程的比特稀疏度快照。

未指定 `--stats-output-dir` 时，JSON/CSV 也会保存到 `--results-dir`。CSV 按 phase、层名、层号、操作数、格式及 `outlier_masked` 分行；JSON 额外包含各 phase/操作数/mask 口径及全体计数的汇总。PPL 数据在进入 prefill/decode、清空统计器之前导出。

| 操作数 | 来源 | 计数频率 |
|---|---|---|
| activation | Linear 输入 A | 每次量化 forward |
| weight | 无 outlier 的 Linear 静态 W | 每层、每阶段一次 |
| weight（outlier_masked=true） | Linear 中随当前输入 mask 改变的 normal W | 每次量化 forward，禁用静态计数缓存 |
| Q / K | QK MatMul 的 A / B | 每次量化 forward |
| attention_probs / V | PV MatMul 的 A / B | 每次量化 forward |

K/V 统计描述参与 attention 的量化操作数。decode 延续已有 GQA packing，共享的 K/V 不按 query head 重复统计；prefill 按实际展开后的计算操作数统计。历史 cache 会在后续 decode 中反复参与计算，因此动态计数不是“缓存中每个存储元素只计一次”。全体汇总是上述计数频率下的混合操作数比例，比较模型时应优先看各 phase/操作数行。它不是全模型唯一参数的存储稀疏度。

outlier 统计保留 normal 张量原始形状，protected 位置置零后参与统计，分母包含这些位置；不筛除 mask 零，也不额外统计高精度 protected 操作数。四项数值运算照常执行，量化 normal A/W/B 各计一次，不因在交叉项中复用而重复计数。JSON schema 版本为 2，`outlier_sparsity` 明确记录 `mask_generated_zeros: included`、`high_precision_sidepath_counted: false`；各行还有 `outlier_masked` 和 `counting`，TXT 也标注范围。

直接调用模块时使用 `QuantStatManager(..., collect_mapping=False)` 即可收集 outlier 编码稀疏度。映射或 EBB/Bitlet/BitWave/Slim-Llama collector 仍拒绝 outlier，原因是旁路及交叉项的硬件成本尚未建模；普通主入口已配置为纯稀疏度采集。

| 格式 | 每个元素参与统计的位 | 编码 |
|---|---:|---|
| INT4 / INT8 / INT16 | 4 / 8 / 16 | 完整补码，包括符号位 |
| FP8 E4M3FN | 3 | 显式尾数 |
| FP8 E5M2 | 2 | 显式尾数 |
| FP16 E5M10 | 10 | 显式尾数 |
| BF16 E8M7 | 7 | 显式尾数 |

FP 的 activation、weight、Q、K、attention_probs、V 全部使用相同 `mantissa` 口径，不计符号、指数或 hidden one。INT 的比例使用补码，不使用绝对值 magnitude；例如 INT4 的 -8 编码为 `1000`，零比特比例为 3/4。

比特稀疏度定义为：

```text
bit_zero_ratio = sum(zero_bits) / sum(counted_bits)
```

汇总先累加分子和分母，不对每层百分比直接取平均。空统计的比例为 JSON `null`。元素零值比例单独保留：FP 的 1.0 是非零值，但其显式尾数全零，所以尾数稀疏度可以为 100%。这不代表完整 FP 运算没有成本。

范围是包装后的 Linear/attention 乘法输入；不额外采集 output、bias、embedding、norm、未启用的 LM head。数值仿真仍使用浮点张量和运算表达量化网格，没有实现原生 INT4 打包 kernel，也不把该稀疏度直接解释为物理加速比。

## FP16 修复

原来 `fp16` 被解析为禁用量化；加载器又优先选择 BF16。因此 YAML 写 FP16 并不能保证模型使用 FP16，统计器将 BF16 值重新解释为 FP16 也无法恢复已经丢掉的精度。

现在完整 FP16 配置自动加载为 FP16，显式指定相冲突的 `model.dtype` 会报错。A/W/B/O 在相应量化步骤执行原生 FP16 网格舍入，FP16 scale 恒为 1，有限越界值饱和到 ±65504。核心 Linear/MatMul 仍以 FP32 表达 codes 和累加；输出按 FP16 配置再舍入，不能将它描述为已验证的原生 half GEMM kernel。

旧 FP16 YAML 如仍指向 INT8 scale 目录，应改为独立目录并重新校准。复用非 1 的 FP16 scale 会明确报错；不可用已有 INT8 scale 乘除来定义所谓“原生 FP16”。

## 默认 FP8/INT4 配置的问题

原 `config/qwen2_14b_linear_matmul_f8i4.yaml` 的格式表是：

| 算子 | 输入 A | 输入 W/B | 输出 O |
|---|---|---|---|
| 七类 Linear 投影 | FP8 E4M3FN | INT4 | FP8 E4M3FN |
| QK MatMul | FP8 Q | FP8 K | FP8 |
| PV MatMul | FP8 attention_probs | FP8 V | FP8 |

因此 INT4 只用于 Linear 权重；MatMul 的 B 是动态 K/V，不应仅因为名称 W4 就改成 INT4。输出也配置了 FP8，并非只有输入和权重参与量化。`d_bit` 是旧映射参数，不定义额外的量化操作数，也不决定尾数统计位宽。

默认配置未启用 `kv_cache.fp8_static`；K/V 在 MatMul 读取时量化，HF cache 仍是模型 dtype 的浮点张量。K/V 的尾数统计不能当作已实现 FP8 cache 打包存储的证明。另需注意 YAML 的数字 `8` 表示 INT8，FP8 要写 `e4m3` 或 `e5m2`。

默认七类 Linear 的 `outlier_ratio: 0.0001` 都为正。激活按每个输入通道跨 token 的最大绝对值排序，权重按每个输入通道跨输出行的最大绝对值排序，两侧各选 `max(1, int(H * ratio))` 个通道，再取并集，作为激活和权重共同的输入通道 mask。MatMul 同样在 A 的最后一维和 B 的倒数第二维（乘法归约维）分别选通道、取并集、同时保护两侧；输出仍单独保护高幅值输出通道。例如 H=5120 且 ratio=0.0001 时，两侧各选 1 个通道，重合时共保护 1 个、不重合时共保护 2 个，实际比例约为 0.0195%–0.0391%。不再额外按权重元素 top-k 生成 mask。

该旁路将数据分为 normal 和 protected 两部分，并以四项相加恢复结果：

```text
x_q @ w_q + x_f @ w_f + x_f @ w_q + x_q @ w_f
```

`f` 部分保留原数据的精度，再用 FP32 表达各项计算；代码注释中的“FP16”不意味着它一定由 FP16 加载，加载器也可能使用 BF16。正常部分被 mask 掉的零不是自然产生的量化零，可能抬高统计稀疏度。当前按用户指定口径接受这些零，统计的是含 mask 零的 normal 操作数；高精度旁路的编码与硬件成本不进入该比例。

当前恢复七类 Linear 的 outlier 为 `0.0001`，显式关闭 mixed precision，并使用 `f8i4_outlier_v3` scale 目录。此前关闭 outlier 的 strict-v2 scale，不能直接作为带旁路的 normal 范围；INT8 和 FP8/INT4 的量化网格及范围也不同。旧的“复制 INT8 scale”建议不适用，需重新校准。

旁路同时修复了旧数值实现：normal codes 以 FP32 保存，乘实际 scale 反量化后计算四项，bias 在真实值域加一次；输出 normal 部分按实际 O scale 量化/还原，高精度输出通道保留。移除了固定 16/24/48-bit 移位的近似缩放与除法前转 half，避免小 scale 截零、half 溢出或 INT16 code 失真。Linear 校准与推理共用维度保持的 mask，支持 channel W scale；MatMul 的 B 始终沿 K 轴（倒数第二维）保护，修复方阵时误选最后一维的问题。

原 YAML 的 `model.num_layers: 28` 是未参与 wrapper 遍历的元数据，已移除以免误以为只统计 28 层；wrapper 遍历 checkpoint 中的真实层。

## 验证与旧结果影响

测试使用真实 PyTorch 2.6.0+cpu、Transformers 4.43.1。保存并加载本地小型 OPT/Qwen，分别跑 INT8、FP8/INT4、FP16 的无 outlier 与有 outlier 两种实际校准、PPL 与 prefill/decode，验证 CSV/JSON、位宽、phase 隔离、mask 权重每次重计和 unit 默认关闭；测试数据本地生成，没有下载真实 checkpoint 或数据集。另用独立原始编码参考检查 FP 子正规数、尾数零位及 INT4 负值；旁路参考覆盖 Linear 形状/channel scales/bias/输出格式、方阵及批量 MatMul、极小 scale、half 输入 INT16 codes、mask 零和架构采集拒绝。

旧结果不能直接与当前结果拼接：此前 FP 权重/KV 从 storage 改为 mantissa，FP8 每元素分母从 8 改为 3（E4M3），FP16 从 16 改为 10；全局计数新增 W/KV，比例分母和 FP16 数值语义改变。当前又恢复默认 outlier，改为包含 mask 零的 normal 统计，masked W 每次重计，且修复旁路数值。旧 strict-v2 或旧旁路的 PPL/比例与新结果不可混称同口径，需重新校准并采集。无 outlier 操作数的数值、位统计及去重行为保持不变。JSON/CSV 新增范围字段，读取旧 schema 的脚本需兼容版本 2。硬件 collector 的独立对齐/调度口径不以本文件的尾数统计替代。

完整 OPT/Qwen checkpoint 的 PPL、14B、CUDA、真实数据集和 GPU 内存开销尚未执行验证。缺失的 BitNet/Qwen3.5/MoE 模块导入暂时注释；本快照不提供这些模型的包装器。
