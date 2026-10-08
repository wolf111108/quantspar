# 修改记录

每次提交在同一提交中增加一条，记录目的、内容、验证和限制。当前提交用与 commit message 一致的标题标识；提交前不填自身 SHA。规则见 [AGENTS.md](AGENTS.md)。

## 2026-10-08 — Add OPT/Qwen FineWeb sparsity and PPL rerun matrix

- 目的：按截图中 GPT-2 以外的三个模型、五种 A/W 格式创建可本地重跑的配置与脚本，重新采集比特稀疏度和 PPL，包括旧 NaN 项。
- 内容：新增 OPT-1.3B、OPT-6.7B、Qwen2.5-7B × BF16/BF16、E4M3FN/E4M3FN、INT8/INT8、INT8/INT4、E4M3FN/INT4 共 15 份 YAML；全部显式 BF16 加载、eager、关闭 mixed precision/unit，使用 FineWeb sample-10BT/train、64 校准样本/seed23、统一 65536 输入 token PPL 预算，OPT 长度 2048、Qwen 长度 8192。量化组 Linear outlier=0.0001，BF16 与 QK/PV 为零；K/V 跟随 A 格式。新增逐组子进程 Python/Shell 脚本、独立 attempt/scales、dry-run/子集/resume/outlier 覆盖、失败继续与退出码；主入口新增完整数值 PPL/阶段快照 JSON，脚本导出 summary JSON/CSV 与操作数原始计数 CSV，比例按分子/分母加权，NaN 保留 null/status。更新 README 和专门运行说明。
- 验证：新增 9 项无模型依赖 unittest 全部通过，覆盖 15 配置实际 wrapper 层名/格式/上下文/数据预算与 scale 隔离、dry-run 无模型调用/无文件写入、完整 PPL 数值、加权计数与非有限 JSON、失败后继续/空值、resume 指纹变化及文件缺失/错误计数/详细文件 SHA256 损坏重跑、中断状态、PPL-only 和跨工作目录 Shell 入口。Python compileall、Shell bash -n 与 diff 检查通过。
- 限制与旧结果影响：只使用合成报告和调度回调验证，没有在这里运行完整 checkpoint、FineWeb、CUDA 或真实 PPL，当前环境无 PyTorch/Transformers。FP 显式尾数、INT 补码、mask 零计入及 protected codes 排除口径保持不变；截图的指数对齐尾数脚注与本次不同，旧上传配置的 FP16/BF16、零 outlier、Qwen token 预算差异均在说明中明确，新结果不可直接混称旧设置。resume 校验路径/配置/代码/依赖，不计算 checkpoint 权重哈希；数值仿真不含原生低位 kernel 或硬件时延。既有 FP16 配置、默认 14B 配置与硬件 collector 未改。

## 2026-10-08 — Allow masked outlier bit sparsity in OPT/Qwen

- 目的：按用户指定口径，允许带 outlier sidepath 的量化推理采集 normal 操作数稀疏度，接受 mask 人为产生的零，继续统一 FP 显式尾数并默认关闭 unit。
- 内容：Linear/MatMul 将 full-shape masked normal codes 接入纯稀疏统计器，四项数值分支全部执行，normal 操作数各计一次，不额外计 protected 分支。带输入相关 mask 的 W 每次 forward 重新统计，关闭其静态计数缓存；无 outlier 的 W 仍每阶段一次。JSON schema 升为 2，CSV/JSON/TXT 新增 outlier_masked/counting 和 mask 零计入、protected 编码排除的范围说明，masked/unmasked 汇总分开。默认七类 Linear 恢复 outlier_ratio=0.0001，QK/PV 保持零且代码支持正比例，独立 outlier-v3 scales 重新校准。旁路改为 FP32 codes 乘实际 scale 的真实值域四项计算、bias 加一次和真实 O scale 还原，修复固定移位近似导致小 scale 截零、除法前 half 溢出及 INT16 code 失真；Linear 校准/推理共用保形 mask 和 channel W scale，MatMul 始终沿 B 的 K 轴保护，修复方阵轴歧义。主入口 collect_mapping=False 允许 outlier，硬件 collector 仍拒绝忽略旁路成本的映射；更新说明和 README。
- 验证：真实 PyTorch 2.6.0+cpu、Transformers 4.43.1 下全部 92 项 unittest 通过（86 项既有、6 项新增）；小型 OPT/Qwen 的 INT8、FP8/INT4、FP16 各验证无 outlier/有 outlier，共 12 组实际保存加载、校准、PPL 与 prefill/decode 流程，本地替换数据 loader。新增独立数值/编码参考验证随 mask 变化的 W 计数、mask 零与加权 JSON/CSV、1D–4D Linear/channel scale/bias/多种输出、方阵与 2D–4D QK/PV、极小 scale、half 输入 INT16、masked/unmasked 范围隔离及映射/EBB/Bitlet 拒绝；unit 默认关闭。compileall 和 diff 检查通过。
- 限制与旧结果影响：未运行完整 OPT/Qwen checkpoint、14B、CUDA、真实 FineWeb 或硬件。比例仅描述 full-shape normal 量化编码，mask 零按用户要求计入，高精度 protected 编码及四项硬件成本不进入统计，不代表完整旁路存储比例或架构加速。默认恢复 outlier、masked W 改为每调用计数、旁路 scale 与数值行为修正，旧 strict-v2/旧旁路的比例和 PPL 需重新校准采集，不能混称相同设置；schema 1 消费脚本需兼容新增字段/版本。无 outlier 数值、FP 尾数/INT 补码口径和静态 W 去重保持不变，原生 FP16 YAML 保持无 outlier；各架构周期公式未修改。

## 2026-10-08 — Fix OPT/Qwen bit sparsity pipeline and native FP16

- 目的：接通当前 OPT/Qwen 校准、量化、PPL 与 prefill/decode 的编码比特稀疏度采集，用统一的 FP 显式尾数口径统计激活、静态权重及动态 K/V，暂时默认关闭 unit。
- 内容：注释缺失的 BitNet/Qwen3.5/MoE/others.evaluation 导入，主入口改用现有 Qwen 模式切换与根目录 perplexity；量化时绑定统计器，修复 shift 分母累加零位以及 reset 遗留，补全 global/phase/A/W/KV 计数；FP W/KV 与 A 共用 bit_scope。新增按 phase/layer/operand/format 的 JSON/CSV、加权汇总与保留 PPL/PD 两份快照的 TXT；主入口 collect_mapping=False、静态 W 每阶段一次并缓存计数；统计器/CLI 默认关闭 unit，PD 不再由旧 YAML 重新开启。FP16 改为原生网格舍入、scale=1、有限越界饱和；完整 FP16 模型强制以 half 加载，两个加载入口共用 dtype resolver；BF16 codes helper 显式舍入到 BF16。新增独立 OPT/Qwen FP16 YAML（OPT 上下文不超 2048）。默认 FP8/INT4 七类 Linear 的 outlier 置零、显式关闭混精、隔离 strict-v2 scale，移除未使用的层数元数据和复制 INT8 scale 建议。PPL forward 明确 use_cache=False；补充当前口径、范围、FP16 和默认 FP8/INT4 旁路问题的文档。
- 验证：真实 PyTorch 2.6.0+cpu、Transformers 4.43.1 下全套 86 项 unittest 通过（77 项既有、9 项新增）；保存并加载本地两层 OPT/Qwen，六组模型/格式组合实际校准、PPL 与 prefill/decode 通过，数据 loader 仅替换为本地合成输入。独立原始编码参考覆盖 E4M3/E5M2/FP16/BF16 子正规数及所有操作数的尾数位数、INT4 补码、比例/reset、静态 W 去重与加权导出；验证校准不采集、quant_forward 采集、不调用 mapping/unit、FP16 舍入/输出/旧 scale 拒绝、FP16 模板和默认配置。两个入口 --help、compileall、diff 检查通过。
- 限制与旧结果影响：没有运行完整 OPT/Qwen checkpoint、14B、CUDA、真实 FineWeb 或其他远端数据集；上述 PPL 仅为本地小模型的链路验证，不能作为实际模型精度结果。统计范围为包装的乘法输入，不含额外 output/bias/embedding/norm/LM head；K/V 是各次 attention 的操作数，历史 cache 会重复参与计数，默认 cache 仍为浮点存储。FP W/KV 从全存储位改为显式尾数（E4M3 8→3、FP16 16→10），global 新增 W/KV、比例分母及 FP16 数值语义改变，旧结果需重采集。默认关闭 outlier 改变数值和校准范围，旧 outlier/INT8/旧 FP16 scale 不应复用，需独立目录重新校准。架构 collector 的对齐、调度与周期公式没有修改，编码稀疏度不等同于完整运算成本或物理加速比；仍无原生 INT4 打包 kernel。


## 2026-10-08 — Add quantization pipeline entry and perplexity eval scripts

- 目的：记录 OPT/Qwen 量化流水线主入口与独立 PPL 评测、稀疏采集脚本，便于复现校准、评测和 prefill/decode profile 流程。
- 内容：新增 `0103_quant_pipeline_main.py`（校准、量化、PPL 评测与 prefill/decode 稀疏 profile 的完整流水线入口）、根目录 `perplexity.py`（wikitext/流式数据集 PPL 评测、prefill/decode 稀疏 profile、滑动窗口与对比评测）、`finweb_ppl.py`（FineWeb 流式 token 采集与 PPL 评测）；`run.md` 追加 EBB、bit-arch、asyn-cim 采集命令与输出摘录；`.gitignore` 排除 `*.zip` 与 `e2eltc/`，本地结果压缩包与个人笔记不入库。
- 验证：三个新脚本 `py_compile` 语法检查通过；`smtqt` 环境（PyTorch/Transformers 可用）下主入口可成功导入并解析 `--help`；未修改任何既有代码，77 项既有测试不受影响。
- 限制与旧结果影响：`0103_quant_pipeline_main.py` 导入 `quant.bitnet_wrapper` 与 `others.evaluation`，这两个模块在当前仓库中不存在（对应函数位于 `quant/qwen_wrapper.py` 与根目录 `perplexity.py`），运行前需按实际环境调整导入路径；未运行完整 14B、CUDA 或 PPL 数值验证；本提交不改变统计口径、映射、位宽或默认量化行为。

## 2026-10-06 — Fix Qwen Asyn-CIM profiling and LLMCompass export

- 目的：让严格 IA E4M3/W INT4 的 Qwen 256＋32 完整采集并导出匹配当前 LLMCompass 的计算倍率。
- 内容：新增专用短配置与入口；decode 按共享 KV 重新组织统计 operand，数值 forward 保持原样；记录逐调用上下文、完整覆盖和中断状态；输出纯稀疏倍率及单独 capacity bridge 文件。专用 immutable-weight 推理可缓存静态 W 编码计数，单位稀疏额外扫描关闭。声明 Linear packed W4、FP8 KV 和本地一字节权重写入假设。
- 验证：真实PyTorch 2.6.0+cpu、Transformers4.43.1下全套77项测试通过；新增3项覆盖两层小Qwen的256＋32、一次真实校准、greedy/reuse、中断无manifest、共享KV与独立packed映射。实际manifest成功导入当前后端，并通过Figure10 CLI生成完整32个decode context的报告，未使用Torch/SCALEsim占位。修改文件语法与diff检查通过。
- 限制与旧结果影响：未运行完整14B、CUDA、PPL或芯片；默认0MMM不含hidden-one/指数对齐；旧展开head decode倍率与新的shared-KV结果不可混用。bridge为逐phase/operator加权平均，非逐layer/context系统轨迹；32 decode forward应配合LLMCompass输出33token、stride1。静态缓存只用于不可变权重/scale。

## 2026-10-06 — Add short online profiles and calibrated offline latency extrapolation

- 目的：用 256＋32 在线采集当前量化数据，再按目标 shape 重新计算长序列 GEMM/最低搬运量，减少完整长序列 collector 开销，提供类似 quantspar＋LLMCompass 的数据/系统分工。
- 内容：新增独立短 YAML（独立 scales，校准长度256，64样本不变）；联合入口默认短配置，继续默认三架构共同采集，支持单独选择 EBB（275 MHz）和逐架构时钟覆盖。完整 summary 自动导出逐层/算子/phase cycles/MAC profile；离线标准库入口兼容旧完整 summary，重算 Linear、完整 attention、逐 decode context 与 GQA 物理 KV/紧凑 W4 字节，逐层算子合并重叠/串行情景，输出 JSON/CSV、dense/online_mapped、EBB 两种映射和可选在线 dense＋显式论文倍率。要求覆盖完整、操作数工作量一致，超限 EBB 默认拒绝外推，显式允许后保留标记；目标其他成本按架构导入、强制验证目标长度，缺成本或带宽为 null。README 和专门说明记录公式、命令、校准/shape 假设及现有 LLMCompass 需要补充的 memory/scheduler 工作。
- 验证：新增 8 项测试通过，覆盖独立算子/GQA 字节公式、同 shape 周期回算、时钟与 IO 分离、论文倍率不双计、覆盖/超限拒绝、零 decode、目标其他成本/激活 IO、无 PyTorch 的 python -S CLI 与输出保护；真实小型 Qwen 在 PyTorch 2.6.0+cpu、Transformers 4.43.1 下分别选择 EBB 与 BitWave，验证校准/短推理、profile 导出与离线 compute 回算。全套 74 项检查通过，语法与 diff 检查通过。
- 限制与旧结果影响：未运行完整14B、CUDA、长序列或物理芯片；没有修改 LLMCompass 后端/运行其系统仿真。cycles/MAC 固定假设包含源 padding/utilization/dataflow/FP8 扩展，长上下文稀疏与分片变化尚未验证；最低 traffic 未含容量重读、CIM refill、metadata、bank/NoC/spill，补成本后的 E2E 仍是条件情景。旧 YAML/独立入口、量化与映射公式不变，旧完整 summary 可复用；联合入口的新默认长度/scale路径改变，显式旧config保持旧长度。短校准 scale 可能改变分布，复用旧匹配 scales 的来源须自行保留。EBB仍单独采集，未宣称四架构共同推理；Bitlet双DMA未自动等同单片外带宽；不自动赋予 Bitlet/Slim-Llama Fig16 倍率。

## 2026-10-06 — Align joint profiling clocks with Asyn-CIM Table V

- 目的：让同次校准/推理的 Bitlet、BitWave、Slim-Llama 默认频率对应用户提供的 Asyn-CIM Table V，避免混用 Slim-Llama 的 50 MHz benchmark 与 200 MHz 带宽。
- 内容：Bitlet 1 GHz、BitWave 250 MHz 保持不变；联合 YAML 和 SlimLlamaConfig 默认改为 50 MHz，50 MHz 下未确认的片外带宽改为 null。未知 DRAM 时仍采集周期、计算时间和全部字节，汇总 GEMM/IO/E2E 保持 null，即使补其他算子也不生成完整时延；支持原有显式带宽覆盖。selected_frequency_hz 改为实际配置值，导出 Table V/峰值频率和带宽缺口，更新 CLI 输出/help、离线补成本错误、README 与说明，并保留显式恢复 200 MHz/1.6 GB/s 情景的办法。
- 验证：真实 PyTorch 2.6.0+cpu、Transformers 4.43.1 下全套 66 项测试通过；11 项 Slim-Llama 单独测试也通过。新增检查相同操作数在 50/200 MHz 下周期不变、计算时间比为 4、未知 DRAM 下多调用聚合/导出为 null、补其他算子仍拒绝 E2E、显式 200 MHz/1.6 GB/s 情景可恢复，selected_frequency_hz 对应实际配置。小型 Qwen 验证一次校准/共同推理及三者独立结果一致、默认无带宽路径、显式带宽下 teacher-forced/greedy CLI 和 E2E 补成本；CLI help、语法及 diff 检查通过。
- 限制与旧结果影响：未运行完整 14B、CUDA 或 RTL；只对齐频率，不声称所有硬件参数已补齐或完整 E2E 已验证。相同操作数/映射下 Slim-Llama 周期、流量、量化与 scale 不变，50 MHz 计算时间为旧 200 MHz 的四倍；旧报告仍是其原工作点，需按已记录周期重算。50 MHz 带宽没有静默借用或线性缩放；手动提供的速率是用户选择的工作点/系统假设。原有 EBB 默认、其他架构资源与周期公式不变；Table V 中的峰值/功耗是否与各频率同点仍需单独核对。

## 2026-10-06 — Add offline Fig16 capacity and bandwidth latency estimates

- 目的：增加不加载 14B、不采集真实操作数的另一种估算方法，按 Fig.16 Qwen2.5-14B 倍率、各架构原生计算资源与片外带宽估计 2048 prefill＋256 decode 的条件时延。
- 内容：新增标准库离线入口、五架构 JSON 资源配置、使用说明与 README 索引。Fig.16 使用 SIGMA/BitWave/EBB-CIM/Bit-Pragmatic/Asyn-CIM 的 1.58/1.58/2.71/3.33/4.01 倍；明确 Bitlet/Slim-Llama 不在该图。分别按 MAC、SMM、PIP 和 CIM bank 语义推导 dense 16.384/0.256/2.2528/依频率/8.192 TOPS，不重复使用已含稀疏收益的峰值。保存来源、buffer、片内带宽说明、缺失片外带宽/Pragmatic 频率和显式覆盖。按 GQA 的 query compute/physical KV traffic、完整因果 attention shape、逐 decode cache 增长计算工作量；逐算子合并 full/no overlap，保存 dense/fig16 对照、逐阶段/算子/step、JSON/CSV、Git provenance。支持 W4/KV 搬运与额外一次 activation/output IO 两种情景；兼容现有 remaining-cost JSON，若附 workload 则验证长度，没有其他成本不生成 E2E。
- 验证：Python 标准库和 python -S 下 6 项测试通过，覆盖独立 Qwen 参数/FLOPs/GQA 字节公式、decode 的前后 cache 边界、原生 dense 推导、逐算子标量 roofline、倍率仅改变计算、其他成本相加/shape 拒绝、缺失参数为 null、零 decode、非法数值、CLI 覆盖/provenance、输出保护和无模型依赖。实际执行 2048＋256 离线计算器：原生 SIGMA 1024 GB/s、Asyn 1000 GB/s 的乐观 GEMM/IO 为 3.954790/3.518821 s；另外执行共同 1000 GB/s 和 Pragmatic 1 GHz 的显式假设情景。语法与 diff 检查在提交前完成。
- 限制与旧结果影响：未运行完整 Qwen、GPU、RTL或物理芯片；平均 Fig.16 比例从 8192 外推到所有算子和 prefill/decode，Asyn 4.01 还可能保留原来源 I/O 损失。BitWave/EBB/Pragmatic 使用原生整数容量代理 FP8，未验证实际对齐位宽/多 pass 或转换成本；utilization=1 是乐观容量假设，原始 token 并行可能无法用于 batch-one decode。配置记录 buffer/片内 feed，但没有实现 residency/refill、metadata、bank/NoC/control/spill；两种重叠情景不是物理保证上下界。部分原生时钟/片外带宽缺失，没有静默借用；other 成本仍需按 2048＋256 的 LLMCompass 设置提供。原量化、stat manager、所有实际操作数 collector、scale 和既有输出 schema 均未改，旧结果不会自动获得 Fig.16 数据，不应与新容量代理数值混称相同精度的原生性能。

## 2026-10-06 — Add Slim-Llama to joint architecture profiling

- 目的：在同一次 Qwen2.5-14B IA FP8 / Linear W INT4、2048＋256 实验中增加 Slim-Llama，使 Bitlet、BitWave 和 Slim-Llama 共用实际量化操作数、校准与推理。
- 内容：新增 SlimLlamaConfig/Stats、Mapping_stat_slimllama、独立 trace/summary、单独入口、三后端 YAML 和说明；联合入口默认启用三个 collector，支持 --architectures 子集和旧双后端 YAML。默认硬件依据 ISSCC 2025 Fig.23.9.2/3/4/7：8×8 SBC、每 SBC 八列/每列八 S-LUT、200 MHz、500 KB（明确按 512000 bytes 解释）、1.6 GB/s 外部带宽。每层用可复现的实际权重 prototype 与 feature-Hamming assignment 聚类一次，保留完整 INT4 差量并以 INT5 验证；统计 center/mixed residual/buffer residual 周期、真实 tile 零比例、位宽与 activation passes，计中心 store/reuse。FP8 数值无损定点对齐，宽 A 拆成 signed INT4 digits，B 分 magnitude planes，每 plane 同步；Mixed 用四 LUT/four buffer registers 对应七 K 位置，Buffer 用八位置/two nonzero reads。QK/PV 不进行静态聚类，GQA 保留物理 KV 搬运；SRAM 检查扩展 tile、中心输出和 partial-sum 行窗口，容量情景计原始 W4、中心系数、IDs 及重读 A，无免费片外压缩。剩余成本离线入口增加 Slim-Llama，旧 schema 保留；更新 README 与使用文档。
- 验证：真实 PyTorch 2.6.0+cpu、Transformers 4.43.1 下全套 59 项测试通过（49 项既有＋10 项 Slim-Llama）；之后补充 half 输入 INT16 差量溢出保护用例，10 项 Slim-Llama 再次通过。独立纯 Python 参考覆盖 Mixed/Buffer、逐 plane barrier、INT4/FP8、宽 K/V、M/N/K 尾组和 center/delta/reuse；检查差量 ±15、无损 FP8、INT4 -8、零 A 不跳过、配置拒绝、抽样与 chunk 不变性、容量拒绝、中心/ID 流量、一次聚类与变化 probe。真实保存/加载小型 Qwen：一次校准＋一次 prefill＋两次 decode 共四次 forward，每个后端 9＋18 调用；三者单独/共同统计相同、teacher-forced/greedy CLI、GQA/cache、E2E 补成本、选择两后端和失败后全部 trace 关闭通过；旧双后端、Bitlet、BitWave、EBB、Asyn 回归通过，CLI help、语法与 diff 检查通过。
- 限制与旧结果影响：未运行完整 14B、CUDA、PPL 或 RTL。论文未完整公开聚类算法、INT4/FP8 控制、LUT 初始化与切换周期、SRAM/NoC 带宽；feature 聚类、signed-INT4 分片、Mixed/full-buffer 切换、中心复用吞吐及 SRAM 窗口是显式分析选择。默认 setup=0 是占位，内部带宽 null 表示未计入；两种情景不能保证物理上下界，完整 E2E 仍需剩余算子及转换/控制成本。静态量化权重假定不变，64 元素 probe 不能检测所有修改；--exact 仅精确枚举当前映射周期，未优化全 K 聚类。host 预处理时间不计推理时延；index reordering 不给予免费周期/能耗收益。新默认 joint 入口增加统计时间/文件大小和 profile_backend 元数据，原量化与 scale 路径不变，Bitlet/BitWave/EBB/Asyn 周期及流量公式未改；相同旧 YAML 仍只运行原两个 collector，旧结果不会自动获得 Slim-Llama 数据。

## 2026-10-06 — Add joint Bitlet and BitWave profiling

- 目的：让同一次 Qwen2.5-14B IA FP8 / Linear W INT4、2048＋256 实验同时采集 Bitlet 与 BitWave 的独立数据，避免重复校准和数值推理。
- 内容：新增 BitWave collector、Mapping_stat_bitwave、SU1–SU6 候选映射、独立 trace/summary，以及 scripts.profile_bit_arches 共同入口、scripts.profile_bitwave 单独入口和共同 YAML；manager 将同一对量化张量分发给两者，runner 分别检查覆盖、输出检查点和关闭资源，生成 bit_arch_comparison.json。BitWave 默认参考原论文 512 BCE / 4096 SMM、250 MHz、两块 256 KiB SRAM、16x64-bit banks 与 SU 对应带宽；DRAM 速率未给出，保留 null，允许显式覆盖。采用非零 magnitude 列数量而非 Bitlet 的最大列 population，统计符号/索引/尾组及压缩开销；FP8 通过无损数值定点对齐和 signed INT8 分片进行分析，记录位宽扩展，禁止 Bit-Flip 改变共同输入。片外仍按紧凑 W4 / FP8 KV 计量，BCS 压缩单独诊断；auto 按估计计算及 streaming SRAM 读量选择 SU。既有离线剩余成本入口扩展接受 BitWave，并统一浮点相加顺序；更新使用说明与 README。
- 验证：真实 PyTorch 2.6.0+cpu、Transformers 4.43.1 下 49 项测试通过（42 项原有＋7 项 BitWave）。独立 Python 标量参考验证六种 SU、Linear/attention、整数/FP8 对齐、宽 B、K/N 尾组；检查 INT4 -8、符号、hidden one、INT8 -128、A 零值不免费跳过、抽样复现/chunk 不变性、SU 选择与 SRAM 容量拒绝、索引压缩、DRAM 空缺与显式流量。真实保存/加载小型 Qwen：共同运行一次校准＋一次 prefill＋两次 decode 共四次 forward；两个 collector 各 9＋18 调用，单独与共同统计相同；teacher-forced/greedy CLI、GQA、cache、E2E 补成本、故障中断与两个 trace 关闭通过。旧 EBB/Bitlet/Asyn 回归通过。
- 限制与旧结果影响：未运行完整 14B、CUDA、PPL 或 RTL。BitWave 原文为整数引擎，没有原生 FP8 Qwen 实测；定点 scale、分片累加、FP8 metadata、转换与控制成本属于显式扩展，auto 搜索不是 ZigZag，抽样选择也存在估计误差/选择偏差。SRAM 检查只覆盖一个 tile，不模拟完整 reuse、register traffic 或 spill；未填 DRAM 速率时不产生 GEMM/IO 总时间或 E2E，搬运情景也不是物理保证上下界。共同配置的量化/scale 路径与 Bitlet 单独配置一致，可以在确认匹配后复用；Bitlet 周期/搬运公式与旧 EBB/Asyn 公式保持不变，单独入口保留；compute 加速比来自各自 dense 布局，不等同于跨架构 E2E 倍率。统计增加运行时间和 trace 大小。

## 2026-10-06 — Add paper-configured Bitlet profiling and latency scenarios

- 目的：为 Qwen2.5-14B IA FP8 / Linear W INT4、2048 prefill＋256 decode 获取 Bitlet BCE 所需的实际列负载，并提供依据原论文硬件参数的独立 GEMM/搬运时延估计。
- 内容：新增 Bitlet collector、Mapping_stat_bitlet、独立 YAML 和本地入口；默认使用原论文 32 PEs、N=64、1 GHz、24 mantissa lanes、两条各 12.8 GB/s DMA 及 25.6 GB/s local buffer 带宽，容量未报告保留 null。按产品指数 EA＋EB 对齐 B significand，包含 hidden one、符号控制、补码存储、截断和尾组诊断，不给 A 全零组加入论文未描述的 value-skip。支持 bounded chunk 的精确枚举或确定性 wave 抽样，利用精确 dense 基准估计节省量、导出误差与观察样本直方图。共享 runner 保留 EBB 默认路径；加入 GQA 物理 KV 流量、分算子 resident/full-overlap 与 streaming/no-overlap 情景、streamed trace、phase/layer/step 聚合、覆盖和 cache 增长检查、检查点及 Git commit provenance。未提供剩余成本时 E2E 字段为 null；新增离线入口将包含 LM head 的剩余算子成本补入完整采集结果，无需重跑模型；更新 README 和运行说明。
- 验证：真实 PyTorch 2.6.0+cpu、Transformers 4.43.1 下 42 项测试通过（31 项原有回归＋11 项 Bitlet 检查），包括独立 Python 标量列/调度参考、hidden one、产品指数对齐/截断、补码与尾组、零 A 不绕过 B 工作、抽样复现/chunk 不变性/误差和 dense 尾组保护、双 DMA、GQA、导出隔离及剩余成本补全。保存并加载小型 Qwen，实际校准、推理和 teacher-forced/greedy subprocess CLI，确认九种算子覆盖与逐步 cache 增长。
- 限制与旧结果影响：未运行完整 14B、CUDA、PPL 或 RTL。FP8×INT4 的 FP32 BCE 提升、同步 PE tile 调度、输出/cache 写通道、output_storage_bytes=2、group_setup_cycles=0 与 memory reuse 是显式分析假设；buffer 容量、控制停顿、partial-sum spill 和转换成本仍缺失，两种搬运情景不是物理保证上下界。项目未实现原生 W4 打包 kernel，硬件工作中的对齐截断不改变数值 forward。Bitlet 使用独立统计/schema/scale 目录，旧 Asyn-CIM/EBB 公式和倍率不变，不能直接导入 Bitlet；默认抽样直方图也不能当作全张量数量。256 decode 采用 256 次 forward 的工作量定义。

## 2026-10-06 — Support short EBB profiling and cap calibration length

- 目的：允许内存有限的本地环境先收集 2048 prefill＋256 decode 的 EBB 数据，修复仅缩短 profiling 时仍按 8192 tokens 校准的隐藏内存开销。
- 内容：新增 2048＋256 YAML，校准长度/文档筛选长度均为 2048，保留 64 样本、batch=1 和原量化/EBB 设置，scale 目录独立；入口默认将校准长度限制到有效 prefill，新增 `--calibration-length` 显式覆盖；导出有效校准配置、实际 batch 数与算子复用情况；补充短命令、调用计数以及按 Linear、attention、KV/权重搬运分别外推的方法。
- 验证：真实 PyTorch 2.6.0+cpu、Transformers 4.43.1 下 30 项测试通过；扩展保存小型 Qwen 的集成检查，用本地 tensor loader 实际校准并推理，确认 YAML 为 8192 时短 prefill 会使 loader 和模型都只校准 8 tokens，显式设置可改为 16 tokens；teacher-forced/greedy CLI 与旧回归通过。另验证新 YAML 的量化/EBB 参数一致、scale 隔离及无效参数拒绝；diff 检查通过。
- 限制与旧结果影响：未运行完整 14B 或 CUDA；短序列位宽分布不能验证长序列的 EBB 收益/溢出率，工作量倍率不等于时延倍率，也未自动生成长序列端到端结果。8192 默认实验与 EBB 统计/映射公式不变；命令行缩短 prefill 后的首次校准长度会改变，需用独立 scale 目录记录新实验。已有 scale 不会因改长度自动重校准，复用时导出的请求配置不代表 scale 原始生成历史。

## 2026-10-06 — Relax transformers version gate to a verified range

- 目的：解除 `scripts/profile_ebb` 对 `transformers` 的精确相等断言，使已安装依赖（如 4.40.x）无需替换环境即可运行，同时不放弃对已知不兼容版本的拦截。
- 内容：把 `!= "4.43.1"` 改为 `>=4.40,<4.45` 的范围检查，新增 `TRANSFORMERS_MIN`、`TRANSFORMERS_MAX_EXCLUSIVE` 与不引入新依赖的 `transformers_version_tuple`，错误信息改为报告实际版本、支持区间和越界原因；新增 `scripts/__init__.py` 使 `python -m scripts.<name>` 在本仓解析为本地包（原被 conda 环境 site-packages 中同名 `scripts` 包遮蔽，导致 `No module named scripts.profile_ebb`）；新增版本窗口边界回归测试。
- 验证：真实 PyTorch 2.6.0+cpu、Transformers **4.40.0** 下 31 项测试通过（原 30 项＋版本窗口边界 1 项），其中 CLI 端到端用例（teacher-forced 与 greedy、含 subprocess）不再需要伪装版本号即通过；`py_compile` 通过。上界依据为逐 tag 源码核对：4.43.1／4.44.0 仍为 `rotary_emb(x, seq_len=...)` 且保留 `Cache.get_usable_length`，4.45.0 起改为 `rotary_emb(x, position_ids)` 并移除 `get_usable_length`。**4.40.0 与 4.43.1 为实际执行验证，4.41–4.44 仅签核对（未执行）；完整 14B 与 CUDA 工作负载未执行。**
- 限制：不改变 `requirements-model.txt` 的 `transformers==4.43.1` 钉版本，也未扩大输出 schema。旧结果与已发表数字均在 4.43.1 下产生，本次放宽不改变任何数值路径，但 4.40–4.44 产生的统计结果与此前基线可能因底层实现差异而不可直接混用，需重跑才能对比。训练与生成路径（`0104_single_sample_inference.py` 的 qwen3.5 分支依赖 4.43+ 的 `AutoModelForImageTextToText`）不在本次放宽范围内。

## 2026-10-06 — Add MulTCIM EBB bounds and local Qwen profiling

- 目的：为 Qwen2.5-14B 的 IA FP8 / Linear W INT4、8192 prefill＋1024 decode 收集论文 EBB 所需的实际有效位宽、分组失衡及位宽覆盖数据，提供可在本地运行的独立入口。
- 内容：新增独立 EBB collector、Mapping_stat_ebb 和专用导出；按 K 维 8 元素分组，FP8 使用保留完整 significand 的无损共同指数对齐，记录 INT8/16 溢出并输出理想平衡下界、仅前导零上界及相同布局 dense 基准；显式记录符号、bank pass、宏布局和频率假设；加入 step/cache provenance、流式 JSONL 和汇总检查点；Qwen 支持 RoPE 后 FP8 cache 写入语义、共享 KV 的 GQA 计算与最低搬运量统计，历史 cache 分组精确增量更新；新增无 outlier 的独立 YAML、全模型依赖、CLI、测试和说明；移除旧入口未使用且缺失的评估导入。
- 验证：真实 PyTorch 2.6.0+cpu、Transformers 4.43.1 下 30 项测试通过（原 19 项回归＋11 项 EBB 检查），覆盖 Fig.13 例子、hidden-one/指数跨度、补码/tail、溢出、chunk 不变性、独立标量调度、dense 几何吞吐、增量 KV 与完整重算对照、GQA 流量、导出隔离、小型 Qwen 缓存增长及本地保存模型的 teacher-forced/greedy CLI。完整 14B 和 CUDA 工作负载未执行。
- 限制：论文原生是 INT8/16；FP8 对齐、INT16 bank pass 和理想平衡是显式分析假设，不声称复现未公开的 EBB 控制器。超出 word width 时保留扩展位宽统计，周期不能归为原生 MulTCIM 性能。未计 FP8/partial-sum 转换、累加、CIM 写入、片外搬运和非 GEMM；流量仅为最低必需量。旧 Asyn 统计 schema/倍率保持独立，不能导入 EBB。新实验关闭 outlier、在写 cache 前量化 K/V，需使用新目录重新校准；旧量化统计结果不能直接替代新结果。

## 2026-10-05 — Fix fixed FP8 INT4 quantization and mapped sparsity statistics

- 目的：修复严格 IA FP8 / Linear W INT4 通路的量化旁路、缺失依赖、编码计数和错误映射倍率，使核心模块与单工作负载可以独立验证。
- 内容：补全 QuantSpec、codes 量化工具及包入口；默认关闭混精；修正 disabled output、bias、动态 collector 参数和 channel W scale；恢复两阶段和权重稀疏采集；统一原生编码和 unit 统计；修正 token/K-round reshape、独立 operand 聚合、几何/sync 参数及 dense 基准；新增分块映射、原始 JSON 和 LLMCompass bridge manifest 导出；改进旧入口错误/阶段导出；新增独立工作负载脚本、测试、依赖与使用说明。
- 验证：真实 PyTorch 2.6.0+cpu 下 19 项回归通过，包含数值参考、编码边界及独立标量调度对照；静态/动态单 Linear 示例、旧入口 --help、语法及 diff 检查通过；合成小模型 manifest 在当前 LLMCompass fig10 中导入验证。LLMCompass 冒烟运行只对缺失 SCALEsim 使用导入占位，其 GPU/systolic 路径未执行。
- 限制：没有运行完整 Qwen checkpoint/PPL 或 CUDA 验证。公开快照的全模型包装器与数据/评估模块仍未提交；本次没有用旧附件覆盖。周期只描述所选 activation 编码位，不包含完整 FP 固定成本、访存和权重更新；bridge 为每 phase/operator 的平均换算。旧 sparsity/mapping 结果需重采集，旧 scale 需按新配置重新校准。

## 2026-10-05 — Add quant core modules and single-sample inference script（ca07eb0）

- 目的：提交量化与单样本推理相关代码。
- 内容：加入 mapping、quant_linear、quant_matmul、stat_manager 和单样本脚本共 5 个文件。
- 验证：本次按 Git 内容补录；不代表初始通路已通过完整执行验证。
- 限制：该快照缺少导入模块，且数值/统计/映射存在上述后续修复的问题。

