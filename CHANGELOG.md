# 修改记录

每次提交在同一提交中增加一条，记录目的、内容、验证和限制。当前提交用与 commit message 一致的标题标识；提交前不填自身 SHA。规则见 [AGENTS.md](AGENTS.md)。

## 2026-10-06 — Support short EBB profiling and cap calibration length

- 目的：允许内存有限的本地环境先收集 2048 prefill＋256 decode 的 EBB 数据，修复仅缩短 profiling 时仍按 8192 tokens 校准的隐藏内存开销。
- 内容：新增 2048＋256 YAML，校准长度/文档筛选长度均为 2048，保留 64 样本、batch=1 和原量化/EBB 设置，scale 目录独立；入口默认将校准长度限制到有效 prefill，新增 `--calibration-length` 显式覆盖；导出有效校准配置、实际 batch 数与算子复用情况；补充短命令、调用计数以及按 Linear、attention、KV/权重搬运分别外推的方法。
- 验证：真实 PyTorch 2.6.0+cpu、Transformers 4.43.1 下 30 项测试通过；扩展保存小型 Qwen 的集成检查，用本地 tensor loader 实际校准并推理，确认 YAML 为 8192 时短 prefill 会使 loader 和模型都只校准 8 tokens，显式设置可改为 16 tokens；teacher-forced/greedy CLI 与旧回归通过。另验证新 YAML 的量化/EBB 参数一致、scale 隔离及无效参数拒绝；diff 检查通过。
- 限制与旧结果影响：未运行完整 14B 或 CUDA；短序列位宽分布不能验证长序列的 EBB 收益/溢出率，工作量倍率不等于时延倍率，也未自动生成长序列端到端结果。8192 默认实验与 EBB 统计/映射公式不变；命令行缩短 prefill 后的首次校准长度会改变，需用独立 scale 目录记录新实验。已有 scale 不会因改长度自动重校准，复用时导出的请求配置不代表 scale 原始生成历史。

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
