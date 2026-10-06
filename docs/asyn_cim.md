# Qwen IA FP8／W INT4 Asyn-CIM：256＋32 → LLMCompass

## 当前入口与修复

使用 `scripts.profile_asyn_cim`，不要直接复用旧 `0104_single_sample_inference.py` 和带 outlier 的旧 YAML。默认 Qwen2.5-14B，实际层数和维度从 checkpoint 读取：48 层、hidden 5120、FFN 13824、40 Q heads、8 KV heads。每层统计七个 Linear 加 QK/PV，共九个 GEMM。

专用 YAML 默认 256 prefill、32 次 decode forward；校准长度 256、64 样本、独立 scale 目录，batch=1。严格 Linear IA=E4M3FN/W=INT4，Attention A/B=E4M3FN，输出 E4M3FN，无 outlier、无混精，FP8 KV 写入用反量化浮点 cache 数值仿真。eager attention 必须保留。实际加载的是原始 checkpoint，再在本程序中执行量化；不是要求预先打包的 INT4 checkpoint。

Asyn-CIM 默认 16 macros、每 macro 16 banks、64×48、1 GHz、1 cycle/effective bit，对应后端默认几何。旧单样本入口默认32 macros不能与后端16 cores混用；如果修改几何，两端必须同时修改并重新统计。

修复 decode GQA：Qwen 数值 forward 仍执行 repeat_kv；Asyn-CIM collector 将 `[B,Qheads,1,K]` 打包为 `[B,KVheads,Qheads/KVheads,K]`，每个共享 K/V operand 对应一组 Q 行。prefill 保持展开 head BMM，匹配当前 LLMCompass prefill。统计不改变模型输出。

## 本地采集

安装 `requirements.txt` 与 `requirements-model.txt`；Transformers 支持 >=4.40,<4.45，验证版本4.43.1。在 quantspar 仓库根目录运行：

```bash
python -m scripts.profile_asyn_cim \
  --model-path /path/to/Qwen2.5-14B \
  --output-dir outputs/asyn_cim_256_32
```

默认自动校准缺失的 scale 并复用已经匹配的 scale。要复用之前的严格 FP8/W4、相同 granularity/mixed/outlier 设置的全套 scales：

```bash
python -m scripts.profile_asyn_cim \
  --model-path /path/to/Qwen2.5-14B \
  --scale-dir /path/to/matching_scales \
  --skip-calibration \
  --output-dir outputs/asyn_cim_256_32_reuse
```

`--skip-calibration` 不表示忽略缺失文件；缺 scale 会报错。新校准可能改变精度和稀疏分布，复用 scales 的校准来源应另行保留。默认 teacher-forced：使用同一文档后续32个 token；`--greedy-decode` 才调用 LM head 选择下一个 token，不因 EOS 提前缩短统计。可提供 `--token-file tokens.pt`（torch.long，至少288个 token；greedy至少256），避免统计文本下载；新校准仍使用配置的数据集。可显式覆盖 `--prefill-length`、`--decode-steps`、`--calibration-samples`、`--calibration-length`。

完整14B应有 prefill432次、decode13824次调用。每一步验证 KV cache 增长；最终验证每层、每算子、每decode step覆盖。中断只留下标记为 interrupted 的 summary/run_config，不留下可导入的 speedups 文件。

## 输出与倍率定义

- `asyn_cim_summary.json`：逐调用 shape、context、layout、编码计数、dense/sparse steps；`phases` 中按算子及阶段聚合 `sum(dense)/sum(sparse)`，这是报告稀疏计算收益的倍率，不能平均各调用的ratio。
- `llmcompass_speedups.json`：当前后端的容量分母 / 实测 mapped sparse steps；采用 `baseline=effective`。这是计算周期桥接倍率，可能包含padding容量换算，与上一个纯稀疏倍率不同。后端只对计算应用一次，不缩短搬运成本。
- `run_config.json`：实际维度、版本、校准是否执行、执行长度、完成状态和源commit。

默认 `bit_scope=mantissa`，统计显式0MMM三个尾数位；不包含 hidden-one、指数对齐或额外符号成本。需要SMMM时在专用YAML将 `asyn_cim.bit_scope` 改成 `sign_mantissa`，输出自动变为4个dense bits；不能把0MMM的倍率用于SMMM分母。raw storage的8个编码位不能直接解释为8个硬件串行周期。

新 manifest 显式声明 Linear 片外 packed INT4=0.5B、K/V FP8=1B、本地 Linear CIM写入=1B。最后一项沿用现有后端一字节系数写入假设，不是已验证的4-bit紧凑本地阵列；若硬件实现支持不同位宽，可显式调整 manifest 的 `transport.local_linear_weight_storage_bits` 并记录条件。权重零值不自动等价于片外压缩。

为减少额外统计开销，专用入口关闭unit稀疏扫描，并缓存每阶段静态权重编码计数；仅适用于本入口不可变的权重和固定scales，decode动态K/V仍每次计数。激活映射仍完整枚举，不是Bitlet/BitWave的wave采样。

## 导入 LLMCompass

先更新 LLMCompass_mod 的 `CIM_Arch` 分支。将 `llmcompass_speedups.json` 复制或通过绝对路径提供给该仓库，在其根目录运行：

```bash
python -m ae.figure10_qwen.test_latency \
  --input-lengths 256 --output-lengths 33 --sample-stride 1 \
  --cores 16 --array-height 64 --array-width 48 --banks 16 \
  --speedups-json /path/to/asyn_cim_256_32/llmcompass_speedups.json \
  --output-dir outputs/asyn_cim_256_32
```

**33不是写错：** quantspar执行32次decode forward；LLMCompass的输出长度G包含prefill产生的首token，因此用G=33、G−1=32。stride1让上下文完整覆盖256..287，严格匹配导出范围。若使用默认output-length32或stride64，范围不匹配会拒绝导入。要估计生成32token的请求，应在quantspar改为 `--decode-steps 31`，后端用 `--output-lengths 32 --sample-stride 1`。

后端当前 CIM_IO 和 CIM_WU_IO 默认各1TB/s；仍另计片上GB、CIM权重更新和复制成本，不是只给计算加最低W/KV字节。新transport将Linear和Attention分开，避免packed W4被按1B传输，或K/V误按INT4传输。旧manifest未带transport时保留原后端行为。

输出 `report.json` 与 `requests.csv` 为 **Figure-10 Transformer-stack E2E估计**：包含GEMM/IO和现有向量算子，仍使用LayerNorm/RMSNorm、GeLU/SiLU代理及串行调度；不包含embedding、final norm、LM head、RoPE、residual、gate-times-up、采样、host/network等。不能称为完整生成服务的实测E2E。

## 验证与限制

```bash
python -m unittest discover -s tests -v
```

真实CPU PyTorch/Transformers小型两层Qwen验证256＋32、新校准、scale复用、greedy、GQA分组、完整覆盖和中断导出；独立packed映射核对GQA周期。设置 `LLMCOMPASS_QUANTSPAR_BACKEND=/path/to/LLMCompass_mod/software_model/quantspar_cim.py` 可额外执行实际manifest loader和Figure10 CLI联调。

未运行完整14B、CUDA、PPL或RTL。每phase/operator一个加权平均ratio，后端没有逐layer/context trace接口；换长度需重新采集或显式建立外推假设。不能把本次短序列manifest的context字段直接改成长序列后冒充观测结果。旧展开head decode结果需重采或用保留的实际编码重新映射，不能只改JSON里的shared-KV标签。
