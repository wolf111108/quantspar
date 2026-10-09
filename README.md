# quantspar

量化核心支持 IA FP8 E4M3FN、Linear W INT4 的数值仿真、实际编码稀疏统计和 Asyn-CIM 映射计算周期。IA 显式指定 `a_bit="e4m3"`，W 指定 `w_bit=4`；数字 `8` 代表 INT8。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.profile_fp8_int4 --demo --output outputs/fp8_int4_demo.json
```

完整 Qwen Asyn-CIM FP8/W4 的256＋32统计与LLMCompass导入：

```bash
python -m scripts.profile_asyn_cim --model-path /path/to/Qwen2.5-14B --output-dir outputs/asyn_cim_256_32
```

输出逐阶段/算子mapped compute speedup和独立的 `llmcompass_speedups.json`。decode按共享KV分组，完整覆盖后才导出桥接文件；默认16 macros、64×48、16 banks、0MMM。命令、复用scales、SMMM口径、后端W4/FP8搬运与32次decode对应output-length33的约定见[Asyn-CIM采集说明](docs/asyn_cim.md)。

`--demo` 是合成张量的接口示例，不是 Qwen 推理结果。真实 Linear 工作负载可以通过 `--tensor-file` 输入包含 `activation`、`weight` 和可选 `bias` 的张量字典。

默认关闭混精。需要严格 IA FP8 / W INT4 时使用 `mixed_precision=False`、`outlier_ratio=0`。数值仿真使用浮点运算表达量化后的网格，不包含原生 INT4 kernel 或打包存储实现。

普通比特稀疏度入口允许 outlier；默认 Linear 比例为 0.0001，仅统计未被 outlier mask 保护的 normal codes，不计入 mask 产生的零。高精度旁路保留在数值推理中，不计入该比例；含 mask 的 W 按每次 forward 重计。FP 仍只计尾数，unit 默认关闭。默认 scales 使用独立 outlier-v3 目录；统计口径变化只需重新采集稀疏度，无需重新校准。硬件映射入口继续要求 outlier=0。

OPT-1.3B、OPT-6.7B、Qwen2.5-7B 的 FineWeb 重跑矩阵已提供 15 份配置（BF16/BF16、FP8/FP8、INT8/INT8、INT8/INT4、FP8/INT4）。本地顺序校准、PPL 与 prefill/decode，自动导出加权比特比例和 PPL 汇总：

```bash
bash scripts/run_sparsity_ppl_matrix.sh \
  --opt-1-3b-path /path/to/opt-1.3b \
  --opt-6-7b-path /path/to/opt-6.7b \
  --qwen-7b-path /path/to/Qwen2.5-7B
```

用 `--dry-run` 检查矩阵，`--resume` 继续完整且设置一致的运行，`--eval-flow ppl` 只采 PPL/full-forward；默认 unit 关闭、FP 只计显式尾数、量化组 Linear outlier=0.0001。输出在 `outputs/sparsity_ppl_rerun/`，设置与旧表脚注的区别见[重跑说明](docs/sparsity_ppl_rerun.md)。

- [三个模型的 15 组 FineWeb 配置、运行脚本与 PPL/稀疏度汇总](docs/sparsity_ppl_rerun.md)
- [OPT/Qwen 比特稀疏度入口、统一 FP 尾数口径与原生 FP16 配置](docs/bit_sparsity_pipeline.md)
- [FP8/INT4 修复与使用说明](docs/fp8_int4_path.md)
- [EBB-CIM / MulTCIM 统计与本地运行](docs/ebb_cim.md)
- [Bitlet 论文参数、2048＋256 采集与时延估计](docs/bitlet.md)
- [一次推理同时统计 Bitlet / BitWave / Slim-Llama](docs/bitlet_bitwave.md)
- [Slim-Llama 聚类、S-LUT 与访存估计口径](docs/slimllama.md)
- [Fig.16 倍率、原生计算容量与带宽的离线时延估算](docs/paper_fig16_latency.md)
- [每次提交的修改记录](CHANGELOG.md)
- [提交维护规则](AGENTS.md)

直接使用论文 Fig.16 倍率估计，无需加载 14B 或安装模型依赖：

~~~bash
python -m scripts.estimate_paper_latency \
  --output-dir outputs/paper_fig16_2048_256
~~~

默认同时输出 SIGMA、BitWave、EBB-CIM、Bit-Pragmatic、Asyn-CIM 的 dense 容量、2048＋256 工作量、计算与带宽情景。Bitlet/Slim-Llama 不在 Fig.16；缺失的时钟、片外带宽或剩余算子耗时保持 null。可显式覆盖带宽/频率并复用 other-latency JSON 得到条件 E2E，命令和原生 FP8 支持限制见说明文档。

一次校准与推理同时获取 Bitlet、BitWave、Slim-Llama 的独立统计：

~~~bash
python -m scripts.profile_bit_arches \
  --model-path /path/to/Qwen2.5-14B \
  --output-dir outputs/bit_arches_256_32
~~~

联合入口默认 256＋32、IA FP8 / W INT4，采用独立短 scale 目录；显式旧 YAML 仍保持原长度与 scale 设置。分别输出三份 summary/trace 和 bit_arch_comparison.json。默认频率对齐 Asyn-CIM Table V：Bitlet 1 GHz、BitWave 250 MHz、Slim-Llama 50 MHz。Slim-Llama 保留 64 SBC、500 KB SRAM，50 MHz 下片外带宽未确认，默认 null，可用 --slimllama-dram-bandwidth-gbps 显式提供；未填时只输出计算时间/流量，GEMM/IO 和 E2E 保持 null。它以实际权重中心和无损差量建模输出复用；FP8 分片、聚类算法及调度是显式分析假设。BitWave 论文未给 DRAM 速率，可用 --bitwave-dram-bandwidth-gbps 指定搬运假设。--architectures bitlet bitwave 可只运行两者；原双后端 YAML 仍只启用其两个 collector，三种架构也保留单独入口。

Bitlet 默认使用原论文 32 PEs、64 元素组、1 GHz、两条各 12.8 GB/s DMA 和 25.6 GB/s local buffer 带宽；容量未报告，保持 null。安装全模型依赖后可直接运行：

~~~bash
python -m scripts.profile_bitlet \
  --model-path /path/to/Qwen2.5-14B \
  --output-dir outputs/bitlet_2048_256
~~~

默认对真实 FP8/W4 操作数进行可复现的 BCE wave 抽样，也可用 --exact 全量枚举。bitlet_summary.json 输出独立的列负载、计算周期、GEMM/搬运情景和抽样误差。完整 E2E 需补充 LM head 等剩余算子的耗时；可在采集后使用 scripts.estimate_bitlet_latency 补全，见运行说明。

Qwen 包装器与校准加载器已经上传。全模型依赖见 `requirements-model.txt`；EBB 使用独立的 `scripts.profile_ebb` 入口和 `qwen2_14b_ebb_f8i4.yaml` 配置，默认收集 8192 prefill＋1024 decode。内存有限时使用 `config/qwen2_14b_ebb_f8i4_2048_256.yaml`，校准和 profiling 都缩短，scale 目录独立；命令行缩短 prefill 也会默认限制校准长度。EBB 输出是基于显式 FP8 对齐和调度假设的 GEMM 周期范围，不是论文测得的 FP8 性能或端到端时延。完整 14B checkpoint 与 CUDA 实验需本地运行。


### Short online profiles / offline target-length estimates

`python -m scripts.profile_bit_arches` now defaults to 256+32. Select collectors with `--architectures`; EBB is selected alone at 275 MHz in the new short config. Completed runs export per-layer/operator/phase cycle coefficients. `python -m scripts.estimate_profile_latency --profile-dir ... --output-dir ...` estimates target shapes without loading weights, keeps missing IO/E2E inputs null, and accepts target remaining costs from LLMCompass. See [short_profile_latency.md](docs/short_profile_latency.md) for assumptions and commands.

