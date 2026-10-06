# quantspar

量化核心支持 IA FP8 E4M3FN、Linear W INT4 的数值仿真、实际编码稀疏统计和 Asyn-CIM 映射计算周期。IA 显式指定 `a_bit="e4m3"`，W 指定 `w_bit=4`；数字 `8` 代表 INT8。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.profile_fp8_int4 --demo --output outputs/fp8_int4_demo.json
```

`--demo` 是合成张量的接口示例，不是 Qwen 推理结果。真实 Linear 工作负载可以通过 `--tensor-file` 输入包含 `activation`、`weight` 和可选 `bias` 的张量字典。

默认关闭混精。需要严格 IA FP8 / W INT4 时使用 `mixed_precision=False`、`outlier_ratio=0`。数值仿真使用浮点运算表达量化后的网格，不包含原生 INT4 kernel 或打包存储实现。

- [FP8/INT4 修复与使用说明](docs/fp8_int4_path.md)
- [EBB-CIM / MulTCIM 统计与本地运行](docs/ebb_cim.md)
- [每次提交的修改记录](CHANGELOG.md)
- [提交维护规则](AGENTS.md)

Qwen 包装器与校准加载器已经上传。全模型依赖见 `requirements-model.txt`；EBB 使用独立的 `scripts.profile_ebb` 入口和 `qwen2_14b_ebb_f8i4.yaml` 配置，默认收集 8192 prefill＋1024 decode。内存有限时使用 `config/qwen2_14b_ebb_f8i4_2048_256.yaml`，校准和 profiling 都缩短，scale 目录独立；命令行缩短 prefill 也会默认限制校准长度。EBB 输出是基于显式 FP8 对齐和调度假设的 GEMM 周期范围，不是论文测得的 FP8 性能或端到端时延。完整 14B checkpoint 与 CUDA 实验需本地运行。
