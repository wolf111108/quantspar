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
- [每次提交的修改记录](CHANGELOG.md)
- [提交维护规则](AGENTS.md)

公开快照的 `0104_single_sample_inference.py` 还依赖未提交的完整模型包装器和数据/评估模块；本次没有用过期附件覆盖这些缺失模块。核心接口和单 Linear 工作负载脚本可以独立运行。完整 Qwen checkpoint 推理、PPL 和 GPU 验证尚未完成。
