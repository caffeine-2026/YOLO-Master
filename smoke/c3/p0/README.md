# C3 V-PEFT 工业缺陷小样本实战

官方 C3 P0 已关闭：NEU-DET 与 DeepPCB 各完成一次 RTX 4090 GPU 0、FP32、1 epoch 的 V-PEFT 最小闭环。两次运行均为 strict `ACCEPT`，实际后端为 `peft`，无静默回退或数值恢复。

| Dataset | Planner | Actual backend | Planned / applied | Exit | Adapter | Status |
| --- | --- | --- | ---: | ---: | --- | --- |
| NEU-DET | `ACCEPT` | `peft` | 59 / 52 | 0 | yes | `PASS` |
| DeepPCB | `ACCEPT` | `peft` | 59 / 52 | 0 | yes | `PASS` |

## 交付入口

- [C3 P0 最终报告](docs/C3_P0_FINAL_REPORT.md)
- [8.25 统一验收记录](docs/ADMISSION_20260825.md)
- [完整复现指南](PEFT_RUN_GUIDE.md)
- [P0 结构化汇总](evidence/c3_p0_summary.json)
- [静态验收报告](evidence/static_validation.json)
- [DeepPCB 数据来源清单](evidence/deeppcb_source_manifest.json)
- [DeepPCB 转换清单](evidence/deeppcb_manifest.json)
- [DeepPCB 数据 smoke](evidence/deeppcb_data_validation.json)
- [NEU-DET V-PEFT 日志](logs/neu_det_vpeft_gpu_fp32_seed824/)
- [DeepPCB V-PEFT 日志](logs/deeppcb_vpeft_gpu_fp32_seed824/)

## Visual Evidence

- [NEU-DET overview](visualizations/neu_det_overview.jpg)
- [DeepPCB overview](visualizations/deeppcb_overview.jpg)
- [NEU-DET GT vs Prediction samples](visualizations/neu_det/comparison/)
- [DeepPCB GT vs Prediction samples](visualizations/deeppcb/comparison/)
- [NEU-DET prediction JSON](evidence/neu_det_visual_predictions.json)
- [DeepPCB prediction JSON](evidence/deeppcb_visual_predictions.json)
- [Visualization manifest](evidence/visualization_manifest.json)

这些图片使用已完成的 V-PEFT `best.pt` 在固定 test 样本上执行真实 GPU inference，用于证明模型确实完成了实际目标检测闭环。`conf=0.25` 下两组固定样本均没有预测框，证据按原结果保留，没有降低阈值或替换样本。P0 的 1 epoch / single-seed 结果不用于证明最终方法优劣。

## 边界

- Official C3 P0 = NEU-DET + DeepPCB 各一次 V-PEFT；MVTec AD 仅是未来可选扩展。
- NEU-DET 的 Full-SFT 与冻结主干结果只属于 preliminary smoke evidence，不是 P1 结论。
- 单 epoch、单 seed 只证明闭环可执行，不评价收敛或方法优劣。
- 原始数据、派生图像、模型权重和 adapter 权重均由忽略规则排除；仓库只提交工具、配置、完整文本日志、清单和 SHA-256。
