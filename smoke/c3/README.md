# C3 V-PEFT 工业缺陷小样本实战

本目录是 C3 交付的统一入口，完整覆盖环境安装、基线与最小任务、复现命令、配置文件、完整日志、结果证据、设计说明以及风险与降级。正式实验已在 RTX 4090 GPU 0 上完成；CPU 结果只保留为辅助功能核验，不作为 GPU 性能替代品。

## 当前状态

| 项目 | 状态 | 可核验事实 |
| --- | --- | --- |
| GPU 环境 | `PASS` | 8 张 RTX 4090；驱动 595.84；PyTorch 2.13.0+cu130；CUDA 张量分配通过 |
| GPU FP32 三方运行 | `PASS` | GPU 0、`amp=false`；三种方案均单次完成且未触发数值恢复 |
| NEU-DET | `PASS` | 完整划分 1,440/180/180；5-shot 划分 30/180/180；标签与清单验证通过 |
| V-PEFT 验收 | `PASS` | strict 规划状态 `ACCEPT`，规划 59 个目标，安全过滤后应用 52 个目标，独立适配器导出成功 |
| 完整日志与结果证据 | `PASS` | 三次正式 GPU FP32 运行均包含命令、完整日志、最终配置、指标、资源、耗时、退出码和产物校验和 |

本次验收范围是 NEU-DET GPU FP32 三方最小任务，表内项目全部通过。历史 AMP 和 CPU 运行仅保留为
补充排障证据，不参与本次合格判定；MVTec AD 双数据集任务列在统一文档末尾，作为下一阶段 P0。

## 交付入口

- [8.25 交付评估](docs/ADMISSION_20260825.md)：面向评审的单一文档，包含环境安装、基线/最小任务、复现命令、配置文件、完整日志、结果证据、设计说明以及风险与降级。
- [完整复现指南](PEFT_RUN_GUIDE.md)
- [配置文件](config/)
- [完整日志](logs/README.md)

## 关键证据

- [GPU 三方运行汇总](evidence/gpu_smoke_comparison.json)
- [交付评估与结果说明](docs/ADMISSION_20260825.md)
- [CPU 三方运行汇总](evidence/cpu_smoke_comparison.json)
- [环境实测](evidence/environment.json)
- [静态验证报告](evidence/static_validation.json)
- [V-PEFT GPU FP32 完整运行目录](logs/neu_det_vpeft_gpu_fp32_seed824/)
- [Full-SFT GPU FP32 完整运行目录](logs/neu_det_full_sft_gpu_fp32_seed824/)
- [冻结主干 GPU FP32 完整运行目录](logs/neu_det_frozen_backbone_gpu_fp32_seed824/)

## 结论边界

- GPU 是 VRAM、吞吐和最终对照实验的唯一有效执行设备；CPU 不是 GPU 的性能替代品。
- GPU FP32 的 `PASS` 只表示单轮 GPU 最小任务稳定完成；精度仍不代表收敛或方案优劣。
- 历史 AMP 运行触发过数值恢复，因此只保留为排障证据，不纳入本次 GPU FP32 验收。
- 三次 CPU 运行证明数据加载、模型初始化、训练、验证和保存链路可用；单轮指标不代表收敛性能。
- GPU 显存为每秒一次的设备级采样，包含空闲基线，并可能漏过不足一秒的瞬时峰值。
- 实际数据和模型权重由仓库忽略规则排除；提交中保留生成工具、清单、运行元数据和 SHA-256。
