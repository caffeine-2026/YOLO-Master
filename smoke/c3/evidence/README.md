# C3 结果证据索引

## 环境与数据证据

- `environment.json`：最终 GPU 提交版本、系统、核心包版本、CUDA 入口状态、模型与数据 SHA-256。
- `environment_cpu_fallback.json`：切换到 CUDA wheel 前的 CPU 降级环境和 GPU 隔离证据。
- `static_validation.json`：全部 YAML、NEU-DET 标签、清单和可发布路径检查结果。
- `neu_det_full_manifest.json`：NEU-DET 完整划分的图像与目标数量。
- `neu_det_fewshot_manifest.json`：NEU-DET 5-shot 划分、seed 和目标数量。

实际压缩包、图像、标签和模型权重由仓库忽略规则排除。清单、配置、工具和 SHA-256 用于绑定本地数据与产物。

## 运行结果证据

- `gpu_smoke_comparison.json`：GPU FP32 稳定组与 GPU AMP 降级组的结构化三方对比。
- `cpu_smoke_comparison.json`：V-PEFT、Full-SFT 和冻结主干的结构化对比。
- `../docs/ADMISSION_20260825.md`：面向评审的统一结果、结论边界和风险说明。
- `../logs/neu_det_vpeft_cpu_seed824/`：V-PEFT 完整日志、规划元数据、独立适配器配置和产物索引。
- `../logs/neu_det_full_sft_cpu_seed824/`：Full-SFT 完整日志和产物索引。
- `../logs/neu_det_frozen_backbone_cpu_seed824/`：冻结主干完整日志和产物索引。

9 次运行均为 `completed`、退出码 0，并完成最终验证与检查点保存。三个 V-PEFT 运行另外导出了独立适配器；适配器权重保留在本地忽略目录，其配置、完整运行元数据、大小和 SHA-256 已进入提交证据。

## 尚未生成的证据

- 未发生数值恢复的纯 AMP 基准：当前 AMP 三方运行都触发一次恢复，已如实标记为降级。
- MVTec 派生检测运行：官方页面要求填写下载表单，实际数据尚未取得。
- 多轮收敛比较：当前只执行单轮最小任务，不将指标解释为最终性能。

上述状态均在环境记录和设计说明中明确标注，不使用估算值或其他机器结果补齐。
