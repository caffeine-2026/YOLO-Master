# C3 结果证据索引

## 环境与数据证据

- `environment.json`：最终 GPU 提交版本、系统、核心包版本、CUDA 入口状态、模型与数据 SHA-256。
- `environment_cpu_fallback.json`：切换到 CUDA wheel 前的 CPU 降级环境和 GPU 隔离证据。
- `static_validation.json`：全部 YAML、双数据集、运行证据、产物哈希和可发布路径检查结果。
- `neu_det_full_manifest.json`：NEU-DET 完整划分的图像与目标数量。
- `neu_det_fewshot_manifest.json`：NEU-DET 5-shot 划分、seed 和目标数量。
- `deeppcb_source_manifest.json`：DeepPCB 原作者仓库、锁定提交、原始数量、官方划分和关键哈希。
- `deeppcb_manifest.json`：DeepPCB few-shot 转换策略、划分、类别与目标数量。
- `deeppcb_data_validation.json`：bbox、split 重叠和真实 dataloader batch 检查。

实际压缩包、图像、标签和模型权重由仓库忽略规则排除。清单、配置、工具和 SHA-256 用于绑定本地数据与产物。

## 运行结果证据

- `gpu_smoke_comparison.json`：GPU FP32 稳定组与 GPU AMP 降级组的结构化三方对比。
- `c3_p0_summary.json`：NEU-DET + DeepPCB Official C3 P0 结构化结论。
- `cpu_smoke_comparison.json`：V-PEFT、Full-SFT 和冻结主干的结构化对比。
- `../docs/ADMISSION_20260825.md`：面向评审的统一结果、结论边界和风险说明。
- `../docs/C3_P0_FINAL_REPORT.md`：双数据集 P0 最终报告。
- `../logs/neu_det_vpeft_cpu_seed824/`：V-PEFT 完整日志、规划元数据、独立适配器配置和产物索引。
- `../logs/neu_det_full_sft_cpu_seed824/`：Full-SFT 完整日志和产物索引。
- `../logs/neu_det_frozen_backbone_cpu_seed824/`：冻结主干完整日志和产物索引。

既有 9 次 NEU 运行与新增 DeepPCB V-PEFT 运行均为 `completed`、退出码 0，并完成最终验证与检查点保存。V-PEFT 运行另外导出了独立适配器；适配器权重保留在本地忽略目录，其配置、完整运行元数据、大小和 SHA-256 已进入提交证据。

## 验收边界

- Official C3 P0 采用 NEU-DET 与 DeepPCB 各一次 V-PEFT GPU FP32 结果。
- NEU 三方、历史 AMP 与 CPU 运行仅作 preliminary smoke / 补充排障证据。
- MVTec AD 只属于未来可选扩展。
- 当前只执行单轮最小任务，不将指标解释为最终收敛性能。

上述边界均在统一验收文档中明确标注，不使用估算值或其他机器结果补齐。
