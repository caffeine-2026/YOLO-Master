# C3 研究实验与提交证据进展报告

## 结论

本次工作严格聚焦 C3 研究实验，没有把范围改成网页演示。服务器上的 P0/P1/P2 原始日志、配置、checkpoint、metrics、资源记录和统计结果已经完成交叉核验：P0、P1、P2 的实验交付均通过；LOVO 正式校准仍未完成，保持 `LOVO calibration pending`。

## 实际服务器与复用原则

- 主机可见 8 张 NVIDIA GeForce RTX 4090，PyTorch 2.13.0+cu130、CUDA 13.0 可用。
- 检查时 GPU 0 有其他用户计算进程，GPU 1–7 空闲；未中断任何其他用户任务。
- P1/P2 共 72 个最终矩阵单元全部具有 100 epoch 完整日志、locked test、resolved config、资源/时间记录、artifact SHA-256 和 checkpoint。
- 72/72 个 `best.pt` 均实际加载成功，模型总参数、seed、epoch metadata 与 metrics 一致。因此没有重复训练已经有效的实验。

## P0 / Planner / Solver

- NEU-DET 与 DeepPCB 的 V-PEFT 均真实完成，Planner decision 为 `ACCEPT`，budget 2,100,000，计划 59 个 module，安全过滤后实际应用 52 个 module。
- AO 与 DCO 均有真实执行日志。DCO 修复后两个数据集均 exit 0。
- MIPR 在 OR-Tools 缺失时显式降级到 AO，runtime metadata 记录 requested=`mip`、effective=`ao`、`ImportError` 和原因。
- DCO 历史失败日志保留：`rank 64` 超过 `1.conv` capacity 16。修复后该层 rank 为 16，并有回归测试。

## P1 三策略结果（3 seeds）

| 数据集 | Full-SFT mAP50-95 | Frozen mAP50-95 | V-PEFT mAP50-95 | V-PEFT retention | V-PEFT trainable params |
| --- | ---: | ---: | ---: | ---: | ---: |
| NEU-DET | 0.3329 | 0.2935 | 0.3203 | 96.20% | 613,602 |
| DeepPCB | 0.6486 | 0.4844 | 0.5166 | 79.65% | 613,602 |

V-PEFT 相对 Full-SFT 减少 76.32% 可训练参数，但显存只减少约 1.2%，训练时间反而增加约 13%。报告因此不宣称 V-PEFT 是通用赢家，而是如实呈现准确率、参数、显存和时间 trade-off。

## P2 scaling（3 seeds）

- NEU V-PEFT retention（10/50/100/500）：98.33% / 93.26% / 96.20% / 97.83%。
- DeepPCB V-PEFT retention：65.34% / 65.42% / 79.65% / 93.69%。
- 72-cell 原始表、均值/95% CI、paired deltas、retention CSV 与 14 张最终图已经从原始 metrics 重新生成并独立复算一致。

## LOVO 边界

当前 `predicted_delta=0.06602954545454547`、confidence=0、state=`cold_start`、source=`default_prior`、observation count=0、`uses_learned_evidence=false`。它不是实测 ΔmAP，也不是 V-PEFT 性能提升证据。历史 P1/P2 metadata 的 null 保持原样；没有使用 test set 做 calibration。独立正式观测不足 5 个，因此 `LOVO calibration pending`。

## 修复与验证

- 修复 P1 validator 的 P0 旧路径问题。
- 修复 P2 seed-824 历史 validator 与 final protocol schema 不兼容的问题。
- 修复 P2 final validator 把 P1 验证器代码变更误判为原始实验历史变更的问题；不可变范围现在只覆盖 logs、artifacts、split/multiseed manifests 和 raw metrics。
- 修复 manual LoRA fallback 未先冻结全部 base parameters 的真实回归；核心 suite 65/65 通过，LoRA/MoLoRA/PEFT/Planner 相关 pytest 全量结果为 295 passed、7 skipped、2 warnings、0 failed。
- 保留 DCO 失败、修复、成功重跑证据。
- 统一验证器已通过 P0 solver、72 checkpoints、原始 CSV/JSON、图表 provenance、JSON/YAML、文档路径和隐私扫描。

## 证据入口

- `smoke/c3/final/README.md`
- `smoke/c3/final/evidence/research_delivery_validation.json`
- `smoke/c3/final/evidence/raw_command_manifest.json`
- `smoke/c3/p1/results/p1_all_runs.csv`
- `smoke/c3/p2/results/p2_all_runs.csv`
- `smoke/c3/p0/evidence/solver_audit_20260831.json`
- `smoke/c3/final/FAILURE_REPAIR_RERUN.md`

## 尚存限制

- 只有 3 个 seed，部分 95% CI 较宽。
- 固定 100 epoch，不同 shot 下 optimizer update 数不同。
- LOVO 尚无可用的正式 calibration dataset。
- 当前实现中 V-PEFT 的参数优势没有转化为显著显存或时间优势。
