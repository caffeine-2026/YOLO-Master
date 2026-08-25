# C3 P1｜三方小样本检测对照

本目录与 P0 / Smoke Test 完全隔离。P1 比较 NEU-DET 与 DeepPCB 上的 V-PEFT、Full Fine-tuning 和 Frozen Backbone；同一数据集的三个方法共享同一 100-image train split、验证/测试集、模型、优化器、增强和训练预算。

seed824 的 30-epoch 历史 pilot 与六组独立 50-epoch candidate 均已完成。50-epoch 审计有 4/6 组仍超过固定收敛阈值，统一决定 `EXTEND_ALL_TO_75`；当前 `MULTISEED_READY = NO`，未运行 seed825/826。不得使用 P0 的 1 epoch 或未收敛的单 seed 指标评价方法优劣。

- 协议：`config/protocol.yaml`
- 数据计划：`evidence/P1_DATA_PLAN.md`
- 日志：`logs/`
- 30-epoch 历史对照：`results/pilot_seed824.md`
- 50-epoch candidate 对照：`results/pilot_seed824_e50.md`
- 30-epoch 历史审计：`docs/CONVERGENCE_AUDIT.md`
- 50-epoch 当前审计：`docs/CONVERGENCE_AUDIT_E50.md`
- 效率分析：`docs/VPEFT_EFFICIENCY_ANALYSIS.md`
- 报告：`docs/C3_P1_REPORT.md`
