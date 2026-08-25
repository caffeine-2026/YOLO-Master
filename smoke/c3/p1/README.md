# C3 P1｜三方小样本检测对照

本目录与 P0 / Smoke Test 完全隔离。P1 比较 NEU-DET 与 DeepPCB 上的 V-PEFT、Full Fine-tuning 和 Frozen Backbone；同一数据集的三个方法共享同一 100-image train split、验证/测试集、模型、优化器、增强和训练预算。

seed 824、30 epoch 的六个 pilot 已完成。统一收敛审计决定 `EXTEND_ALL_TO_50`；现有运行保持冻结，在六组均以非覆盖方式达到 50 epoch 并重新审计前，不启动 seed 825/826。不得使用 P0 的 1 epoch 指标评价方法优劣。

- 协议：`config/protocol.yaml`
- 数据计划：`evidence/P1_DATA_PLAN.md`
- Pilot 日志：`logs/`
- Pilot 对照：`results/pilot_seed824.md`
- 收敛审计：`docs/CONVERGENCE_AUDIT.md`
- 效率分析：`docs/VPEFT_EFFICIENCY_ANALYSIS.md`
- 报告：`docs/C3_P1_REPORT.md`
