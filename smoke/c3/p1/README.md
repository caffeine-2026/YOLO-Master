# C3 P1｜三方小样本检测对照

本目录与 P0 / Smoke Test 完全隔离。P1 比较 NEU-DET 与 DeepPCB 上的 V-PEFT、Full Fine-tuning 和 Frozen Backbone；同一数据集的三个方法共享同一 100-image train split、验证/测试集、模型、优化器、增强和训练预算。

seed824 的 30-epoch early pilot、50-epoch intermediate、75-epoch convergence candidate 与六组独立 100-epoch final single-seed candidate 均已完成。100-epoch 审计为 6/6 `CONVERGED_OR_PLATEAU`，75→100 方法排序稳定，因此 `FINAL_SINGLE_SEED_EPOCH = 100`、`MULTISEED_READY = YES`。本轮只生成 seed825/826 的 12-run 计划，没有运行它们。不得使用 P0 的 1 epoch 或单 seed 指标声明总体方法优劣。

- 协议：`config/protocol.yaml`
- 数据计划：`evidence/P1_DATA_PLAN.md`
- 日志：`logs/`
- 30-epoch 历史对照：`results/pilot_seed824.md`
- 50-epoch candidate 对照：`results/pilot_seed824_e50.md`
- 75-epoch candidate 对照：`results/pilot_seed824_e75.md`
- 100-epoch final single-seed 对照：`results/pilot_seed824_e100.md`
- 30-epoch 历史审计：`docs/CONVERGENCE_AUDIT.md`
- 50-epoch 历史审计：`docs/CONVERGENCE_AUDIT_E50.md`
- 75-epoch 历史审计：`docs/CONVERGENCE_AUDIT_E75.md`
- 100-epoch 最终审计：`docs/CONVERGENCE_AUDIT_E100.md`
- multi-seed 计划（未执行）：`config/multiseed_plan.yaml`
- 效率分析：`docs/VPEFT_EFFICIENCY_ANALYSIS.md`
- 报告：`docs/C3_P1_REPORT.md`
