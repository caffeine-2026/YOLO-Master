# Issue #2 后续进展：C3 LOVO、原生 MIP、Planner 三分支与参数效率复核

本轮工作针对剩余研究项进行了真实服务器验证，不以 README 或网页展示替代实验，也没有手工修正指标。完整韩文报告与 24 行三策略统计见 [C3 completion report](C3_COMPLETION_REPORT.md) 和 [final summary CSV](../results/final_summary.csv)。

## 已完成的实证工作

1. 在项目虚拟环境安装并固定 `ortools==9.15.6755`。NEU-DET、DeepPCB 各自实际运行 AO、DCO、MIP；两次 MIP 均为 `requested=mip`、`effective=mip`、`fallback=false`、SCIP `OPTIMAL`。原始 metadata、目标模块和 rank pattern 见 [solver evidence](../evidence/solvers)。
2. 使用真实 YOLO11n nc=6 模型和有效正 budget 构造 Planner 三分支：ACCEPT 选择 59/60 个候选模块；ADAPT 因不支持的 attention target 触发 `attention_target_policy` 并安全关闭 attention；REFUSE 在 budget=1 时找不到任何可行 adapter，选择 0 个模块。结构化 JSON、raw log、audit 见 [branch evidence](../evidence/planner_branches)。
3. 预先锁定 validation-only 四候选搜索，禁止 test 参与选择。8/8 search run 成功，选择 `b250k_predictors`。该设置实际 trainable=195,410，为 Full-SFT 2,590,994 的 7.54%，参数减少 92.46%。
4. 因最终 V-PEFT 设置发生变化，重新执行 `2 datasets × 4 shots × 3 seeds = 24` 个 V-PEFT run。24/24 成功；每个 run 均有 100 个有限 epoch、真实 CUDA 记录、resolved config、完整 stdout/stderr、资源采样、locked-test 结果、三个 checkpoint 及 manifest hash。原始目录见 [final logs](../logs/final) 和服务器本地 [final artifacts](../artifacts/final)。
5. Full/Frozen 的 48 个既有 run 逐项与 raw metrics 和 manifest 对照后复用；旧 V-PEFT 24 个也在 old/new 比较前重新核对。最终 72 行矩阵、三 seed mean/SD/95% CI、paired delta 和 scaling curve 全部由原始 CSV 重建，见 [results](../results) 与 [figures](../visualizations)。

## 参数效率结论（负面结果如实保留）

新设置虽然达到 ≤10% 参数目标，但没有保持整体精度。八个 dataset×shot 单元的非加权平均 mAP50-95 从旧 V-PEFT 的 0.35181 降至 0.27750；平均 peak memory 仅从 2,636.4 MiB 降至 2,598.0 MiB，平均时间从 552.0 秒降至 542.1 秒。DeepPCB 10/50/100-shot 对 Full-SFT 的 retention 分别只有 38.0%/36.2%/57.3%。因此不能宣称新 V-PEFT 在精度上优于 Frozen 或旧 V-PEFT。

NEU-DET 100-shot 与 DeepPCB 500-shot 的 V-PEFT mean 分别比 Frozen 高 0.0051 与 0.0097，但 paired 95% CI 均跨越 0；其余六个单元低于 Frozen。这是参数—精度 trade-off，而不是精度胜利。

## LOVO 正式 calibration

LOVO 只读取最后一个 epoch 的 validation mAP50-95，不读取 P1/P2 locked test。seed 824/825/826 被视为同一实验单元的重复测量，不用于虚增 observation 数。

- calibration：NEU-DET/DeepPCB × 10/50/100-shot，共 6 个唯一 observation；36 个 source run ID 无重复。
- held-out：两数据集 500-shot，共 2 个 observation；12 个 source run ID 与 calibration 完全分离。
- `observation_count=6`，`uses_learned_evidence=true`，`source=learned_regression`，`confidence=0.01667`。
- calibration CV：RMSE 0.12842，MAE 0.11389，R² -0.44000。
- held-out：预测 ΔmAP50-95=-0.16002；NEU-DET 实测 -0.04492，DeepPCB 实测 -0.07737；RMSE 0.10020，MAE 0.09888，95% coverage=1.0，但 interval [-0.40821, 0.08816] 很宽。

这不是高质量 calibration：所有 observation 都是同一 YOLO11n/LoRA/rank-8，design rank 只有 1/12；shot subset 和 validation image 也有重叠。旧的 `predicted_delta=0.0660, confidence=0, source=default_prior` 仍明确标记为 cold-start prior，不作为实测提升证据。完整结果见 [LOVO report](../evidence/lovo/lovo_calibration_report.json)。

## 代码修复与验证

- 增加 `lora_head_train_policy=full|predictors|frozen`，在 `predictors` 模式只训练 detection terminal predictors。
- MIP metadata 现在记录 backend、native status、objective/bound、runtime、budget、module/rank、fallback。
- LOVO observation 增加稳定 ID、metric split、source run IDs，并拒绝重复 experimental unit；confidence 根据样本充分度与可识别 feature rank 计算。
- Planner ACCEPT/ADAPT/REFUSE 与 native MIP 均新增 regression test。
- 既有 DCO rank 超过 layer capacity 的失败、修复与重跑仍保留在 [failure/repair report](../../final/FAILURE_REPAIR_RERUN.md)。

相关 pytest 首次沙箱运行结果为 417 passed / 17 skipped / 1 localhost-socket permission failure；同一 Gloo test 在真实 host 权限下重跑为 1 passed，因此 code failure 为 0。新 completion code 的完整 Ruff/format、legacy 变更文件 critical Ruff、`git diff --check` 均通过。原始记录见 [release checks](../evidence/release_checks_final.json)，P0/P1/P2 与新 integrated validator 的最终 release 重跑 6/6 均通过，记录见 [validation release evidence](../evidence/validation_release)。首次因测试 warning 中含绝对虚拟环境路径而失败的记录仍保留在 [initial validation evidence](../evidence/validation)，没有删除或掩盖。

实际服务器命令清单见 [executed commands](EXECUTED_COMMANDS.md)。

## 仍有限制

- 195,410 参数设置是参数目标成功、精度保持失败的结果，不能替代旧 V-PEFT 作为默认高精度方案。
- 完整 checkpoint 包含已训练 predictor head；当前 adapter-only safetensors 不含该 non-adapter head 权重，独立 adapter 部署需后续修复。
- LOVO 样本小且 rank-deficient，confidence 只有 0.01667。
- 全项目 unfiltered Ruff 仍有既有 legacy style debt；没有通过改规则或放松 validator 来伪造 PASS。
- 当前 `gh auth status` 显示 GitHub CLI 凭据失效，因此本文尚未直接评论到 Issue #2；此文件可直接复制发布。

## 状态清单

- [x] P0 原生 MIP（两数据集）
- [x] P0 Planner ACCEPT/ADAPT/REFUSE 实证
- [x] P1 validation-only ≤10% 搜索与 24-run 重跑
- [x] P2 10/50/100/500-shot 三策略 scaling 重建
- [x] LOVO 6 calibration + 2 held-out，误差/区间/coverage 报告
- [x] checkpoint/config/seed/epoch/hash 与 raw metrics/manifest 交叉验证
- [ ] Issue #2 在线评论（GitHub CLI 认证失效；本文件为可复制完成稿）
- [ ] 全项目 unfiltered Ruff（既有 legacy debt；本轮新增代码与 critical gate 已通过）
