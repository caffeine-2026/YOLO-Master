# Issue #2 后续：C3 V-PEFT 数据增强消融与经验性能上限

基线提交保持为 `bf6c7c508635dec0be849aedaa3eac5d88ed220d`，工作分支为 `codex/c3-augmentation-ablation-20260831`。原有 checkpoint、CSV、JSON 和 P0/P1/P2 结果均未覆盖。准确率优先基线是 613,602 个可训练参数的 V-PEFT；195,410 参数的 ≤10% 方案仅保留为效率优先负面结果。

## 防止数据泄漏

只用 train/validation 搜索 baseline、mild、medium、strong；所有搜索 YAML 都没有 `test` key。先按 validation 固定最终策略，再访问 locked test。训练条件固定为 100 epochs、imgsz 640、batch 8、AdamW、seed 824/825/826。冻结证据中 `test_access_before_selection=false`、`test_metrics_used_for_selection=false`。

## 结果

| 数据集 | validation 选择 | 100-shot locked-test mAP50-95 | 相对原有 613,602-param 基线 | paired 95% CI | 结论 |
|---|---|---:|---:|---:|---|
| NEU-DET | mild | 0.30596 | -0.01433 | [-0.06441, 0.03575] | 未改善，不显著 |
| DeepPCB | medium | 0.62621 | +0.10960 | [0.05316, 0.16605] | 显著改善 |

相对本次 no-augmentation 对照，NEU-DET 是 +0.07857，CI [-0.00463, 0.16177]；DeepPCB 是 +0.08977，CI [0.06140, 0.11814]。NEU-DET 的 Full-SFT retention 为 91.90%，DeepPCB 为 96.55%。两者可训练参数仍是 613,602，peak GPU memory 均为 2,621.44 MiB；增强把 100-shot 平均训练时间从 336.3s 增至 526.8s（NEU），从 347.3s 增至 546.1s（DeepPCB）。

DeepPCB 通过预注册 validation scaling trigger，因此完成 10/50/100/500-shot 三 seed 扩展。mAP50-95 paired delta 分别为 +0.09082、+0.09344、+0.08977、+0.04047，四个 CI 都大于 0；500-shot 的增益减小，显示收益趋于饱和。NEU-DET 的 validation CI 包含 0，按规则未扩展 scaling。

强度曲线也显示饱和：NEU-DET 在 mild 达峰，medium/strong 下降；DeepPCB 从 mild 到 medium 仅增加 0.00275，strong 下降。因此这里只称为“在已测试增强范围内的经验性能上限”，不称理论上限。

per-class AP50-95：NEU 的 crazing、inclusion、patches、rolled-in_scale 显著提高，pitted_surface、scratches 不显著；DeepPCB 的 open、short、mousebite、spur 显著提高，copper、pin-hole 不显著。没有类别在 95% CI 上显著下降。

## 证据与验证

- 协议：[augmentation_protocol.yaml](../config/augmentation_protocol.yaml)
- 冻结选择：[frozen_selection.json](../results/frozen_selection.json)
- seed 结果与统计：[results](../results)
- 原始命令/config/stdout/stderr/resource：[logs](../logs)
- locked-test overall/per-class JSON：[evaluations](../evaluations)
- checkpoint（本地、gitignored）与 SHA 清单：[artifacts](../artifacts)
- 图表及嵌入的 CSV 行/hash：[figures](../figures)
- 失败、修复、重跑记录：[failures](../failures)
- 测试与 validator：[evidence](../evidence)

P0/P1/P2 和 integrated research 共 5 个 validator 已 PASS，且受保护的旧证据 hash 未变化。bf6c completion gate 在独立 detached worktree 中 PASS，并重新加载 72 个旧 checkpoint。相关 pytest 为 426 passed、17 skipped；Ruff、format、git diff check 均 PASS。首次 sandbox pytest 因 localhost socket 权限失败，原始失败已保留，host 重跑通过。

限制：仅 3 个 seed，NEU-DET 未证明相对原有准确率基线有提升；DeepPCB Full-SFT 是原有参考结果，并非本次同增强重跑；三个 DeepPCB medium checkpoint 因首次 test orchestration 中断而物理访问 locked test 两次，但两次 overall/per-class 数值和 checkpoint hash 完全一致，且首次 test 后没有重新选择或调参。
