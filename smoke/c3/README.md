# C3｜工业缺陷检测 V-PEFT 小样本实战

> 首轮可比实验进展与任务书对照审计（数据截止：2026-08-31）

## 一、任务书原始目标

任务书的核心问题是：**V-PEFT 能否在小样本工业场景中，以更少的可训练参数、显存和训练时长逼近全量微调？**

任务书同时要求：

- **P0**：在 DeepPCB 与 NEU-DET 跑通 V-PEFT，记录实际 `ΔmAP`、LOVO 预测和至少两种 solver 日志；
- **P1**：在严格同预算条件下比较 V-PEFT、Full-SFT、Frozen Backbone，统一报告精度、可训练参数、显存和时长；
- **P2**：完成 10/50/100/500 张小样本缩放曲线，或发现并修复一个可复现的 planner 真实 bug；
- **交付物**：三方对照、小样本曲线、planner 流程图、护栏/求解器日志，以及可选 bug-fix PR；
- **验收提醒**：V-PEFT 至少在参数效率上形成数量级优势；若走 bug 路线，必须有 issue、fix 与复现证据。

## 二、当前状态：自动化 PASS 不等于课题书全部验收

| 项目 | 当前证据 | 对照任务书的状态 |
| --- | --- | --- |
| P0 两数据集真实 V-PEFT 闭环 | NEU-DET、DeepPCB 均为 strict planner、实际 `peft` 后端、adapter 非空 | 已完成 |
| P0 solver / LOVO 审计 | 两个数据集均只有 AO；`predicted_delta=null`、`confidence=null` | **尚未完成任务书要求** |
| P1 三方同预算对照 | 两数据集均完成 Full-SFT / Frozen / V-PEFT，对比精度、参数、显存、时长 | 已完成可比实验 |
| P1 数量级参数效率验收线 | 总可训练参数仅减少 76.32%，即 4.22×；adapter 本身为 14.26× 更小 | **按总可训练参数口径尚未达线** |
| P2 10/50/100/500 缩放路线 | 2 数据集 × 4 规模 × 3 方法 × 3 seed，共 72 个单元 | 已完成，72/72 自动验证 PASS |
| 交付物 | 三方表、缩放图、日志已存在；planner 流程图尚缺；无 bug-fix PR | 部分完成 |

因此，当前最准确的表述是：**首轮可比实验和 P2 缩放矩阵已经完成，但原任务书的 planner/LOVO 证据与 P1 数量级参数效率验收线仍有缺口，不能把 P2 自动验收 PASS 写成整个 C3 已正式结项。**

## 三、版本、数据与统一实验协议

- 任务筹备会给出的稳定基线为 `YOLO-Master-v26.08@43d4011`，研究 HEAD 锁定点为 `57b9ea3`；
- C3 实际 P0 证据固定 `BASE_REF=acce839c7e895d6b179de7f7093fa879e237cc7b`、`START_REF=bad9294f4217773dcf0ba0666c64de121a324fda`，其中 `acce839c` 位于 `57b9ea3` 之后；
- 使用 HEAD 路线是因为本课题依赖 V-PEFT planner；最终报告应继续保留完整 commit，并说明与稳定 release 基线的关系，不能只写 `main` 或 `HEAD`；
- 数据集：NEU-DET、DeepPCB；
- 最终对照协议：YOLO11n、100 epochs、batch 8、imgsz 640、AdamW、`lr0=0.001`、`weight_decay=0.0005`、cosine scheduler；
- 训练集采用 `10 ⊂ 50 ⊂ 100 ⊂ 500` 嵌套划分，验证集和测试集固定；
- seed：824 / 825 / 826；当前只有 3 个 seed，因此只报告均值、区间和同 seed 配对差值，不作普适显著性声明。

## 四、P0：真实 planner 闭环与尚缺证据

### 已验证事实

NEU-DET 与 DeepPCB 的 P0 配置相同：

| 字段 | 实际记录 |
| --- | --- |
| planner backend / solver | `vpeft` / `ao` (`AlternatingOptimizationSolver`) |
| 状态 | `ACCEPT` |
| adapter 参数预算 | 最大 2,100,000；planner 使用 191,616 |
| 目标模块 | planner 规划 59 个；安全兼容过滤后实际应用 52 个 |
| rank | 8 |
| utility | 29.5 |
| 实际训练后端 | `peft` |
| fallback / 数值恢复 | 未发生 |
| adapter 导出 | 两个数据集均非空 |

两份原始审计记录：

- [NEU-DET V-PEFT runtime metadata](p0/logs/neu_det_vpeft_gpu_fp32_seed824/vpeft_runtime_metadata.json)
- [DeepPCB V-PEFT runtime metadata](p0/logs/deeppcb_vpeft_gpu_fp32_seed824/vpeft_runtime_metadata.json)

### 与任务书仍有差距

1. 当前正式日志只有 **AO 一种 solver**；尚无 DCO 或 MIPR 对照日志，未满足“至少两种求解器日志”；
2. 两个 P0 placement plan 的 `predicted_delta` 与 `confidence` 都是 `null`，因此不能声称已记录 LOVO 预测；
3. MIPR 没有正式运行，所以 OR-Tools 缺失时切换 AO 的降级路径也没有被本课题证据实际覆盖；
4. 当前没有独立的 planner 流程图交付物。

P0 的 1 epoch / 单 seed 指标仅证明链路可运行，不用于评价收敛或方法优劣。详见 [P0 说明](p0/README.md) 与 [P0 最终报告](p0/docs/C3_P0_FINAL_REPORT.md)。

## 五、P1/P2：首个可比实验结果

下表为三次独立 seed 的 **mAP50-95 均值**；`Δ` 为 `V-PEFT - Full-SFT`，Retention 为 `V-PEFT / Full-SFT`。

| 数据集 | 图像数 | Full-SFT | Frozen Backbone | V-PEFT | Δ mAP50-95 | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NEU-DET | 10 | 0.1212 | **0.1325** | 0.1192 | -0.0020 | 98.33% |
| NEU-DET | 50 | **0.2687** | 0.2195 | 0.2505 | -0.0181 | 93.26% |
| NEU-DET | 100 | **0.3329** | 0.2935 | 0.3203 | -0.0126 | 96.20% |
| NEU-DET | 500 | **0.3995** | 0.3769 | 0.3909 | -0.0087 | 97.83% |
| DeepPCB | 10 | **0.2903** | 0.2206 | 0.1897 | -0.1006 | 65.34% |
| DeepPCB | 50 | **0.5669** | 0.3898 | 0.3709 | -0.1960 | 65.42% |
| DeepPCB | 100 | **0.6486** | 0.4844 | 0.5166 | -0.1320 | 79.65% |
| DeepPCB | 500 | **0.7006** | 0.5979 | 0.6564 | -0.0442 | 93.69% |

完整均值、标准差和 95% 置信区间见 [P2 summary](p2/results/p2_summary.md)，同 seed 差值见 [paired analysis](p2/results/paired_analysis.md)。

## 六、对任务书核心问题的回答

### 1. 精度：只在部分场景接近 Full-SFT

- NEU-DET 上 V-PEFT retention 为 93.26%–98.33%，与 Full-SFT 的差距较小；
- DeepPCB 10/50-shot retention 仅为 65.34%/65.42%，到 500-shot 才达到 93.69%；
- V-PEFT 在 8 个条件中的 5 个高于 Frozen Backbone，但 8 个均值都没有超过 Full-SFT；
- 结论是数据集与样本规模依赖，而不是 V-PEFT 普遍更准确。

### 2. 可训练参数：明显减少，但总量尚非数量级

| 方法 | 总可训练参数 | 相对 Full-SFT |
| --- | ---: | ---: |
| Full-SFT | 2,590,994 | 1.00× |
| Frozen Backbone | 1,225,522 | 2.11× 更少 |
| V-PEFT | 613,602 | 4.22× 更少 |

V-PEFT 中 adapter 参数为 181,760，单看 adapter 是 Full-SFT 的 1/14.26；但实际训练还包括因类别不匹配重新初始化的检测头参数。因此验收时应报告 **613,602 个总可训练参数**，不能用 adapter-only 数量替代总口径。按该口径，76.32% 减少尚未达到任务书所写的“数量级优势”。

### 3. 显存与时长：当前实现没有显示优势

- V-PEFT 相比 Full-SFT 的峰值显存仅减少约 **1.14%–1.53%**；
- 实测 GPU-hours 反而增加约 **5.67%–19.57%**；
- 参数量减少没有自动转化为 total peak memory 或 wall-clock 优势；adapter/planner 路径、激活和框架开销的占比需要 profiler 才能确认。

所以本轮数据只支持：**V-PEFT 明显减少总可训练参数，并在部分区间保留较高精度；尚不支持它更快、更省显存或普遍更准确。**

## 七、P2 缩放路线完成情况

- 最终矩阵：`2 数据集 × 4 样本规模 × 3 方法 × 3 seed = 72`；
- 自动运行验证：72/72 PASS；
- V-PEFT strict backend / adapter 完整性：24/24 PASS；
- P1 的 18 个 100-shot 单元原样复用，P2 新增单元没有改写 P1 历史；
- 该 PASS 表示矩阵完整性、协议公平性和产物完整性通过，不代表官方已经授予 C3 P2 或最终优秀等级。

证据见 [P2 final validation](p2/evidence/p2_final_validation.json) 与 [P2 最终报告](p2/docs/C3_P2_REPORT.md)。

缩放图：

- [NEU-DET mAP50-95 多 seed 曲线](p2/visualizations/final/neu_map5095_multiseed.png)
- [DeepPCB mAP50-95 多 seed 曲线](p2/visualizations/final/deeppcb_map5095_multiseed.png)
- [NEU-DET accuracy vs. parameters](p2/visualizations/final/neu_accuracy_vs_params_multiseed.png)
- [DeepPCB accuracy vs. parameters](p2/visualizations/final/deeppcb_accuracy_vs_params_multiseed.png)

## 八、下一阶段应优先补什么

### 9.01–9.07：中期演示 / 机制分析 / 消融设计

1. 补一张可审查的 planner 流程图：模型图与约束 → AO/DCO/MIPR → budget/targets → `ACCEPT/ADAPT/REFUSE` → 安全过滤 → PEFT 注入 → 日志与 adapter；
2. 固定同一模型与预算，至少补 **AO + DCO** 两种 solver 日志；如选 MIPR，必须同时测试 OR-Tools 缺失时的 AO fallback；
3. 明确 LOVO 在当前代码中的可用入口；若仍输出 `null`，应把它作为真实缺口提交 issue，而不是伪造预测值；
4. 针对 DeepPCB 10/50-shot 做目标尺度和失败样本分析，再决定是否扫描 target modules、rank 或学习率；
5. 用 profiler 分解 adapter、激活、优化器和框架开销，解释为什么参数减少但显存/时间没有同比改善。

### 9.08–9.12：多 seed / 参数扫描 / PR 冻结

- 多 seed 已提前完成，应把时间用于受控的单变量消融；
- 仅使用验证集选择设置，测试集保持冻结；
- 若不能稳定改善多个 seed，应保留负结果并给出 Accept/Adapt/Refuse 建议；
- PR 或正式 issue 必须包含改动摘要、测试证据、消融数据和已知局限。

## 九、复现与证据入口

- [P0：最小 smoke 与交付证据](p0/README.md)
- [P1：100-shot 三策略对照](p1/README.md)
- [P1 效率分析](p1/docs/VPEFT_EFFICIENCY_ANALYSIS.md)
- [P2：多 seed 缩放实验](p2/README.md)
- [P2 汇总结果](p2/results/p2_summary.md)
- [同 seed 配对分析](p2/results/paired_analysis.md)
- [最终自动验收 JSON](p2/evidence/p2_final_validation.json)
- [YOLO-V-PEFT 可视化演示](https://yoloc3vpeft.com/)（补充展示；仓库中固定的 CSV/JSON 才是实验结论的数据源）

## 十、当前结论

**首轮可比数字、三方对照、多 seed 与 10/50/100/500 缩放矩阵已经形成，数据证据链较完整；但原任务书要求的第二种 solver、LOVO 预测、planner 流程图和总可训练参数数量级优势仍未满足。下一阶段不应人为“把 V-PEFT 做得更好看”，而应优先补齐这些验收缺口，并通过受控消融解释 DeepPCB 低样本和运行效率问题。**
