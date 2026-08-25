# C3 P1 Pilot 报告（seed 824）

## 1. Research Question

在工业缺陷小样本条件下，V-PEFT 是否能以显著更少的可训练参数和资源成本，保持或改善 Full-SFT / Frozen Backbone 的检测性能？

## 2. Protocol

两个数据集统一使用 YOLO11n 预训练权重、100 张训练图、30 epochs、batch 8、imgsz 640、AdamW、lr0=0.001、weight decay=0.0005、cosine scheduler、GPU 0、FP32、seed 824。三个方法仅训练策略不同，最终精度统一在锁定 test split 上评测。

## 3. Dataset / Split

NEU-DET 按来源类别确定性分层抽样；DeepPCB 按 seed824 确定性无放回抽样并审计多标签分布。三种方法共享同一成员列表，原有 val/test 不变且无重叠。详见 `../evidence/P1_DATA_PLAN.md`。

## 4. Three Methods

- Full-SFT：`lora_r=0, freeze=0`。
- Frozen Backbone：依据 `yolo11.yaml` 的结构冻结 `model.0-model.10`，训练 `model.11-model.23`。
- V-PEFT：rank=8、alpha=16、strict V-PEFT AO Planner，不允许 fallback，实际 backend 必须为 PEFT。

## 5. Accuracy Results

| Dataset | Method | mAP50-95 | mAP50 | Precision | Recall | Trainable / Total | Trainable vs Full | Peak GPU Mem | Memory Saving | Time | GPU-hours | Time Ratio | Checkpoint | Adapter | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.2574 | 0.4919 | 0.4574 | 0.5392 | 2,590,994 / 2,591,010 | 100.00% | 2652 MiB | 0.00% | 115.3s | 0.03201 | 1.000× | 5.23 MiB | 0.00 MiB | PASS |
| NEU-DET | Frozen Backbone | 0.2384 | 0.4922 | 0.6109 | 0.4761 | 1,225,522 / 2,591,010 | 47.30% | 1700 MiB | 35.91% | 106.5s | 0.02959 | 0.924× | 5.23 MiB | 0.00 MiB | PASS |
| NEU-DET | V-PEFT | 0.2043 | 0.3875 | 0.3991 | 0.3922 | 613,602 / 2,772,770 | 23.68% | 2621 MiB | 1.16% | 129.8s | 0.03605 | 1.126× | 5.75 MiB | 0.73 MiB | PASS |
| DeepPCB | Full-SFT | 0.4294 | 0.6606 | 0.5914 | 0.6741 | 2,590,994 / 2,591,010 | 100.00% | 2673 MiB | 0.00% | 119.6s | 0.03322 | 1.000× | 5.23 MiB | 0.00 MiB | PASS |
| DeepPCB | Frozen Backbone | 0.3057 | 0.5264 | 0.4857 | 0.5893 | 1,225,522 / 2,591,010 | 47.30% | 1710 MiB | 36.02% | 112.7s | 0.03131 | 0.942× | 5.23 MiB | 0.00 MiB | PASS |
| DeepPCB | V-PEFT | 0.2117 | 0.3877 | 0.3575 | 0.5029 | 613,602 / 2,772,770 | 23.68% | 2642 MiB | 1.15% | 137.3s | 0.03815 | 1.148× | 5.75 MiB | 0.73 MiB | PASS |

表内为固定 test split 的单 seed 指标；不按 mAP 单独排名。

## 6. Resource Results

同表记录 trainable/total parameters、GPU 峰值、训练耗时、GPU-hours、checkpoint 与 adapter 大小。Memory Saving 和 Time Ratio 均以同数据集 Full-SFT 为基准。

## 7. Multi-seed Statistics

本阶段仅执行 seed=824 pilot。seed=825/826 尚未运行，因此不报告 mean/std/95% CI，也不把当前结果作为正式 P1 多 seed 结论。

## 8. Qualitative Results

正式四栏 GT / Full-SFT / Frozen Backbone / V-PEFT 可视化留到 multi-seed protocol 确认后生成；本阶段没有挑选或替换样本。

## 9. Planner Analysis

- NEU-DET: Planner=ACCEPT, planner backend=vpeft, actual backend=peft, planned/applied=59/52.
- DeepPCB: Planner=ACCEPT, planner backend=vpeft, actual backend=peft, planned/applied=59/52.

## 10. Limitations

当前只有一个 seed，30 epochs 是统一 pilot 预算。收敛记录如下：

- NEU-DET / Full-SFT: best validation mAP50-95=0.2457, epoch=30.
- NEU-DET / Frozen Backbone: best validation mAP50-95=0.2170, epoch=30.
- NEU-DET / V-PEFT: best validation mAP50-95=0.1912, epoch=30.
- DeepPCB / Full-SFT: best validation mAP50-95=0.4107, epoch=30.
- DeepPCB / Frozen Backbone: best validation mAP50-95=0.3006, epoch=30.
- DeepPCB / V-PEFT: best validation mAP50-95=0.2066, epoch=30.

至少一组曲线在最后四个 epoch 内达到最佳值，正式 multi-seed epoch 预算仍需统一收敛审计后锁定。 当前差异可能包含随机波动，不能外推为方法普遍优劣。

## 11. P1 Conclusion

六组 seed824 pilot 均通过闭环验收；本表仅验证公平协议可运行并给出单 seed 初步测量，不声明任一方法优于其他方法。
