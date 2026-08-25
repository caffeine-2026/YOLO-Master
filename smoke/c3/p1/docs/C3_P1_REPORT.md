# C3 P1 对照报告（seed824，50-Epoch Candidate）

## 1. Research Question

在工业缺陷小样本条件下，V-PEFT 是否能以显著更少的可训练参数和资源成本，保持或改善 Full-SFT / Frozen Backbone 的检测性能？

## 2. Protocol

`restart_all`，YOLO11n/yolo11n.pt、每数据集 100 张训练图、50 epochs、batch 8、imgsz 640、AdamW、lr0=0.001、weight decay=0.0005、cosine scheduler、GPU0、FP32、seed824。30 epoch 结果仅为历史 pilot；50 epoch 是当前 convergence candidate。

## 3. Dataset / Split

NEU-DET 与 DeepPCB 沿用 seed824 固定成员列表以及原 val/test，三种方法在同一数据集内成员完全相同。

## 4. Three Methods

- Full-SFT：`lora_r=0, freeze=0`。
- Frozen Backbone：冻结 `model.0-model.10`，训练 `model.11-model.23`。
- V-PEFT：rank=8、alpha=16、strict AO Planner，actual backend=`peft`，不允许 silent fallback。

## 5. Accuracy and Resource Results

| Dataset | Method | mAP50-95 | mAP50 | Precision | Recall | Trainable / Total | Trainable Ratio | Peak GPU Mem | Memory Saving | Time | GPU-hours | Time Ratio | Checkpoint | Adapter | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.2901 | 0.5580 | 0.7520 | 0.4788 | 2,590,994 / 2,591,010 | 100.00% | 2652 MiB | 0.00% | 182.3s | 0.05065 | 1.000x | 5.24 MiB | 0.00 MiB | PASS |
| NEU-DET | Frozen Backbone | 0.2708 | 0.5410 | 0.6636 | 0.4983 | 1,225,522 / 2,591,010 | 47.30% | 1700 MiB | 35.91% | 166.9s | 0.04635 | 0.915x | 5.24 MiB | 0.00 MiB | PASS |
| NEU-DET | V-PEFT | 0.2379 | 0.4827 | 0.7208 | 0.4158 | 613,602 / 2,772,770 | 22.13% | 2621 MiB | 1.16% | 211.7s | 0.05879 | 1.161x | 5.75 MiB | 0.73 MiB | PASS |
| DeepPCB | Full-SFT | 0.5324 | 0.8027 | 0.7696 | 0.7563 | 2,590,994 / 2,591,010 | 100.00% | 2673 MiB | 0.00% | 195.8s | 0.05439 | 1.000x | 5.24 MiB | 0.00 MiB | PASS |
| DeepPCB | Frozen Backbone | 0.4045 | 0.6861 | 0.6586 | 0.6521 | 1,225,522 / 2,591,010 | 47.30% | 1710 MiB | 36.02% | 180.7s | 0.05020 | 0.923x | 5.24 MiB | 0.00 MiB | PASS |
| DeepPCB | V-PEFT | 0.3631 | 0.5784 | 0.5217 | 0.6320 | 613,602 / 2,772,770 | 22.13% | 2642 MiB | 1.15% | 213.9s | 0.05942 | 1.093x | 5.75 MiB | 0.73 MiB | PASS |

## 6. 30 vs 50

见 `../results/e30_vs_e50.csv`。两个数据集的单 seed 方法顺序没有反转，但 50 epoch 仍未满足统一收敛门槛。

## 7. V-PEFT Trade-off

- NEU-DET: parameter reduction=76.32%, accuracy retention=82.02%, memory saving=1.16%, time change=+16.09%.
- DeepPCB: parameter reduction=76.32%, accuracy retention=68.21%, memory saving=1.15%, time change=+9.26%.

## 8. Multi-seed Statistics

未运行 seed825/826；不报告 mean/std/95% CI。

## 9. Planner Analysis

NEU-DET 与 DeepPCB 均为 Planner=ACCEPT、planner backend=vpeft、actual backend=peft、planned/applied targets=59/52，adapter 导出成功。

## 10. Convergence and Limitations

| Dataset | Method | Epoch 41-45 mean | Epoch 46-50 mean | Delta | Best epoch | Best | Last | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.263518 | 0.286780 | +0.023262 | 50 | 0.292300 | 0.292300 | NOT_CONVERGED |
| NEU-DET | Frozen Backbone | 0.239738 | 0.255286 | +0.015548 | 50 | 0.259790 | 0.259790 | NOT_CONVERGED |
| NEU-DET | V-PEFT | 0.253876 | 0.258444 | +0.004568 | 48 | 0.260200 | 0.257910 | CONVERGED_OR_PLATEAU |
| DeepPCB | Full-SFT | 0.481210 | 0.483938 | +0.002728 | 45 | 0.507880 | 0.470700 | CONVERGED_OR_PLATEAU |
| DeepPCB | Frozen Backbone | 0.390756 | 0.404088 | +0.013332 | 47 | 0.405300 | 0.404210 | NOT_CONVERGED |
| DeepPCB | V-PEFT | 0.296432 | 0.313152 | +0.016720 | 39 | 0.354620 | 0.309640 | NOT_CONVERGED |

本阶段仍为单 seed；`EXTEND_ALL_TO_75`，因此 50 epoch 不能称为 final P1 result，也不能声明任一方法普遍优于其他方法。

## 11. P1 Conclusion

六组公平闭环均 PASS，但只有 2/6 达到固定 plateau 判据。`MULTISEED_READY = NO`，下一步必须继续统一预算审计。
