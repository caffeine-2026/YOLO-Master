# C3 P1 对照报告（seed824，75-Epoch Convergence Candidate）

## 1. Research Question

在工业缺陷小样本条件下，V-PEFT 是否能以显著更少的可训练参数和资源成本，保持或改善 Full-SFT / Frozen Backbone 的检测性能？

## 2. Protocol

`restart_all`，YOLO11n/yolo11n.pt、每数据集 100 张训练图、75 epochs、batch 8、imgsz 640、AdamW、lr0=0.001、weight decay=0.0005、cosine scheduler、GPU0、FP32、seed824。30 epoch 是 early pilot，50 epoch 是 intermediate convergence check，75 epoch 是 current final-epoch candidate。

## 3. Dataset / Split

NEU-DET 与 DeepPCB 沿用 seed824 固定成员列表以及原 val/test，三种方法在同一数据集内成员完全相同。

## 4. Three Methods

- Full-SFT：`lora_r=0, freeze=0`。
- Frozen Backbone：冻结 `model.0-model.10`，训练 `model.11-model.23`。
- V-PEFT：rank=8、alpha=16、strict AO Planner，actual backend=`peft`，不允许 silent fallback。

## 5. Accuracy and Resource Results

| Dataset | Method | mAP50-95 | mAP50 | Precision | Recall | Trainable / Total | Trainable Ratio | Peak GPU Mem | Memory Saving | Time | GPU-hours | Time Ratio | Checkpoint | Adapter | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.3242 | 0.6025 | 0.5384 | 0.5825 | 2,590,994 / 2,591,010 | 100.00% | 2652 MiB | 0.00% | 265.8s | 0.07385 | 1.000x | 5.24 MiB | 0.00 MiB | PASS |
| NEU-DET | Frozen Backbone | 0.2870 | 0.5621 | 0.4984 | 0.5442 | 1,225,522 / 2,591,010 | 47.30% | 1700 MiB | 35.91% | 239.7s | 0.06658 | 0.902x | 5.24 MiB | 0.00 MiB | PASS |
| NEU-DET | V-PEFT | 0.2933 | 0.5792 | 0.6980 | 0.5128 | 613,602 / 2,772,770 | 22.13% | 2621 MiB | 1.16% | 298.4s | 0.08288 | 1.122x | 5.75 MiB | 0.73 MiB | PASS |
| DeepPCB | Full-SFT | 0.5951 | 0.9029 | 0.8870 | 0.8311 | 2,590,994 / 2,591,010 | 100.00% | 2673 MiB | 0.00% | 283.2s | 0.07866 | 1.000x | 5.24 MiB | 0.00 MiB | PASS |
| DeepPCB | Frozen Backbone | 0.4534 | 0.7712 | 0.7575 | 0.7192 | 1,225,522 / 2,591,010 | 47.30% | 1710 MiB | 36.02% | 254.0s | 0.07056 | 0.897x | 5.24 MiB | 0.00 MiB | PASS |
| DeepPCB | V-PEFT | 0.4633 | 0.7318 | 0.6960 | 0.6938 | 613,602 / 2,772,770 | 22.13% | 2642 MiB | 1.15% | 314.0s | 0.08721 | 1.109x | 5.75 MiB | 0.73 MiB | PASS |

## 6. 30/50/75

见 `../results/e30_e50_e75.csv`。三个预算下的方法排序为“未稳定”；单 seed 排序不能外推为总体方法优劣。

## 7. V-PEFT Trade-off

- NEU-DET: parameter reduction=76.32%, accuracy retention=90.48%, memory saving=1.16%, time change=+12.24%.
- DeepPCB: parameter reduction=76.32%, accuracy retention=77.86%, memory saving=1.15%, time change=+10.88%.

## 8. Multi-seed Statistics

未运行 seed825/826；不报告 mean/std/95% CI。

## 9. Planner Analysis

NEU-DET 与 DeepPCB 均为 Planner=ACCEPT、planner backend=vpeft、actual backend=peft、planned/applied targets=59/52，adapter 导出成功。

## 10. Convergence and Limitations

| Dataset | Method | Epoch 66-70 mean | Epoch 71-75 mean | Delta | Best epoch | Best | Last | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.321914 | 0.322986 | +0.001072 | 64 | 0.326280 | 0.321680 | CONVERGED_OR_PLATEAU |
| NEU-DET | Frozen Backbone | 0.277736 | 0.280250 | +0.002514 | 62 | 0.284140 | 0.279660 | CONVERGED_OR_PLATEAU |
| NEU-DET | V-PEFT | 0.281754 | 0.271284 | -0.010470 | 66 | 0.294180 | 0.270360 | CONVERGED_OR_PLATEAU |
| DeepPCB | Full-SFT | 0.566432 | 0.585302 | +0.018870 | 58 | 0.608530 | 0.576220 | NOT_CONVERGED |
| DeepPCB | Frozen Backbone | 0.441126 | 0.462256 | +0.021130 | 74 | 0.463270 | 0.463220 | NOT_CONVERGED |
| DeepPCB | V-PEFT | 0.376926 | 0.371006 | -0.005920 | 62 | 0.460170 | 0.363560 | CONVERGED_OR_PLATEAU |

本阶段仍为单 seed；`EXTEND_ALL_TO_100`。75 epoch 仍是 convergence candidate，不能称为 final P1 结果；下一阶段须六种条件统一扩展到 100 epoch。即使冻结预算，也不能用 seed824 声明任一方法普遍优于其他方法。

## 11. P1 Conclusion

六组公平闭环均 PASS，4/6 达到固定 plateau 判据；`MULTISEED_READY = NO`。本轮未运行 seed825/826。
