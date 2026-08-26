# C3 P1 对照报告（seed824，100-Epoch Final-Epoch Protocol）

## 1. Research Question

在工业缺陷小样本条件下，V-PEFT 是否能以显著更少的可训练参数和资源成本，保持或改善 Full-SFT / Frozen Backbone 的检测性能？

## 2. Protocol

`restart_all`，YOLO11n/yolo11n.pt、每数据集 100 张训练图、100 epochs、batch 8、imgsz 640、AdamW、lr0=0.001、weight decay=0.0005、cosine scheduler、GPU0、FP32、seed824。30 epoch = early pilot；50 epoch = intermediate；75 epoch = convergence candidate；100 epoch = final single-seed candidate。

## 3. Dataset / Split

NEU-DET 与 DeepPCB 沿用 seed824 固定成员列表以及原 val/test，三种方法在同一数据集内成员完全相同。

## 4. Three Methods

- Full-SFT：`lora_r=0, freeze=0`。
- Frozen Backbone：冻结 `model.0-model.10`，训练 `model.11-model.23`。
- V-PEFT：rank=8、alpha=16、strict AO Planner，actual backend=`peft`，不允许 silent fallback。

## 5. Accuracy and Resource Results

| Dataset | Method | mAP50-95 | mAP50 | Precision | Recall | Trainable / Total | Trainable Ratio | Peak GPU Mem | Memory Saving | Time | GPU-hours | Time Ratio | Checkpoint | Adapter | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.3293 | 0.6401 | 0.5927 | 0.6448 | 2,590,994 / 2,591,010 | 100.00% | 2652 MiB | 0.00% | 349.7s | 0.09713 | 1.000x | 5.24 MiB | 0.00 MiB | PASS |
| NEU-DET | Frozen Backbone | 0.2901 | 0.5736 | 0.4913 | 0.5862 | 1,225,522 / 2,591,010 | 47.30% | 1700 MiB | 35.91% | 317.4s | 0.08817 | 0.908x | 5.24 MiB | 0.00 MiB | PASS |
| NEU-DET | V-PEFT | 0.3280 | 0.6239 | 0.6162 | 0.5839 | 613,602 / 2,772,770 | 22.13% | 2621 MiB | 1.16% | 395.1s | 0.10976 | 1.130x | 5.76 MiB | 0.73 MiB | PASS |
| DeepPCB | Full-SFT | 0.6345 | 0.9208 | 0.9040 | 0.8558 | 2,590,994 / 2,591,010 | 100.00% | 2673 MiB | 0.00% | 369.0s | 0.10250 | 1.000x | 5.24 MiB | 0.00 MiB | PASS |
| DeepPCB | Frozen Backbone | 0.4754 | 0.7938 | 0.7525 | 0.7587 | 1,225,522 / 2,591,010 | 47.30% | 1710 MiB | 36.02% | 335.6s | 0.09322 | 0.909x | 5.24 MiB | 0.00 MiB | PASS |
| DeepPCB | V-PEFT | 0.5115 | 0.7794 | 0.7591 | 0.7177 | 613,602 / 2,772,770 | 22.13% | 2642 MiB | 1.15% | 412.1s | 0.11448 | 1.117x | 5.76 MiB | 0.73 MiB | PASS |

## 6. 30/50/75/100

见 `../results/e30_e50_e75_e100.csv`。排序为“75→100 稳定”，但全历史“30/50 与 75/100 不完全一致”。

- NEU-DET: e30=Full-SFT > Frozen Backbone > V-PEFT; e50=Full-SFT > Frozen Backbone > V-PEFT; e75=Full-SFT > V-PEFT > Frozen Backbone; e100=Full-SFT > V-PEFT > Frozen Backbone
- DeepPCB: e30=Full-SFT > Frozen Backbone > V-PEFT; e50=Full-SFT > Frozen Backbone > V-PEFT; e75=Full-SFT > V-PEFT > Frozen Backbone; e100=Full-SFT > V-PEFT > Frozen Backbone

因此三种方法的排序在 75→100 已稳定；单 seed 排序不能外推为总体方法优劣。

## 7. V-PEFT Trade-off

- NEU-DET: parameter reduction=76.32%, accuracy retention=99.62%, accuracy drop=0.0012, memory saving=1.16%, time change=+13.00%, GPU-hour change=+13.00%.
- DeepPCB: parameter reduction=76.32%, accuracy retention=80.62%, accuracy drop=0.1230, memory saving=1.15%, time change=+11.69%, GPU-hour change=+11.69%.

### Accuracy vs trainable params

V-PEFT 两个数据集均只训练 613,602 个参数，相对 Full-SFT 的 2,590,994 个可训练参数减少 76.32%。NEU-DET mAP50-95 仅低 0.0012；DeepPCB 低 0.1230。

### Accuracy vs GPU memory

V-PEFT 的峰值显存仅比 Full-SFT 低约 1.15%，未随可训练参数减少而同比下降。Frozen Backbone 的峰值显存节省约 36%，说明本协议下冻结计算路径与 adapter 路径的资源结果不同。

### Accuracy vs GPU-hours

V-PEFT 的 GPU-hours 相对 Full-SFT 在 NEU-DET 增加 13.00%，在 DeepPCB 增加 11.69%；参数效率没有转化为训练吞吐优势。

## 8. Three Questions

### Q1 — V-PEFT 是否保持明显的参数效率优势？

是。两个数据集均观测到 76.32% 的可训练参数减少。

### Q2 — 参数减少是否转化为显存或训练时间优势？

否。显存仅节省约 1.15%，训练时间和 GPU-hours 反而增加 11.69%–13.00%。

### Q3 — 两个工业数据集上的 accuracy retention 是否达到可接受范围？

结果不一致：NEU-DET 为 99.62%，接近 Full-SFT；DeepPCB 为 80.62%，绝对 mAP50-95 下降 0.1230。由于协议没有预先定义“可接受”的 retention threshold，数据支持“NEU 近乎保留、DeepPCB 有明显损失”，不支持声称两个数据集都已普遍达到可接受范围。

## 9. Multi-seed Statistics

未运行 seed825/826；不报告 mean/std/95% CI。

## 10. Planner Analysis

NEU-DET 与 DeepPCB 均为 Planner=ACCEPT、planner backend=vpeft、actual backend=peft、planned/applied targets=59/52，adapter 导出成功。

## 11. Convergence and Limitations

| Dataset | Method | Epoch 91-95 mean | Epoch 96-100 mean | Delta | Best epoch | Best | Last | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.316820 | 0.315564 | -0.001256 | 85 | 0.328660 | 0.316170 | CONVERGED_OR_PLATEAU |
| NEU-DET | Frozen Backbone | 0.254322 | 0.261958 | +0.007636 | 79 | 0.269400 | 0.266420 | CONVERGED_OR_PLATEAU |
| NEU-DET | V-PEFT | 0.307594 | 0.303062 | -0.004532 | 87 | 0.317860 | 0.303940 | CONVERGED_OR_PLATEAU |
| DeepPCB | Full-SFT | 0.588734 | 0.555780 | -0.032954 | 75 | 0.634960 | 0.547630 | CONVERGED_OR_PLATEAU |
| DeepPCB | Frozen Backbone | 0.473090 | 0.482846 | +0.009756 | 63 | 0.486680 | 0.478640 | CONVERGED_OR_PLATEAU |
| DeepPCB | V-PEFT | 0.404170 | 0.391320 | -0.012850 | 70 | 0.517760 | 0.379310 | CONVERGED_OR_PLATEAU |

本阶段仍为单 seed；`KEEP_100`。100 epoch 已冻结为 multi-seed 的统一最终预算。即使冻结预算，也不能用 seed824 声明任一方法普遍优于其他方法。

## 12. P1 Conclusion

六组公平闭环均 PASS，6/6 达到固定 plateau 判据；`MULTISEED_READY = YES`。本轮未运行 seed825/826。
