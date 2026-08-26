# C3 P2 Report — Seed824 Scaling Gate

## 1. Research Question

Which method is suitable at which industrial few-shot sample scale, without declaring a universal winner?

## 2. P1 Starting Point

P1 completed 18/18 at 100 images. P2 reuses those cells only after byte-identical training-list, fixed val/test, and locked-protocol audits.

## 3. Scaling Protocol

YOLO11n, 100 epochs, batch 8, imgsz 640, AdamW, lr0=0.001, weight decay=0.0005, cosine scheduler, seed824. Full/Frozen/V-PEFT definitions are unchanged from P1.

## 4. Nested Few-shot Split

Both datasets pass strict `10 ⊂ 50 ⊂ 100 ⊂ 500`; all four scales cover 6/6 classes. The 100-image files are byte-identical to P1.

## 5. 10/50/100/500 Results

| Dataset | Images | Method | mAP50-95 | mAP50 | Trainable Params | Peak GPU MiB | Time (s) | Source |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | 10 | Full-SFT | 0.1214 | 0.2871 | 2,590,994 | 2682.9 | 192.7 | new_p2_seed824 |
| NEU-DET | 10 | Frozen Backbone | 0.1432 | 0.3215 | 1,225,522 | 1607.7 | 175.9 | new_p2_seed824 |
| NEU-DET | 10 | V-PEFT | 0.1054 | 0.2323 | 613,602 | 2641.9 | 220.3 | new_p2_seed824 |
| NEU-DET | 50 | Full-SFT | 0.2793 | 0.5551 | 2,590,994 | 2682.9 | 267.6 | new_p2_seed824 |
| NEU-DET | 50 | Frozen Backbone | 0.2237 | 0.4815 | 1,225,522 | 1710.1 | 241.4 | new_p2_seed824 |
| NEU-DET | 50 | V-PEFT | 0.2237 | 0.4765 | 613,602 | 2641.9 | 303.2 | new_p2_seed824 |
| NEU-DET | 100 | Full-SFT | 0.3293 | 0.6401 | 2,590,994 | 2652.2 | 349.7 | reused_p1_seed824_e100 |
| NEU-DET | 100 | Frozen Backbone | 0.2901 | 0.5736 | 1,225,522 | 1699.8 | 317.4 | reused_p1_seed824_e100 |
| NEU-DET | 100 | V-PEFT | 0.3280 | 0.6239 | 613,602 | 2621.4 | 395.1 | reused_p1_seed824_e100 |
| NEU-DET | 500 | Full-SFT | 0.3956 | 0.7170 | 2,590,994 | 2652.2 | 1159.3 | new_p2_seed824 |
| NEU-DET | 500 | Frozen Backbone | 0.3859 | 0.6858 | 1,225,522 | 1699.8 | 1031.9 | new_p2_seed824 |
| NEU-DET | 500 | V-PEFT | 0.3904 | 0.6962 | 613,602 | 2621.4 | 1216.4 | new_p2_seed824 |
| DeepPCB | 10 | Full-SFT | 0.2492 | 0.4638 | 2,590,994 | 2693.1 | 203.7 | new_p2_seed824 |
| DeepPCB | 10 | Frozen Backbone | 0.2068 | 0.3735 | 1,225,522 | 1628.2 | 188.0 | new_p2_seed824 |
| DeepPCB | 10 | V-PEFT | 0.1908 | 0.3272 | 613,602 | 2662.4 | 243.8 | new_p2_seed824 |
| DeepPCB | 50 | Full-SFT | 0.5577 | 0.8399 | 2,590,994 | 2693.1 | 282.4 | new_p2_seed824 |
| DeepPCB | 50 | Frozen Backbone | 0.3803 | 0.6567 | 1,225,522 | 1730.6 | 258.0 | new_p2_seed824 |
| DeepPCB | 50 | V-PEFT | 0.3562 | 0.5718 | 613,602 | 2652.2 | 317.4 | new_p2_seed824 |
| DeepPCB | 100 | Full-SFT | 0.6345 | 0.9208 | 2,590,994 | 2672.6 | 369.0 | reused_p1_seed824_e100 |
| DeepPCB | 100 | Frozen Backbone | 0.4754 | 0.7938 | 1,225,522 | 1710.1 | 335.6 | reused_p1_seed824_e100 |
| DeepPCB | 100 | V-PEFT | 0.5115 | 0.7794 | 613,602 | 2641.9 | 412.1 | reused_p1_seed824_e100 |
| DeepPCB | 500 | Full-SFT | 0.7006 | 0.9660 | 2,590,994 | 2652.2 | 1154.6 | new_p2_seed824 |
| DeepPCB | 500 | Frozen Backbone | 0.5885 | 0.9200 | 1,225,522 | 1699.8 | 1057.4 | new_p2_seed824 |
| DeepPCB | 500 | V-PEFT | 0.6571 | 0.9275 | 613,602 | 2621.4 | 1241.5 | new_p2_seed824 |

## 6. Multi-seed Statistics

Not reported at this gate: only seed824 was run. `MULTISEED_READY=YES`; seed825/826 remain not run.

## 7. Accuracy Retention

- NEU V-PEFT retention @10/50/100/500: 86.77% / 80.07% / 99.62% / 98.68%.
- DeepPCB V-PEFT retention @10/50/100/500: 76.56% / 63.87% / 80.62% / 93.78%.

## 8. Parameter Efficiency

V-PEFT retains the P1 structure with 613,602 trainable parameters versus 2,590,994 for Full-SFT (76.32% reduction) at every sample size. Accuracy per trainable parameter remains sample- and dataset-dependent.

## 9. Memory / Time Efficiency

- NEU V-PEFT memory saving: +1.53% / +1.53% / +1.16% / +1.16%; time change: +14.34% / +13.32% / +13.00% / +4.93%.
- DeepPCB V-PEFT memory saving: +1.14% / +1.52% / +1.15% / +1.16%; time change: +19.72% / +12.39% / +11.69% / +7.53%.

## 10. Dataset-dependent Behavior

The measured retention sequences differ between datasets. Split manifests report class coverage, images/class, objects/class, and object density. Those measurements support describing dataset differences, not assigning an unmeasured causal mechanism.

## 11. Empirical Crossover Analysis

- NEU Full/V-PEFT crossover region(s): none observed.
- DeepPCB Full/V-PEFT crossover region(s): none observed.

These are empirical intervals between tested sizes, not theoretical optima.

## 12. Qualitative Examples

The fixed P1 100-image qualitative comparison remains the audited 100-image reference. New-scale qualitative panels are deferred until the multi-seed P2 phase to avoid selecting examples from a one-seed curve.

## 13. Limitations

This gate has one seed at 10/50/500, only four discrete sample sizes, and a fixed 100-epoch rather than fixed-update budget. Parallel runs use exclusive identical RTX 4090 GPUs. Trend direction and crossover intervals require multi-seed confirmation.

## 14. Final P2 Conclusion

`Overall C3 P2 = IN_PROGRESS`. The seed824 scaling gate is complete and valid, but P2 PASS is withheld until the authorized multi-seed matrix, mean/std/95% CI, and final statistics are completed.
