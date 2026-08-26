# C3 P2 Report — Final Multi-seed Scaling Study

## 1. Research Question

Which of Full-SFT, Frozen Backbone, and V-PEFT is most suitable in each measured industrial few-shot data regime?

## 2. P1 Starting Point

P1 supplied the immutable 100-image cells for seeds 824/825/826. All 18 are reused after P1 final validation plus split and protocol-hash checks; none was rerun.

## 3. Nested Few-shot Protocol

Both datasets retain `10 ⊂ 50 ⊂ 100 ⊂ 500`, split seed 824, fixed val/test membership, and 6/6 class coverage. Only training randomness changes across seeds. YOLO11n, 100 epochs, batch 8, imgsz 640, AdamW, lr0=0.001, weight decay=0.0005, cosine scheduling, augmentation, freeze=11, and V-PEFT r=8/planner settings are fixed.

## 4. Final 72-cell Matrix

The final matrix is 72/72 PASS: 18 immutable P2 seed824 cells, 36 new P2 seed825/826 cells at 10/50/500, and 18 P1 100-image cells. Every new V-PEFT run reports strict mode, ACCEPT/ADAPT, planner backend vpeft, actual backend peft, applied targets > 0, and a non-empty adapter export.

## 5. Multi-seed Scaling Results

| Dataset | Images | Method | mAP50-95 mean ± std | 95% CI | mAP50 mean ± std |
| --- | ---: | --- | ---: | ---: | ---: |
| NEU-DET | 10 | Full-SFT | 0.1212 ± 0.0122 | [0.0911, 0.1514] | 0.3009 ± 0.0264 |
| NEU-DET | 10 | Frozen Backbone | 0.1325 ± 0.0101 | [0.1075, 0.1576] | 0.3212 ± 0.0050 |
| NEU-DET | 10 | V-PEFT | 0.1192 ± 0.0121 | [0.0893, 0.1492] | 0.2599 ± 0.0240 |
| NEU-DET | 50 | Full-SFT | 0.2687 ± 0.0102 | [0.2434, 0.2939] | 0.5580 ± 0.0037 |
| NEU-DET | 50 | Frozen Backbone | 0.2195 ± 0.0040 | [0.2095, 0.2295] | 0.4767 ± 0.0137 |
| NEU-DET | 50 | V-PEFT | 0.2505 ± 0.0233 | [0.1927, 0.3084] | 0.5060 ± 0.0261 |
| NEU-DET | 100 | Full-SFT | 0.3329 ± 0.0039 | [0.3233, 0.3425] | 0.6378 ± 0.0020 |
| NEU-DET | 100 | Frozen Backbone | 0.2935 ± 0.0069 | [0.2764, 0.3106] | 0.5769 ± 0.0040 |
| NEU-DET | 100 | V-PEFT | 0.3203 ± 0.0081 | [0.3002, 0.3404] | 0.6110 ± 0.0129 |
| NEU-DET | 500 | Full-SFT | 0.3995 ± 0.0056 | [0.3856, 0.4134] | 0.7165 ± 0.0005 |
| NEU-DET | 500 | Frozen Backbone | 0.3769 ± 0.0084 | [0.3561, 0.3977] | 0.6794 ± 0.0071 |
| NEU-DET | 500 | V-PEFT | 0.3909 ± 0.0009 | [0.3887, 0.3930] | 0.6944 ± 0.0020 |
| DeepPCB | 10 | Full-SFT | 0.2903 ± 0.0379 | [0.1963, 0.3844] | 0.5020 ± 0.0382 |
| DeepPCB | 10 | Frozen Backbone | 0.2206 ± 0.0170 | [0.1783, 0.2629] | 0.3862 ± 0.0179 |
| DeepPCB | 10 | V-PEFT | 0.1897 ± 0.0076 | [0.1710, 0.2085] | 0.3208 ± 0.0239 |
| DeepPCB | 50 | Full-SFT | 0.5669 ± 0.0113 | [0.5389, 0.5949] | 0.8480 ± 0.0130 |
| DeepPCB | 50 | Frozen Backbone | 0.3898 ± 0.0082 | [0.3693, 0.4103] | 0.6731 ± 0.0143 |
| DeepPCB | 50 | V-PEFT | 0.3709 ± 0.0127 | [0.3393, 0.4024] | 0.5915 ± 0.0179 |
| DeepPCB | 100 | Full-SFT | 0.6486 ± 0.0129 | [0.6164, 0.6808] | 0.9226 ± 0.0026 |
| DeepPCB | 100 | Frozen Backbone | 0.4844 ± 0.0149 | [0.4473, 0.5214] | 0.8060 ± 0.0171 |
| DeepPCB | 100 | V-PEFT | 0.5166 ± 0.0142 | [0.4813, 0.5519] | 0.7958 ± 0.0161 |
| DeepPCB | 500 | Full-SFT | 0.7006 ± 0.0085 | [0.6796, 0.7217] | 0.9677 ± 0.0015 |
| DeepPCB | 500 | Frozen Backbone | 0.5979 ± 0.0086 | [0.5766, 0.6191] | 0.9257 ± 0.0051 |
| DeepPCB | 500 | V-PEFT | 0.6564 ± 0.0080 | [0.6366, 0.6762] | 0.9282 ± 0.0016 |

## 6. Accuracy Retention

- NEU-DET V-PEFT retention @10/50/100/500: 98.33% / 93.26% / 96.20% / 97.83%.
- DeepPCB V-PEFT retention @10/50/100/500: 65.34% / 65.42% / 79.65% / 93.69%.

The 50-image ratio-of-means dip remains for NEU=True, but only 2/3 seeds are below their own 10-image retention and 1/3 are below their own 100-image retention; it is therefore not a directionally stable universal dip. For DeepPCB, the 50-image ratio-of-means retention is slightly above 10 images and below 100, so the seed824-specific 50-image minimum does not persist. The 100→500 mean-retention increase is NEU=True (2/3 paired seeds) and DeepPCB=True (3/3 paired seeds).

## 7. Parameter Efficiency

V-PEFT uses 613,602 trainable parameters versus 2,590,994 for Full-SFT, a fixed 76.32% reduction at all sizes and seeds. This structural advantage is stable; the accuracy obtained per trainable parameter remains dataset- and sample-size-dependent.

## 8. GPU Memory / Time Efficiency

- NEU-DET V-PEFT memory saving @10/50/100/500: +1.53% / +1.53% / +1.16% / +1.16%; time change: +15.23% / +12.66% / +13.04% / +5.67%.
- DeepPCB V-PEFT memory saving @10/50/100/500: +1.14% / +1.52% / +1.28% / +1.16%; time change: +19.57% / +15.94% / +13.37% / +8.25%.

The large trainable-parameter reduction does not translate into a large memory reduction in this implementation. Training-time direction is reported from the measured mean changes and is not inferred from parameter count.

## 9. Very-low / Mid / Higher-data Regimes

- Very-low (10): NEU Frozen−Full paired deltas are +0.0218/+0.0223/-0.0102; direction=mixed. This determines whether the seed824 Frozen lead is stable.
- Mid (50–100): NEU has a 50-image mean-retention dip without consistent paired-seed direction; DeepPCB is nearly flat at 10–50 and then improves at 100.
- Higher measured scale (500): DeepPCB V-PEFT−Full paired deltas are -0.0435/-0.0441/-0.0451; mean=-0.0442, CI=[-0.0462, -0.0423].

These are empirical regimes across four tested sizes, not theoretical crossover points.

## 10. NEU vs DeepPCB

NEU retention exceeds DeepPCB for all 12 paired size×seed comparisons (12/12), so the dataset ordering is stable in this matrix. At 500 images, measured object density is NEU=3.13 versus DeepPCB=7.80; median normalized box area is NEU=0.1064 versus DeepPCB=0.0028. Class balance, object-size distributions, luminance dispersion, and histogram diversity are provided in `dataset_characteristics.csv`. They document measurable differences but do not establish a causal mechanism for retention differences; any mechanism remains a hypothesis.

## 11. Paired Seed Analysis

All three method pairs are compared within seed for every dataset/size. The primary evidence is three deltas, mean delta, 95% t interval, and direction consistency. With n=3, no p-value claim is used.

## 12. Qualitative Evidence

The fixed P1 100-image qualitative panels remain the checkpoint-matched reference for the reused 100-image cells. P2 scaling conclusions are based on fixed-test quantitative comparisons; no post-hoc best-looking sample selection was introduced.

## 13. Limitations

Only three seeds and four discrete sample sizes are tested. Confidence intervals are wide when between-seed variance is large. The 100-epoch budget fixes epochs rather than optimizer updates, and simple histogram/luminance diversity measures do not capture semantic image diversity. Dataset-characteristic associations are descriptive, not causal.

## 14. Final P2 Conclusion

`Overall C3 P2 = PASS`. The fair 72-cell matrix, statistics, paired analysis, resource scaling, figures, integrity evidence, and report are complete. The results support regime- and dataset-specific method selection; they do not support a universal winner or claims that V-PEFT is faster or materially more memory-efficient.
