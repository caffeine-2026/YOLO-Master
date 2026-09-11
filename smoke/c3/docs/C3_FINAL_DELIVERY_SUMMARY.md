# C3 Final Delivery Summary

## 1. Task objective

C3 evaluates V-PEFT for industrial defect detection under limited labeled data. The delivery compares Full-SFT, Frozen Backbone, and V-PEFT on NEU-DET and DeepPCB, then measures how the comparison changes across nested 10/50/100/500-image training sets.

## 2. Completion status

- P0: PASS. The two-dataset smoke path, V-PEFT runtime, planner/solver behavior, fallback, guardrails, and failure-repair evidence are recorded.
- P1: PASS. The controlled three-way comparison contains 18/18 valid runs: two datasets, three methods, and seeds 824/825/826.
- P2: PASS. The nested scaling matrix contains 72/72 valid cells: two datasets, four sample sizes, three methods, and three seeds.
- Final validation: nested split integrity, protocol fairness, multi-seed completeness, result tables, figures, and artifact manifests pass their recorded gates.

## 3. Final protocol

The final protocol uses 100 epochs, batch size 8, image size 640, AdamW, FP32, deterministic execution, and seeds 824/825/826. Within each dataset and sample size, the three methods use the same sample membership and evaluation split. The 10/50/100/500-image training lists are strictly nested; P1's validated 100-image cells are reused in P2 without retraining.

## 4. Key results

V-PEFT reduces trainable parameters by 76.32% relative to Full-SFT. Its mean Full-SFT accuracy retention (mAP50-95) is:

| Dataset | 10 images | 50 images | 100 images | 500 images |
| --- | ---: | ---: | ---: | ---: |
| NEU-DET | 98.33% | 93.26% | 96.20% | 97.83% |
| DeepPCB | 65.34% | 65.42% | 79.65% | 93.69% |

Across the tested scaling protocol, V-PEFT saves only about 1.14%-1.53% peak GPU memory and takes about 5.67%-19.57% longer to train than Full-SFT.

## 5. Core contribution

The contribution is an experimental and evaluation extension around existing V-PEFT capabilities: dataset configuration and conversion, reproducible experiment wrappers, planner and guardrail evidence, a controlled three-way comparison, multi-seed statistics, nested few-shot scaling, visualizations, and machine-readable validation reports. It does not claim to re-invent V-PEFT.

## 6. Negative findings

Trainable-parameter efficiency does not automatically produce GPU-memory or training-time efficiency. Accuracy retention is also dataset- and sample-size-dependent: NEU-DET retains Full-SFT accuracy more consistently, while DeepPCB is substantially more sensitive at 10 and 50 images and improves with scale. The evidence therefore does not support describing V-PEFT as a universal winner.

## 7. Reproduction entry

- Dataset preparation and P0 validation: [`../p0/tools/`](../p0/tools/)
- P1 protocol, runners, analysis, and validation: [`../p1/config/`](../p1/config/), [`../p1/scripts/`](../p1/scripts/)
- P2 protocol, nested split preparation, runners, analysis, and validation: [`../p2/config/`](../p2/config/), [`../p2/tools/`](../p2/tools/)
- Primary P1 results and report: [`../p1/results/`](../p1/results/), [`../p1/docs/C3_P1_REPORT.md`](../p1/docs/C3_P1_REPORT.md)
- Primary P2 results, report, and final figures: [`../p2/results/`](../p2/results/), [`../p2/docs/C3_P2_REPORT.md`](../p2/docs/C3_P2_REPORT.md), [`../p2/visualizations/final/`](../p2/visualizations/final/)
- Cross-stage validator and evidence: [`../final/tools/validate_research_delivery.py`](../final/tools/validate_research_delivery.py), [`../final/evidence/research_delivery_validation.json`](../final/evidence/research_delivery_validation.json)

All entries are repository-relative. Users must obtain NEU-DET and DeepPCB independently and configure their local dataset locations; raw datasets and large training checkpoints are not part of this delivery.

## 8. Known limitations

- The empirical scope is limited to NEU-DET and DeepPCB and does not establish general behavior on other vision tasks or industrial domains.
- Three seeds provide a reproducibility check but still yield wide confidence intervals in some cells.
- A fixed 100-epoch budget does not equalize optimizer-update counts across sample sizes.
- Parameter efficiency is not equivalent to compute efficiency in the tested implementation.
- LOVO remains at a cold-start prior without enough independent calibration observations.
- No state-of-the-art claim is made.
