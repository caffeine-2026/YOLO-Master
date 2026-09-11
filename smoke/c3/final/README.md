# C3 Research Evidence Delivery

This directory is the final evidence index for the DeepPCB/NEU-DET C3 research experiments, not a web demo. Existing valid experiments were not rerun; instead, the original full logs, resolved configs, `args.yaml`, epoch CSVs, locked-test JSON, resource/timing JSON, artifact SHA-256 values, and checkpoints were cross-validated. The final integrated validator returned `PASS`, with all 72/72 training cells and 72/72 `best.pt` load checks passing.

## 1. P1 three-way comparison

All values are means over seeds 824/825/826 with 100 images, 100 epochs, batch size 8, imgsz 640, AdamW, FP32, identical augmentation, and a fixed test split. Brackets show the two-sided 95% Student-t CI for mAP (n=3, df=2). Accuracy retention is the ratio to the Full-SFT mAP50-95 mean on the same dataset.

| Dataset | Method | mAP50-95 mean [95% CI] | mAP50 mean [95% CI] | Trainable / total params | Peak MiB | Time s / GPU-h | Retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NEU-DET | Full-SFT | 0.332935 [0.323344, 0.342525] | 0.637783 [0.632831, 0.642735] | 2,590,994 / 2,591,010 | 2,652.16 | 356.168 / 0.098935 | 100.00% |
| NEU-DET | Frozen Backbone | 0.293499 [0.276374, 0.310623] | 0.576870 [0.566847, 0.586892] | 1,225,522 / 2,591,010 | 1,699.84 | 325.120 / 0.090311 | 88.15% |
| NEU-DET | V-PEFT | 0.320288 [0.300222, 0.340353] | 0.611046 [0.579120, 0.642973] | 613,602 / 2,772,770 | 2,621.44 | 402.629 / 0.111841 | 96.20% |
| DeepPCB | Full-SFT | 0.648599 [0.616436, 0.680763] | 0.922556 [0.916093, 0.929019] | 2,590,994 / 2,591,010 | 2,665.81 | 373.784 / 0.103829 | 100.00% |
| DeepPCB | Frozen Backbone | 0.484355 [0.447318, 0.521391] | 0.805972 [0.763410, 0.848533] | 1,225,522 / 2,591,010 | 1,706.67 | 340.116 / 0.094477 | 74.68% |
| DeepPCB | V-PEFT | 0.516607 [0.481275, 0.551939] | 0.795849 [0.755903, 0.835795] | 613,602 / 2,772,770 | 2,631.68 | 423.744 / 0.117707 | 79.65% |

V-PEFT uses 76.32% fewer trainable parameters than Full-SFT, but under this implementation GPU-memory savings were only 1.16% on NEU and 1.28% on DeepPCB, while training time increased by 13.04% and 13.37%, respectively. At 100 shots, the paired V-PEFT−Frozen mAP50-95 mean difference was +0.02679 on NEU (95% CI [0.00249, 0.05109]) and +0.03225 on DeepPCB ([0.02369, 0.04082]). We therefore interpret V-PEFT as a trade-off across accuracy, trainable parameters, memory, and time, not as a universal winner.

Source data and statistics: [P1 all runs](../p1/results/p1_all_runs.csv), [P1 summary](../p1/results/p1_summary.csv), [paired analysis](../p1/results/paired_full_vs_vpeft.csv), [P1 report](../p1/docs/C3_P1_REPORT.md).

## 2. P2 scaling curve

The table below reports mean mAP50-95 over seeds 824/825/826. The 100-shot cells were reused from validated P1 results, and the remaining 54 cells were reused after validating the previously completed experiments.

| Dataset | Images | Full-SFT | Frozen Backbone | V-PEFT | V-PEFT retention vs Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| NEU-DET | 10 | 0.1212 | 0.1325 | 0.1192 | 98.33% |
| NEU-DET | 50 | 0.2687 | 0.2195 | 0.2505 | 93.26% |
| NEU-DET | 100 | 0.3329 | 0.2935 | 0.3203 | 96.20% |
| NEU-DET | 500 | 0.3995 | 0.3769 | 0.3909 | 97.83% |
| DeepPCB | 10 | 0.2903 | 0.2206 | 0.1897 | 65.34% |
| DeepPCB | 50 | 0.5669 | 0.3898 | 0.3709 | 65.42% |
| DeepPCB | 100 | 0.6486 | 0.4844 | 0.5166 | 79.65% |
| DeepPCB | 500 | 0.7006 | 0.5979 | 0.6564 | 93.69% |

- CSV: [72-cell raw table](../p2/results/p2_all_runs.csv), [mean/95% CI](../p2/results/p2_summary.csv), [retention](../p2/results/retention_multiseed.csv), [paired deltas](../p2/results/paired_analysis.csv).
- Curves: [NEU mAP50-95](../p2/visualizations/final/neu_map5095_multiseed.png), [NEU mAP50](../p2/visualizations/final/neu_map50_multiseed.png), [DeepPCB mAP50-95](../p2/visualizations/final/deeppcb_map5095_multiseed.png), [DeepPCB mAP50](../p2/visualizations/final/deeppcb_map50_multiseed.png).
- Efficiency: [NEU accuracy/params](../p2/visualizations/final/neu_accuracy_vs_params_multiseed.png), [DeepPCB accuracy/params](../p2/visualizations/final/deeppcb_accuracy_vs_params_multiseed.png), [P2 report](../p2/docs/C3_P2_REPORT.md).

## 3. Planner and solver evidence

The planner flow diagram is included in [Planner flow and solver audit](../p0/docs/PLANNER_FLOW_AND_SOLVER_AUDIT_20260831.md).

| Dataset | Requested → effective | Decision | Budget | Planned / applied modules | Ranks | Result |
| --- | --- | --- | ---: | ---: | --- | --- |
| NEU-DET | AO → AO | ACCEPT | 2,100,000 | 59 / 52 | 8 | completed |
| DeepPCB | AO → AO | ACCEPT | 2,100,000 | 59 / 52 | 8 | completed |
| NEU-DET | DCO → DCO | ACCEPT | 2,100,000 | 59 / 52 | 8/16/32/48/64 | completed after fix |
| DeepPCB | DCO → DCO | ACCEPT | 2,100,000 | 59 / 52 | 8/16/32/48/64 | completed after fix |
| NEU-DET | MIPR → AO | ACCEPT | 2,100,000 | 59 / 52 | 8 | OR-Tools ImportError fallback |

Structured evidence: [solver audit JSON](../p0/evidence/solver_audit_20260831.json). Failure, repair, and rerun records are documented in [FAILURE_REPAIR_RERUN.md](FAILURE_REPAIR_RERUN.md).

## 4. LOVO boundary

`predicted_delta=0.06602954545454547`, confidence `0`, state `cold_start`, source `default_prior`, observation count `0`, and `uses_learned_evidence=false`. This is neither a measured ΔmAP nor evidence of a V-PEFT improvement. The 24 historical V-PEFT metadata records in P1/P2 retain their original `null` fields. Test results were not used for calibration, and there are fewer than five independent formal calibration observations; therefore, **LOVO calibration pending**.

## 5. Evidence index

- Actual GPU preflight: [gpu_preflight.json](evidence/gpu_preflight.json).
- Consolidated 79 original command files: [raw_command_manifest.json](evidence/raw_command_manifest.json). Each run's `command.txt` source and SHA-256 are preserved.
- Original full logs/config/metrics/resources/time: `smoke/c3/p1/logs/<run_id>/` or `smoke/c3/p2/logs/<run_id>/` for each run; paths are recorded in the [integrated validation JSON](evidence/research_delivery_validation.json).
- P0 full training logs: [P0 logs index](../p0/logs/README.md).
- P1 per-seed results and 95% CI: [P1 all runs](../p1/results/p1_all_runs.csv), [P1 summary](../p1/results/p1_summary.csv).
- P2 per-seed results and 95% CI: [P2 all runs](../p2/results/p2_all_runs.csv), [P2 summary](../p2/results/p2_summary.csv).
- Checkpoint/adapter SHA-256: each run's `artifact_manifest.json`; the integrated validator actually loaded all 72 `best.pt` files.
- Execution and validation commands: [EXECUTED_COMMANDS.md](EXECUTED_COMMANDS.md).
- Chinese report for GitHub publication: [GITHUB_PROGRESS_ZH.md](GITHUB_PROGRESS_ZH.md).

## 6. Completion boundary and limitations

- P0: complete. Evidence covers V-PEFT on both datasets, AO/DCO, the MIPR fallback, failure logs, and the DCO capacity fix.
- P1: complete. Two datasets × three strategies × three seeds = 18/18, with an identical protocol, isolated test split, and mean/95% CI.
- P2: complete. Two datasets × four scales × three strategies × three seeds = 72/72, with nested-split and curve/CSV consistency verified.
- LOVO: incomplete. There are no formal calibration observations, only a cold-start prior.
- With three seeds, confidence intervals may be wide; 100 fixed epochs do not equalize optimizer update counts across scales.
- V-PEFT provides a large trainable-parameter reduction, but no memory/time benefit was confirmed in this implementation.

Final machine-validation result: [research_delivery_validation.json](evidence/research_delivery_validation.json).

The related LoRA/MoLoRA/PEFT/Planner pytest suite completed with 295 passed, 7 skipped, 2 warnings, and 0 failed. Detailed commands are in [EXECUTED_COMMANDS.md](EXECUTED_COMMANDS.md).
