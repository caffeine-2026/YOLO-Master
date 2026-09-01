# C3 augmentation ablation

This delivery evaluates accuracy-first V-PEFT (613,602 trainable parameters) with validation-selected, grayscale-safe augmentation. It preserves commit `bf6c7c508635dec0be849aedaa3eac5d88ed220d` and its experiment files unchanged.

- Korean research report: [docs/AUGMENTATION_ABLATION_REPORT.md](docs/AUGMENTATION_ABLATION_REPORT.md)
- Chinese GitHub Issue #2 follow-up: [docs/GITHUB_ISSUE_2_FOLLOWUP_ZH.md](docs/GITHUB_ISSUE_2_FOLLOWUP_ZH.md)
- Executed commands: [docs/EXECUTED_COMMANDS.md](docs/EXECUTED_COMMANDS.md)
- Preregistered protocol: [config/augmentation_protocol.yaml](config/augmentation_protocol.yaml)
- Frozen validation selection: [results/frozen_selection.json](results/frozen_selection.json)
- Raw and aggregate result tables: [results](results)
- Locked-test JSON: [evaluations](evaluations)
- Figures and embedded source rows: [figures](figures)
- Validator/test evidence: [evidence](evidence)
- Preserved failures and reruns: [failures](failures)
- Local checkpoints and training plots: [artifacts](artifacts) (intentionally gitignored; hashes are in each run's `artifact_manifest.json` and `metrics.json`)

Headline result: DeepPCB selected `medium` and improved 100-shot locked-test mAP50-95 by 0.10960 over the preserved accuracy-first baseline (paired 95% CI [0.05316, 0.16605]). NEU-DET selected `mild`, but changed mAP50-95 by -0.01433 versus that preserved baseline (95% CI [-0.06441, 0.03575]); it is not an accuracy improvement.

The term used here is **empirical upper bound within the tested augmentation range**, not theoretical upper bound.
