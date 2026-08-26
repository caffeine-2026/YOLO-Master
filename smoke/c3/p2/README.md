# C3 P2 — Industrial Few-shot Scaling

P2 studies Full-SFT, Frozen Backbone, and V-PEFT at 10/50/100/500 training images on NEU-DET and DeepPCB. Splits are nested around the immutable P1 100-image membership. The final matrix contains seeds 824/825/826: P2 seed824 and all P1 100-image runs are reused unchanged, while seeds 825/826 were added only for 10/50/500 images.

Primary entry points:

- `tools/prepare_scaling_splits.py`: deterministic nested split and manifest generation.
- `tools/run_scaling.py`: one immutable training/evaluation run.
- `tools/run_seed824_suite.py`: six-GPU seed824 scheduler for 10/50/500 only.
- `tools/summarize_scaling.py`: historical seed824 result tables, figures, and analysis.
- `tools/validate_p2.py`: independent historical seed824-stage evidence gate.
- `tools/run_multiseed_suite.py`: exclusive-GPU scheduler for the exact 36 seed825/826 runs.
- `tools/summarize_multiseed.py`: final 72-cell tables, statistics, paired analysis, figures, and report.
- `tools/validate_p2_final.py`: independent final multi-seed evidence gate.

Local checkpoints remain under the ignored `smoke/c3/p2/artifacts/` tree. Reproducibility evidence and compact logs are versioned. The final gate is recorded in `evidence/p2_final_validation.json`.
