# C3 P2 — Industrial Few-shot Scaling

P2 studies Full-SFT, Frozen Backbone, and V-PEFT at 10/50/100/500 training images on NEU-DET and DeepPCB. Splits are nested around the immutable P1 seed824 100-image membership. This initial stage runs seed824 only; seed825/826 are gated on the resulting curve.

Primary entry points:

- `tools/prepare_scaling_splits.py`: deterministic nested split and manifest generation.
- `tools/run_scaling.py`: one immutable training/evaluation run.
- `tools/run_seed824_suite.py`: six-GPU seed824 scheduler for 10/50/500 only.
- `tools/summarize_scaling.py`: P1 reuse, result tables, figures, and analysis.
- `tools/validate_p2.py`: independent seed824-stage evidence gate.

Local checkpoints remain under the ignored `smoke/c3/p2/artifacts/` tree. Reproducibility evidence and compact logs are versioned.
