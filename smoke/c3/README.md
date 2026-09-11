# C3 | Industrial Defect Detection with V-PEFT in Few-shot Settings

This package is the reproducible delivery for C3, covering NEU-DET and DeepPCB with Full-SFT, Frozen Backbone, and V-PEFT under a controlled 100-epoch, three-seed protocol and nested 10/50/100/500-image scaling study.

- Final delivery summary: [`docs/C3_FINAL_DELIVERY_SUMMARY.md`](docs/C3_FINAL_DELIVERY_SUMMARY.md)
- P0 smoke validation and planner evidence: [`p0/`](p0/)
- P1 three-way multi-seed comparison: [`p1/`](p1/)
- P2 nested few-shot scaling study: [`p2/`](p2/)
- Cross-stage evidence validation: [`final/`](final/)
- Configuration index: [`config/README.md`](config/README.md)
- Tooling index: [`tools/README.md`](tools/README.md)
- Evidence index: [`evidence/README.md`](evidence/README.md)

Raw datasets, local training checkpoints, caches, and environment directories are intentionally excluded from Git. The contribution is an experimental and evaluation extension, reproduction tooling, statistical analysis, and an evidence-backed characterization of the industrial few-shot applicability boundary; it does not claim to have invented V-PEFT, to establish a universal winner, or to achieve state of the art.
