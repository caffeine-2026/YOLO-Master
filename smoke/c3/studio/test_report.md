# C3 Industrial PEFT Studio v0.1 — Test Report

Date: 2026-08-29 (Asia/Shanghai)

Result: **PASS**

## Scope and safety boundary

- Tested only the new read-only Studio under `smoke/c3/studio/`.
- No training command was executed.
- No P0/P1/P2 CSV, JSON, checkpoint, adapter, split, log, or visualization was edited.
- The application listens on `127.0.0.1` only and uses `share=False`.
- Dashboard tabs remained CPU-only; Torch/Ultralytics are imported lazily by Live Inference.

## Environment

| Component | Version / result |
|---|---|
| Python | 3.10.12 (`.venv`) |
| Gradio | 6.26.0 |
| pandas | 2.3.3 |
| matplotlib | 3.10.9 |
| Pillow | 12.3.0 |
| Torch | 2.13.0+cu130 |
| CUDA available | Yes |
| Ultralytics | 8.4.101 (repository environment) |

Gradio was the only missing top-level Studio dependency. The selected Gradio release retained the existing Pillow 12 line; Torch, CUDA, and Ultralytics were not upgraded or replaced.

## Automated checks

Command:

```bash
C3_STUDIO_FORCE_CPU=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m unittest smoke.c3.studio.tests.test_studio -v
```

Result: **10 tests passed**.

| Requirement | Result | Evidence |
|---|---|---|
| 1. App can start | PASS | `http://127.0.0.1:7860/` returned HTTP 200; listener bound to loopback only |
| 2. P1 CSV loads | PASS | 18 all-run rows, 6 summary rows, 2 tradeoff rows validated |
| 3. P2 CSV loads | PASS | 72 all-run rows, 24 summary rows, 8 retention rows, 24 paired rows validated |
| 4. Dataset dropdown | PASS | Browser changed Comparison and Scaling from NEU-DET to DeepPCB |
| 5. Sample-size switch | PASS | Browser changed Comparison from 100 to 500; resulting state showed `DeepPCB · 500-shot` |
| 6. Charts render | PASS | Comparison 4-panel chart and all 5 scaling metrics generated; browser exposed rendered plot images |
| 7. Planner loads | PASS | Existing metrics JSON yielded ACCEPT / vpeft / peft / 59 planned / 52 applied |
| 8. CPU pages work without GPU | PASS | Full app construction and all dashboard tests passed with `C3_STUDIO_FORCE_CPU=1`; initial server GPU usage stayed at 4 MiB / 0% on all GPUs |
| 9. Real inference | PASS | NEU-DET V-PEFT, final 100-epoch checkpoint, conf 0.25, imgsz 640, GPU 0; 2 detections; UI latency 712.6 ms |
| 10. P0/P1/P2 unchanged | PASS | `git status` / diff scope contained only `smoke/c3/studio/` |

## Model and checkpoint validation

- All six canonical P1 combinations (2 datasets × 3 methods) resolved to a seed-824, 100-epoch PASS run.
- Every selected `weights/best.pt` SHA-256 matched `p1_all_runs.csv`.
- The V-PEFT checkpoint was validated for active `peft` backend, non-empty LoRA tensors, and applied target metadata before prediction.
- Independent adapter bundles were not modified or re-exported.

## Browser checks

- Six tabs visible and selectable.
- Overview displayed P0/P1/P2 PASS and P2 Matrix 72/72 PASS.
- Comparison dataset and sample-size controls updated table and chart.
- Scaling dataset and metric controls updated mean ± 95% CI view.
- Live Inference displayed original, annotated result, class, confidence, bounding boxes, latency, threshold, device, and repository-relative checkpoint path.
- Evidence page displayed repository-relative paths only and no secret/environment values.

Screenshots:

- `smoke/c3/studio/screenshots/overview.png`
- `smoke/c3/studio/screenshots/comparison.png`
- `smoke/c3/studio/screenshots/scaling.png`
- `smoke/c3/studio/screenshots/inference.png`
- `smoke/c3/studio/screenshots/planner.png`

## Static quality checks

```bash
.venv/bin/ruff check smoke/c3/studio
```

Result: **All checks passed**.
