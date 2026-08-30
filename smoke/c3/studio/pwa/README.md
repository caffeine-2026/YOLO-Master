# C3 Edge Lab PWA

Installable, account-independent mobile edge inference for the validated C3 V-PEFT models. The PWA reuses the canonical static ONNX exports without retraining or editing P0/P1/P2.

## Product surfaces

- **Live** — camera-pipeline runtime demo with one inference in flight, FPS and stage timing; not an accuracy-validation surface
- **Photo** — dataset-scope inference for up to nine local images, result selection and JSON export
- **Bench** — five warmups plus 30/100 model-only runs with mean, p50, p95, min, max and FPS
- **Models** — IndexedDB cache, SHA-256 verification, backend detection, download and load test

The runtime attempts WebGL and then falls back to the broadly compatible WASM provider. Browser APIs do not expose thermal state or per-operator hardware placement, so the UI does not claim either.

The shipped artifacts are the canonical P1 100-shot, seed-824, 100-epoch V-PEFT checkpoints. Both use the protocol's fixed qualitative confidence threshold of 0.25. NEU-DET input is limited to 200×200 grayscale hot-rolled steel surface crops; DeepPCB input is limited to 640×640 binarized bare-board tested-image crops. Model candidates are never presented as confirmed defects or quality pass/fail decisions.

## Local development

```bash
cd smoke/c3/studio/pwa
npm install
npm run dev -- --host 127.0.0.1 --port 7864
```

The ignored model artifacts must exist before inference:

```text
public/models/neu_vpeft_640.onnx
public/models/deeppcb_vpeft_640.onnx
```

Rebuild them through `../miniprogram/tools/export_models.py` and copy the verified artifacts from `../miniprogram/dist/models/`. The SHA-256 values are pinned in `lib/models.ts`.

## Validation

```bash
npm test
npm run lint
npm run build
```

Camera access and PWA installation require HTTPS in production; localhost is accepted for development. Models are verified before entering IndexedDB, while selected photos remain in memory and are not uploaded.
