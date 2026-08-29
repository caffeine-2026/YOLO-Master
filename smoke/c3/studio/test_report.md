# C3 Industrial PEFT Studio v0.1 — Test Report

Date: 2026-08-30 (Asia/Shanghai)

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
| 9. Real inference | PASS | Existing single-image path and new realtime-frame function both used the final NEU-DET V-PEFT checkpoint at conf 0.25/imgsz 640. A real NEU frame produced 3 detections; CPU safety mode took 259.7 ms/frame and warmed server GPU inference took 71.6 ms end to end. Stage timing and FPS rendered. |
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
- The nested **Realtime Camera** tab rendered after server restart with browser webcam input, live detection output, and FPS/stage-timing status. Camera permission was intentionally left for the user; frame processing was independently exercised with a real NEU image.
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

## 微信小程序端侧交付（C3 Edge Lab）

实现范围：`smoke/c3/studio/miniprogram/`，包含 Live、Photo、Bench、Models 四页、微信原生 ONNX 推理、JavaScript class-aware NMS、模型缓存/SHA-256 校验、JSON 报告导出与可选照片 fallback API。

### 模型导出与 parity

两个模型均从 P1 CSV 指向的 seed-824、100-epoch PASS V-PEFT checkpoint 导出。导出脚本先复制 checkpoint 到临时目录，再执行 LoRA merge；导出前后重新计算源 checkpoint SHA-256，源文件未改变。ONNX 使用固定 `1×3×640×640` 输入、opset 12、batch 1、graph 内无 NMS。

| Dataset | ONNX bytes | SHA-256 | Real-image parity |
|---|---:|---|---|
| NEU-DET | 10,568,717 | `09fbeaaa79c17a0146f030945274f37774696e1a2dc61c9c18ae80e0f68c065d` | PASS；3 detections；max score delta `6.97e-06`；max bbox delta `4.96e-05 px` |
| DeepPCB | 10,568,692 | `4d75d75a44e444c3b7912986ee4834e80f51ef6fcd535cd31d628827ac1edac1` | PASS；8 detections；max score delta `1.61e-06`；max bbox delta `6.48e-05 px` |

生成的 ONNX 与 manifest 位于忽略的 `miniprogram/dist/`，没有加入 Git，也没有写回 P0/P1/P2。

### 小程序与 fallback 检查

| Check | Result |
|---|---|
| 全部小程序 JavaScript `node --check` | PASS |
| preprocess / class-aware NMS / latency summary 单元测试 | PASS |
| 四页 WXML/WXSS/JS/JSON 完整性 | PASS |
| 可提交源码包大小 | 79,598 bytes / 2 MiB，PASS |
| Python Ruff check + format check | PASS |
| fallback `/health` 与 `/v1/models` | PASS；2 models |
| 模型下载响应与 SHA-256 | PASS；header、下载文件与 catalog 一致 |
| fallback 真实 NEU 图片推理 | PASS；3 detections；CPU；222.7 ms end-to-end |
| 原 Studio regression | PASS；10/10 tests |

### 尚需真机确认

微信开发者工具和真机测试依赖用户的小程序 AppID、已备案 HTTPS 合法域名以及实际 iOS/Android 设备，当前服务器环境无法替代这些外部条件。因此没有声称已测得手机 FPS、温度或整图 NPU placement。微信 API 不公开 thermal state 和逐算子硬件放置；UI 明确显示 `NPU requested`，Bench 只报告可测量的 model-only wall-clock latency。

## 可安装 PWA 交付（C3 Edge Lab）

由于微信小程序主体资质属于外部发布条件，增加 `smoke/c3/studio/pwa/` 作为无需微信账号、可直接通过 HTTPS 安装到主屏幕的移动端入口。PWA 复用同一批通过 real-image parity 的固定 ONNX，不重新训练，也不改动 P0/P1/P2。

| Check | Result |
|---|---|
| 核心 JavaScript/TypeScript 单元测试 | PASS；3/3（catalog/hash、channel-first decode + class-aware NMS、IoU/latency summary） |
| 产品源码 lint | PASS；0 warnings / 0 errors |
| TypeScript strict typecheck | PASS |
| Vinext/Vite production build | PASS |
| npm production dependency audit | PASS；0 vulnerabilities |
| PWA manifest / service worker | PASS；manifest HTTP 200；app shell 离线缓存；模型由 IndexedDB 独立管理 |
| 模型 artifact 校验 | PASS；NEU-DET 与 DeepPCB 文件 SHA-256 均与 catalog 固定值一致 |
| 隐私边界 | PASS；相机帧与选中图片只在浏览器内存处理，不上传服务器 |

浏览器端 runtime 按 WebGPU → WebGL → WASM 顺序尝试，并在 provider 编译失败时自动回退。最终移动设备 FPS、发热与降频结果仍需在目标手机上实测；PWA 不声称浏览器未公开的 thermal state 或逐算子硬件 placement。
