# C3 Industrial PEFT Studio v0.1

一个只读展示层：把已完成的 Tencent YOLO-Master C3 工业缺陷检测实验做成六页 Gradio WebUI。Studio 不启动训练，也不改写 P0/P1/P2 的 CSV、JSON、checkpoint 或 adapter。

## 1. 功能说明

- 展示 P0/P1/P2 与 72/72 P2 matrix 状态。
- 对比 NEU-DET / DeepPCB 在 10、50、100、500-shot 下的 Full-SFT、Frozen Backbone、V-PEFT。
- 绘制多种子 mean 与 95% CI scaling 曲线。
- 使用已完成的 100-epoch final checkpoint 做真实单图与浏览器相机实时 YOLO inference。
- 从既有 `metrics.json` 展示 V-PEFT Planner 决策。
- 索引报告、protocol、split manifest、summary CSV、validation JSON 与最终图表。

## 2. 页面说明

1. **Overview**：阶段状态、核心参数压缩与精度保持结论；明确 V-PEFT 不是 universal winner。
2. **3-Way Comparison**：按 dataset / sample size 查看准确率、可训练参数、显存与训练时间。
3. **Few-shot Scaling**：mAP、accuracy retention、GPU memory、GPU-hours 的 mean ± 95% CI。
4. **Live Inference**：支持上传单图或浏览器实时相机；显示检测框、stream FPS、端到端延迟，以及 preprocess / model / postprocess-NMS 分阶段延迟。
5. **V-PEFT Planner**：自动读取 planner status、backend、targets 与参数统计。
6. **Evidence / Reproduction**：只显示仓库相对路径、Git provenance 和非训练复现命令。

## 3. 安装依赖

优先使用项目现有 `.venv`。项目环境原有 `pandas`、`matplotlib`、`Pillow`、`torch` 与本仓库 `ultralytics`；Studio 新增的最小入口依赖为：

```bash
.venv/bin/python -m pip install "gradio==6.26.0"
```

不要为 Studio 升级 Torch、CUDA 或 Ultralytics。

## 4. 启动命令

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python smoke/c3/studio/app.py
```

应用仅监听 `127.0.0.1`，依次尝试 7860、7861、7862。也可显式指定允许范围内的端口：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python smoke/c3/studio/app.py --port 7861
```

## 5. SSH tunnel

假设服务最终使用 7860：

```bash
ssh -L 7860:127.0.0.1:7860 pll@10.103.69.211
```

本地访问 `http://127.0.0.1:7860`。如果服务器使用 7861/7862，请同步替换 tunnel 两侧端口。

## 6. GPU / CPU 说明

- Overview、Comparison、Scaling、Planner、Evidence 全部只读 CSV/JSON 并在 CPU 上绘图。
- Torch 与 Ultralytics 仅在执行单图或相机推理后延迟导入。
- 进程强制 `CUDA_VISIBLE_DEVICES=0`，不会访问其他 GPU。
- 如果 GPU 0 存在其他 compute process、CUDA 不可用或设置 `C3_STUDIO_FORCE_CPU=1`，推理降级 CPU，并在 UI 显示实际 device。
- Studio 同时只缓存一个模型，切换模型时释放旧引用与 CUDA cache。

## 7. 数据来源

P1：

- `smoke/c3/p1/results/p1_all_runs.csv`
- `smoke/c3/p1/results/p1_summary.csv`
- `smoke/c3/p1/results/tradeoff_multiseed.csv`

P2：

- `smoke/c3/p2/results/p2_all_runs.csv`
- `smoke/c3/p2/results/p2_summary.csv`
- `smoke/c3/p2/results/retention_multiseed.csv`
- `smoke/c3/p2/results/paired_analysis.csv`

Planner：canonical V-PEFT P1 run 的 `smoke/c3/p1/logs/<run_id>/metrics.json`，`run_id` 由 P1 CSV 自动解析。

## 8. 模型来源

Live Inference 根据 `p1_all_runs.csv` 自动定位每个 dataset / method 的 seed 824、100-epoch PASS run，并加载：

`smoke/c3/p1/artifacts/<run_id>/weights/best.pt`

加载前校验 checkpoint SHA-256 与 CSV 一致。V-PEFT `best.pt` 包含已激活的 PEFT model；Studio 沿用仓库现有 `smoke/c3/p0/tools/visualize_predictions.py` 的加载与 adapter tensor/target 校验方式，不重新训练，不猜测 merge 流程。独立 `lora_adapter/` 保留为证据与后续复用产物。

## 9. 已知限制

- Dashboard 展示三种子统计；Live Inference 为稳定演示固定使用 CSV 中 seed 824 的最终 checkpoint。
- 首次切换模型需要从磁盘加载 checkpoint，延迟高于后续同模型推理。
- Realtime Camera 把浏览器帧发送到服务器 GPU/CPU 推理；它不是 iPhone Core ML / Android NCNN 的端侧 benchmark，FPS 还包含浏览器采集、传输与 Gradio 调度开销。
- 浏览器相机需要安全上下文；本机 `http://127.0.0.1` 可直接使用，手机从局域网访问时需 HTTPS 或设备本地安全隧道。
- UI 不接受任意服务器路径，不提供 shell 输入，不显示 token、密码或私有环境变量。
- `P0_REF` / `P1_REF` / `P2_REF` 仅在既有证据明确记录时显示；当前未记录的字段显示 `Not recorded`。

## 10. Screenshots

- `smoke/c3/studio/screenshots/overview.png`
- `smoke/c3/studio/screenshots/comparison.png`
- `smoke/c3/studio/screenshots/scaling.png`
- `smoke/c3/studio/screenshots/inference.png`
- `smoke/c3/studio/screenshots/planner.png`

测试结果见 `smoke/c3/studio/test_report.md`。

## 11. 微信小程序端侧产品

`smoke/c3/studio/miniprogram/` 提供独立的 **C3 Edge Lab**：

- 微信相机实时帧与端侧 ONNX inference
- Live / Photo / Bench / Models 四页产品
- V-PEFT checkpoint 临时 merge、固定 ONNX export、SHA-256 与 PyTorch/ONNX parity
- 可选的 loopback photo fallback API

完整导出、微信开发者工具、模型 CDN 与真机测试说明见 `smoke/c3/studio/miniprogram/README.md`。模型产物位于 `miniprogram/dist/` 并被 Git 忽略；不会提交 checkpoint 或 ONNX 大文件。

## 12. 可安装 PWA 端侧产品（推荐入口）

`smoke/c3/studio/pwa/` 提供不依赖微信主体资质的 **C3 Edge Lab PWA**。用户通过 HTTPS 链接打开后，可在 iOS Safari 或 Android Chrome 中添加到主屏幕并以独立 App 方式运行：

- Live：手机相机实时推理、检测框、FPS 与 preprocess/model/NMS/total 分段耗时
- Photo：最多 9 张本地照片顺序推理、结果浏览与 JSON 报告导出
- Bench：5 次 warmup 后执行 30/100 次 model-only benchmark，报告 mean/p50/p95/min/max/FPS
- Models：NEU-DET 与 DeepPCB 模型下载、SHA-256 校验、IndexedDB 离线缓存、后端加载测试

图片不会上传服务器。运行时依次尝试 WebGPU、WebGL、WASM；不支持的后端会自动回退。模型沿用小程序导出的相同静态 ONNX，源码不包含 checkpoint 或 ONNX 大文件。开发、构建和部署说明见 `smoke/c3/studio/pwa/README.md`。
