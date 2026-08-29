# C3 Edge Lab — 微信小程序

安装即用之前的中间形态：用户扫描小程序码后，优先在手机本地运行合并后的 V-PEFT ONNX 模型；不支持微信原生推理的设备仍可在 **Photo** 页使用可选 HTTPS fallback。它不会启动训练，也不会修改任何 P0/P1/P2 结果或 checkpoint。

## 产品页面

- **Live**：`camera` 实时帧 → RGBA/letterbox → `wx.createInferenceSession` → JavaScript class-aware NMS → Canvas detection overlay。只保留一个 in-flight frame，避免积压。
- **Photo**：相册/相机多图顺序推理，端侧失败时可调用服务器 fallback，导出结构化 JSON。
- **Bench**：5 次 warmup 后运行 30/100 次 model-only 推理，报告 mean、p50、p95、min、max 与 FPS。
- **Models**：HTTPS 模型地址、fallback 地址、confidence/NMS 配置；下载、SHA-256 校验、缓存删除、session load test。

## 为什么使用 ONNX

微信小程序基础库 2.30.0 起提供通用 AI 推理接口。`wx.createInferenceSession` 只接受 ONNX 文件，可选择精度等级、量化与 iOS NPU；`CameraContext.onCameraFrame` 提供 RGBA 实时帧：

- https://developers.weixin.qq.com/miniprogram/dev/framework/open-ability/inference/tutorial.html
- https://developers.weixin.qq.com/miniprogram/dev/api/ai/inference/wx.createInferenceSession.html
- https://developers.weixin.qq.com/miniprogram/dev/api/media/camera/CameraContext.onCameraFrame.html
- https://developers.weixin.qq.com/miniprogram/dev/framework/open-ability/inference/supports.html

当前 C3 模型导出为固定 `1×3×640×640`、opset 12、无图内 NMS 的 ONNX。NMS 在小程序 JavaScript 中执行，因为微信算子表中 NMS 仅支持 CPU。导出的 YOLO graph 包含 `Shape` / `Transpose`，其 iOS NPU 放置未由微信公开算子表确认；因此 UI 只显示 **NPU eligible/requested**，绝不声称整图已完全运行在 NPU。

## 1. 导出模型

依赖已经隔离在项目 `.venv` 中；如需在新环境重建：

```bash
.venv/bin/python -m pip install "onnx>=1.12" "onnxslim>=0.1.82" "onnxruntime<1.20.0"
```

从 P1 CSV 自动选择 seed 824、100-epoch PASS checkpoint，校验原 SHA-256，在临时目录 merge LoRA，随后做 ONNX checker 与真实图片 PyTorch/ONNX prediction parity：

```bash
CUDA_VISIBLE_DEVICES="" \
  .venv/bin/python smoke/c3/studio/miniprogram/tools/export_models.py --dataset all
```

产物位于：

```text
smoke/c3/studio/miniprogram/dist/
├── model-manifest.json
└── models/
    ├── neu_vpeft_640.onnx
    └── deeppcb_vpeft_640.onnx
```

ONNX 与生成 manifest 被 `.gitignore` 排除，不会误提交大模型。不要把导出模型放进小程序代码包；单个主包/分包有体积限制，模型应放在 HTTPS CDN/对象存储。

## 2. 配置微信开发者工具

1. 安装微信开发者工具，导入 `smoke/c3/studio/miniprogram/`。
2. `project.config.json` 已配置 C3 小程序 AppID；个人私有设置放在不提交的 `project.private.config.json`，不要在仓库保存 AppSecret。
3. 在微信公众平台配置 `downloadFile` 与 `request` 合法 HTTPS 域名。
4. 上传 `dist/models/*.onnx` 到对象存储/CDN。
5. 打开小程序 **Models**，将 Model base URL 设为模型文件所在目录，例如 `https://cdn.example.com/c3-models`。
6. 在真机选择 **Download** → **Load test**。开发者工具可以检查 UI/JS，但最终 ONNX 编译、NPU eligibility、相机 FPS 必须真机验证。

导出后把 `dist/model-manifest.json` 中的 SHA-256 同步到 `models/catalog.js`，小程序会在每次首次下载与缓存复用时校验。

## 3. 开发 fallback

fallback 服务默认只监听 loopback，不应直接暴露公网：

```bash
CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python smoke/c3/studio/miniprogram/server.py --host 127.0.0.1 --port 7863
```

Endpoints：

- `GET /health`
- `GET /v1/models`
- `GET /v1/models/{filename}`
- `POST /v1/infer`：multipart `file`、`dataset`、`method`、`confidence`、`iou`

生产环境需要 HTTPS、微信合法域名、身份/限流与对象存储。可选环境变量 `C3_EDGE_API_KEY` 只用于服务端开发保护；不要把长期密钥硬编码进小程序。

## 4. 测试

```bash
node smoke/c3/studio/miniprogram/tests/test-core.js
.venv/bin/python smoke/c3/studio/miniprogram/tools/validate_project.py
.venv/bin/python -m py_compile \
  smoke/c3/studio/miniprogram/server.py \
  smoke/c3/studio/miniprogram/tools/export_models.py \
  smoke/c3/studio/miniprogram/tools/validate_project.py
```

微信开发者工具中还需执行：相机授权、NEU/DeepPCB 模型下载、Load test、Live 30 秒、Photo 端侧与 fallback、30/100 次 Bench、JSON 导出。

## 已知边界

- 当前模型只识别 NEU-DET / DeepPCB 工业缺陷，不识别人、杯子等通用物体。
- 工业小缺陷在远距离实时预览中可能过小；高质量展示应结合近距离/微距、ROI 与 Photo 高分辨率复核。
- 微信不向小程序公开 iOS `ProcessInfo.thermalState`，也不公开每个 ONNX 算子的 CPU/NPU 放置；Bench 只报告可诚实测量的 wall-clock model latency。
- iOS 可请求 NPU，Android 的实际硬件后端与算子覆盖取决于微信版本和设备。
- 微信原生推理不可用时，Live 不做高带宽服务器视频回传；Photo 可以使用 fallback。这是为了避免伪实时与不可控流量。
- 公测发布仍需要小程序 AppID、合法 HTTPS 域名、隐私说明与微信审核。
- 仓库是 AGPLv3；公开发布前应确认代码与模型的分发合规方案。
