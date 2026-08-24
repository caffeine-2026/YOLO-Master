# V-PEFT 复现运行指南

本文档基于 YOLO-Master 提交 `d29ff5a70077018a25ea7dd432b30bb12f753418`，所有命令均从仓库根目录执行，不依赖任何本机用户绝对路径。

## 1. 安装 GPU 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt -e .
```

本次实测最终使用 PyTorch 2.13.0+cu130、torchvision 0.28.0+cu130、PEFT 0.19.1、OpenCV 5.0.0 和 YOLO-Master 8.4.101。GPU 为 RTX 4090，驱动 595.84；完整记录位于 `smoke/c3/evidence/environment.json`。CPU wheel 只用于 GPU 不可用时的辅助功能核验。

## 2. 环境与静态验证

```bash
source .venv/bin/activate
python smoke/c3/tools/capture_environment.py
python smoke/c3/tools/validate_delivery.py \
  --output smoke/c3/evidence/static_validation.json
python scripts/reproduce_yolo_peft_paper.py --check-anchor
```

验证器会检查全部 C3 YAML、NEU-DET 图像/标签数量、类别 ID、归一化检测框、清单哈希和可发布文本中的本机用户绝对路径。

## 3. 数据准备

### NEU-DET

[东北大学官方页面](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)说明该数据集包含 6 类、每类 300 张灰度图像，并提供用于检测任务的 NEU-DET 标注。当前压缩包 SHA-256 为 `cf04bb0b05d364f23e05031b191b20b908a0e5eb11466df74ee7fba98ec835d5`。

生成完整划分：

```bash
python smoke/c3/tools/prepare_neu_det.py \
  --source datasets/raw/neu_det_source/NEU-DET \
  --output datasets/neu_det_yolo \
  --seed 824 --overwrite
```

生成每类 5 张训练图像的划分：

```bash
python smoke/c3/tools/prepare_neu_det.py \
  --source datasets/raw/neu_det_source/NEU-DET \
  --output datasets/neu_det_fewshot_yolo \
  --seed 824 --train-shots-per-class 5 --overwrite
```

### MVTec AD

[MVTec 官方页面](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)说明数据包含 15 类、5,000 多张高分辨率图像、仅含正常样本的训练集、含正常/异常样本的测试集和像素级异常标注；下载区当前要求填写表单，许可证为 CC BY-NC-SA 4.0，禁止商业使用。

合法获取并解压数据后，可执行：

```bash
python smoke/c3/tools/prepare_mvtec_yolo.py \
  --source datasets/raw/mvtec_anomaly_detection \
  --output datasets/mvtec_yolo \
  --seed 824 --train-shots-per-group 1
```

该转换会从异常掩码生成单类别 `defect` 检测框并重新划分异常样本，只适用于有监督检测流程冒烟测试，不能作为官方无监督 MVTec 基准结果。

## 4. 已执行的 GPU FP32 稳定任务

三种任务均使用 GPU 0、`amp=false`、`yolo11n.pt`、NEU-DET 5-shot、1 epoch、batch 1、imgsz 320、workers 0、seed 824 和确定性模式。三次均未触发自动数值恢复。

```bash
python smoke/c3/tools/run_smoke.py \
  --config smoke/c3/config/vpeft_smoke.yaml \
  --name neu_det_vpeft_gpu_fp32_seed824 --device 0 --amp false
```

```bash
python smoke/c3/tools/run_smoke.py \
  --config smoke/c3/config/baselines/full_sft_smoke.yaml \
  --name neu_det_full_sft_gpu_fp32_seed824 --device 0 --amp false
```

```bash
python smoke/c3/tools/run_smoke.py \
  --config smoke/c3/config/baselines/frozen_backbone_smoke.yaml \
  --name neu_det_frozen_backbone_gpu_fp32_seed824 --device 0 --amp false
```

## 5. 已执行的 GPU AMP 降级任务

以下三个 `amp=true` 任务均实际执行并退出 0，但每个都触发一次非有限值恢复；训练器随后关闭 AMP 重试。因此这组只标记为 `DEGRADED`，不得作为纯 AMP 性能基准。

```bash
python smoke/c3/tools/run_smoke.py \
  --config smoke/c3/config/vpeft_smoke.yaml \
  --name neu_det_vpeft_gpu_seed824 --device 0 --amp true
```

```bash
python smoke/c3/tools/run_smoke.py \
  --config smoke/c3/config/baselines/full_sft_smoke.yaml \
  --name neu_det_full_sft_gpu_seed824 --device 0 --amp true
```

```bash
python smoke/c3/tools/run_smoke.py \
  --config smoke/c3/config/baselines/frozen_backbone_smoke.yaml \
  --name neu_det_frozen_backbone_gpu_seed824 --device 0 --amp true
```

## 6. 已执行的 CPU 辅助功能任务

三种任务均使用 `yolo11n.pt`、NEU-DET 5-shot、1 epoch、batch 1、imgsz 320、workers 0、seed 824 和确定性模式。

```bash
python smoke/c3/tools/run_smoke.py \
  --config smoke/c3/config/vpeft_smoke.yaml \
  --name neu_det_vpeft_cpu_seed824 --device cpu --amp false
```

```bash
python smoke/c3/tools/run_smoke.py \
  --config smoke/c3/config/baselines/full_sft_smoke.yaml \
  --name neu_det_full_sft_cpu_seed824 --device cpu --amp false
```

```bash
python smoke/c3/tools/run_smoke.py \
  --config smoke/c3/config/baselines/frozen_backbone_smoke.yaml \
  --name neu_det_frozen_backbone_cpu_seed824 --device cpu --amp false
```

运行器拒绝覆盖同名目录。重复实验时应使用新的 `--name`；仅需重新收集既有运行的配置、适配器和校验和时可添加 `--refresh-existing`。

## 7. GPU 入口复核

先确认以下命令全部成功：

```bash
ls -l /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import torch; torch.empty(1, device='cuda:0'); print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

本次四项检查均已通过。任何复跑都必须使用同一 GPU、相同公共配置和新的运行 ID。任一入口检查失败时不得启动 GPU 任务；OOM 或数值恢复时也不得静默修改参数，具体规则见 `smoke/c3/docs/ADMISSION_20260825.md` 第 8 节。

## 8. V-PEFT 配置要点

- `lora_planner_backend: vpeft` 和 `lora_vpeft_strict: true` 禁止规划失败后静默回退。
- `lora_few_shot_adaptive_rank: false` 保证请求的 rank 8 不被 few-shot 自适应逻辑改成 rank 32。
- `lora_exclude_modules: [0.conv]` 排除输入通道仅为 3 的首个 stem 卷积，否则 rank 8 会超过该层容量。
- 本次规划状态为 `ACCEPT`，规划 59 个目标；模型安全过滤后 PEFT 后端实际应用 52 个目标。
- 运行器在训练结束后通过模型 API 导出独立适配器，并将配置和完整 V-PEFT 元数据复制到证据目录。

## 9. 结果入口

- 面向评审的统一报告：`smoke/c3/docs/ADMISSION_20260825.md`
- GPU 结构化结果：`smoke/c3/evidence/gpu_smoke_comparison.json`
- CPU 辅助结构化结果：`smoke/c3/evidence/cpu_smoke_comparison.json`
- 完整日志规范：`smoke/c3/logs/README.md`
- 9 次运行退出码均为 0；GPU 结果记录设备级显存，CPU 结果的 GPU 显存字段保持为空。
