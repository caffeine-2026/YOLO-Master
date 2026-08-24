# C3｜V-PEFT 工业缺陷小样本

## 1. 课题定位与验收边界

C3 使用仓库已有的 V-PEFT/LoRA 与 `yolo train`，在 NEU-DET 工业缺陷小样本上完成
GPU 最小训练，并与全参数微调、冻结主干做同条件对照。本次没有修改 YOLO-Master 模型、
训练器或 V-PEFT 核心实现；新增内容仅为 `smoke/c3/` 下的数据转换、复现、日志和验收工具。

| 范围 | 本次验证内容 | 状态 |
| --- | --- | --- |
| GPU 环境 | NVIDIA 设备、CUDA 版 PyTorch、设备枚举与 CUDA 张量分配 | `PASS` |
| GPU FP32 最小任务 | V-PEFT、Full-SFT、冻结主干，同条件 1 epoch | `PASS` |
| GPU AMP 运行 | 三种方案均触发一次非有限值恢复，关闭 AMP 后完成 | `DEGRADED` |
| V-PEFT 注入 | strict 规划、PEFT 后端、目标数量与适配器导出 | `PASS` |
| MVTec AD | 官方下载需要填写表单，尚未取得合法数据 | `BLOCKED` |
| 精度结论 | 仅 1 epoch、单 seed，不评价收敛或方案优劣 | `N/A` |

CPU 运行只保留为历史功能链路证据，不替代 GPU，也不用于推断显存或 GPU 吞吐。

## 2. 8.25 交付状态

| 检查项 | 实现与证据 | 结果 |
| --- | --- | --- |
| 环境安装 | 独立 `.venv`、CUDA wheel、GPU 检查、`yolo checks`、论文锚点检查 | `PASS` |
| 基线/最小任务 | GPU FP32 三方同条件运行；另保留 AMP 降级运行 | `PASS` |
| 复现命令 | 数据、预检、三方 GPU 训练和验收命令 | `PASS` |
| 配置文件 | V-PEFT、Full-SFT、冻结主干及数据配置 | `PASS` |
| 完整日志 | 9 次运行均有命令、日志、配置、指标、资源、耗时、退出码和产物哈希 | `PASS` |
| 结果证据 | GPU/CPU 汇总、环境记录、数据清单、checkpoint SHA-256 | `PASS` |
| 设计说明 | 方法数据流、统一条件、结论边界与风险降级 | `PASS` |

## 3. 锁定环境、数据与配置

### 3.1 环境矩阵

基线提交：`d29ff5a70077018a25ea7dd432b30bb12f753418`

| 项目 | 实测值 |
| --- | --- |
| OS / Python | Linux 6.8.0-136-generic x86_64 / Python 3.10.12 |
| YOLO-Master | 8.4.101，editable install 指向当前仓库 |
| PyTorch / torchvision | 2.13.0+cu130 / 0.28.0+cu130 |
| PEFT / OpenCV | 0.19.1 / 5.0.0 |
| GPU | 8 × NVIDIA GeForce RTX 4090；本次使用 GPU 0 |
| Driver / CUDA | 595.84；驱动兼容 CUDA 13.2；PyTorch CUDA build 13.0 |
| GPU 可用性 | 字符设备、`nvidia-smi`、设备枚举和 CUDA 张量分配全部通过 |

### 3.2 数据与公共条件

| 对象 | 锁定值 |
| --- | --- |
| 数据 | [NEU-DET](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)，6 类 |
| 原始归档 SHA-256 | `cf04bb0b05d364f23e05031b191b20b908a0e5eb11466df74ee7fba98ec835d5` |
| 5-shot 划分 | train/val/test = 30/180/180；train 62 个目标；seed 824 |
| 数据清单 SHA-256 | `6cdf14f723f1d3f9f05696a6834d27d832d1362e5a8171995d376db5071d4e34` |
| 预训练权重 | `yolo11n.pt`；SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| 公共训练条件 | `epochs=1`、`batch=1`、`imgsz=320`、`seed=824`、`workers=0`、确定性模式 |

三种方案：

- V-PEFT：rank 8、alpha 16、预算 2,100,000、AO 求解器、strict 模式。
- Full-SFT：`lora_r=0`、`freeze=0`。
- 冻结主干：`lora_r=0`、`freeze=11`，冻结 `yolo11.yaml` 主干模块 0～10。

V-PEFT strict 预检发现 `0.conv` 输入通道为 3，无法承载 rank 8，因此配置显式排除该层，
没有关闭 strict 或静默回退。最终状态为 `ACCEPT`，规划 59 个目标，实际应用 52 个目标。

## 4. 代码复用与新增工具

### 4.1 直接复用的仓库代码

| 入口 | 作用 |
| --- | --- |
| `ultralytics/engine/extensions/adapters.py` | 从常规训练入口触发 LoRA/V-PEFT |
| `ultralytics/utils/lora/api.py` | 构图、编译放置方案并调用适配器后端 |
| `ultralytics/vpeft/` | 预算、约束与模块级 rank/target 规划 |
| `yolo train` / YOLO-Master Trainer | 训练、验证与 checkpoint 保存 |

### 4.2 本次新增的交付工具

| 文件 | 作用 |
| --- | --- |
| `smoke/c3/tools/prepare_neu_det.py` | 转换并锁定 NEU-DET 划分 |
| `smoke/c3/tools/prepare_mvtec_yolo.py` | 合法取得 MVTec 后生成派生检测任务 |
| `smoke/c3/tools/capture_environment.py` | 采集环境、GPU 状态与资产哈希 |
| `smoke/c3/tools/run_smoke.py` | 调用原有训练入口并采集完整证据 |
| `smoke/c3/tools/summarize_runs.py` | 汇总既有日志和 checkpoint |
| `smoke/c3/tools/validate_delivery.py` | 验证配置、数据、日志与发布路径 |

Git 审计中 `ultralytics/`、`tests/` 和原有 `scripts/` 均无修改或新增文件。

## 5. 复现命令

### 5.1 环境与数据

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt -e .

python smoke/c3/tools/prepare_neu_det.py \
  --source datasets/raw/neu_det_source/NEU-DET \
  --output datasets/neu_det_fewshot_yolo \
  --seed 824 --train-shots-per-class 5 --overwrite

python smoke/c3/tools/capture_environment.py
python scripts/reproduce_yolo_peft_paper.py --check-anchor
```

### 5.2 GPU FP32 三方运行

```bash
python smoke/c3/tools/run_smoke.py \
  --config smoke/c3/config/vpeft_smoke.yaml \
  --name neu_det_vpeft_gpu_fp32_seed824 --device 0 --amp false

python smoke/c3/tools/run_smoke.py \
  --config smoke/c3/config/baselines/full_sft_smoke.yaml \
  --name neu_det_full_sft_gpu_fp32_seed824 --device 0 --amp false

python smoke/c3/tools/run_smoke.py \
  --config smoke/c3/config/baselines/frozen_backbone_smoke.yaml \
  --name neu_det_frozen_backbone_gpu_fp32_seed824 --device 0 --amp false

python smoke/c3/tools/summarize_runs.py
python smoke/c3/tools/validate_delivery.py \
  --output smoke/c3/evidence/static_validation.json
```

完整 AMP、CPU 辅助复现命令见 `smoke/c3/PEFT_RUN_GUIDE.md`。运行器拒绝覆盖同名运行，
复跑必须使用新的 `--name`；实际参数以各运行目录的 `resolved_config.yaml` 为准。

## 6. 结果证据

### 6.1 GPU FP32 稳定运行

| 方案 | 可训练参数/总参数 | 速度 | 总耗时 | 观测峰值设备显存 | mAP50-95 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| V-PEFT | 613,602/2,772,770 | 9.4 it/s | 18.315 s | 712 MiB | 0.00501 | `PASS` |
| Full-SFT | 2,590,994/2,591,010 | 11.6 it/s | 13.215 s | 726 MiB | 0.00267 | `PASS` |
| 冻结主干 | 1,225,522/2,591,010 | 12.4 it/s | 13.847 s | 630 MiB | 0.02806 | `PASS` |

三次运行都只有一次 epoch 尝试，无自动数值恢复。V-PEFT 使用实际 `peft` 后端，独立适配器
参数量为 181,760，导出成功。显存为每秒一次的 GPU 0 设备级采样，包含空闲基线，可能漏过
不足一秒的瞬时峰值，不宣称为精确的进程独占值。

### 6.2 GPU AMP 与 CPU 辅助结果

GPU AMP 三次运行均退出 0，但各触发一次非有限值恢复并关闭 AMP 重试，因此统一标记为
`DEGRADED`；含恢复耗时分别为 41.394、22.170、21.757 秒，不作为纯 AMP 基准。

CPU 三次运行均退出 0，只证明数据、训练、验证、保存和适配器导出链路可执行；结构化结果保留在
`smoke/c3/evidence/cpu_smoke_comparison.json`。

### 6.3 验证结果

| 检查 | 结果 |
| --- | --- |
| V-PEFT 定向测试 | `39 passed` |
| C3 配置、数据、日志、证据和发布路径 | `PASS` |
| 论文锚点检查 / `yolo checks` | 退出码均为 0 |

上述 mAP 均为单次 1 epoch 最小任务观测，不代表收敛、稳定收益或方案优劣。

## 7. 证据索引

| 证据 | 仓库内路径 |
| --- | --- |
| 完整复现指南 | `smoke/c3/PEFT_RUN_GUIDE.md` |
| 配置 | `smoke/c3/config/` |
| 完整日志 | `smoke/c3/logs/` |
| GPU 结构化结果 | `smoke/c3/evidence/gpu_smoke_comparison.json` |
| CPU 辅助结果 | `smoke/c3/evidence/cpu_smoke_comparison.json` |
| 环境证据 | `smoke/c3/evidence/environment.json` |
| 静态验收结果 | `smoke/c3/evidence/static_validation.json` |
| NEU-DET 清单 | `smoke/c3/evidence/neu_det_fewshot_manifest.json` |

每个运行目录包含 `command.txt`、`train.log`、`resolved_config.yaml`、`metrics.csv`、
`resources.csv`、`time.txt`、`result.json` 和 `artifact_index.txt`；V-PEFT 目录另含适配器配置、
运行元数据和导出状态。所有公开文本均使用仓库相对路径，不包含本机用户绝对路径。

## 8. 风险与降级

| 风险 | 当前处理 | 恢复标准 |
| --- | --- | --- |
| GPU 不可用 | GPU 运行标记 `BLOCKED`；只允许 CPU 功能核验，不报告 VRAM/吞吐 | 字符设备、`nvidia-smi`、CUDA PyTorch、设备枚举和分配全部通过；当前已通过 |
| AMP 非有限值 | 保留原日志为 `DEGRADED`，使用新 ID 完成 FP32 三方重跑 | AMP 三方均无恢复后才发布纯 AMP 结果 |
| GPU OOM | 保留失败；三方共同按 `imgsz=288`、`256` 依次重跑 | 原配置通过；否则降级结果不得与规范结果混排 |
| V-PEFT 回退 | strict 失败、后端非 `peft` 或目标数为 0时立即失败 | 规划为 `ACCEPT/ADAPT`、实际注入且适配器导出成功 |
| rank 越界 | 显式排除非法模块，不关闭 strict 掩盖错误 | 预检通过并在解析配置中记录排除项；当前已排除 `0.conv` |
| 数据许可 | 不代填个人信息，不使用未核验镜像，不静默替换数据集 | 合法来源、许可、SHA-256 和标签验证齐全 |
| 最小任务过度解读 | 不发布 SOTA、收敛或稳定提升结论 | 正式实验统一预算，至少 3 个 seed 并报告均值与标准差 |
| 日志/版本漂移 | 不覆盖运行；提交、依赖、数据、权重、配置和命令全部留证 | 新版本使用新运行 ID 重跑全部三种方案 |

当前未解除的事项只有两项：MVTec 合法数据尚未取得；GPU AMP 的非有限值原因尚未修复。

## 9. 结论

环境安装、基线/最小任务、复现命令、配置文件、完整日志、结果证据和设计说明七项均已有
可核验证据。正式 GPU FP32 三方最小任务为 `PASS`；GPU AMP 为 `DEGRADED`；MVTec 为
`BLOCKED`。本次训练使用原有 YOLO-Master/V-PEFT 代码，新增工具只负责复现与证据交付。

## 10. P0 最低交付（下一步）

目标是在 **NEU-DET** 和 **MVTec AD 派生检测任务** 上分别跑通一次 V-PEFT GPU 最小闭环。

1. 合法取得并校验两个数据集，锁定来源、许可、SHA-256、划分和类别；MVTec 未取得前保持 `BLOCKED`，不使用非官方镜像替代。
2. 两个数据集统一使用 GPU 0、`amp=false`、`epochs=1`、`batch=1`、`imgsz=320`、`seed=824`；只允许修改数据路径和类别数。
3. 分别执行 V-PEFT strict 训练、验证、checkpoint 保存和独立适配器导出；规划状态必须为 `ACCEPT/ADAPT`，实际后端必须为 `peft`，应用目标数必须大于 0。
4. 每个数据集保存实际命令、最终配置、完整日志、指标、资源采样、退出码、数据清单和产物 SHA-256。
5. 两个运行均退出 0、无静默回退、证据齐全后，P0 标记为 `PASS`；该结果只证明双数据集链路跑通，不作精度收益结论。
