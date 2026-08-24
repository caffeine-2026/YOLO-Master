# C3｜V-PEFT 工业缺陷小样本验收

## 1. 本次验收结论

本次验收范围固定为：在 **NEU-DET 5-shot** 数据上，使用 **RTX 4090 GPU 0、FP32、1 epoch**，
完成 V-PEFT、Full-SFT、冻结主干三种方案的同条件最小训练，并核验 V-PEFT 规划、注入和适配器导出。

| 验收项 | 本次验证内容 | 状态 |
| --- | --- | --- |
| 环境安装 | 独立虚拟环境、CUDA 版 PyTorch、依赖完整性 | `PASS` |
| GPU 环境 | NVIDIA 设备、设备枚举与 CUDA 张量分配 | `PASS` |
| 基线/最小任务 | V-PEFT、Full-SFT、冻结主干，同条件 GPU FP32 1 epoch | `PASS` |
| V-PEFT 注入 | strict 规划、PEFT 后端、目标数量、独立适配器导出 | `PASS` |
| 复现与配置 | 实际命令、最终配置文件、数据和权重哈希均已锁定 | `PASS` |
| 完整日志 | 三次正式运行均有日志、指标、资源、耗时、退出码和产物哈希 | `PASS` |
| 结果证据 | GPU 汇总、环境记录、数据清单、静态验证结果齐全 | `PASS` |
| 设计说明与风险 | 代码复用、验收边界、风险与降级规则明确 | `PASS` |

本次正式验收只使用稳定完成且未触发数值恢复的 **GPU FP32** 结果。历史 AMP 和 CPU 运行仅作为
补充排障材料，不参与本次合格判定；MVTec AD 属于下一阶段双数据集 P0，不属于本次验收范围。

## 2. 锁定环境与数据

基线提交：`d29ff5a70077018a25ea7dd432b30bb12f753418`

| 项目 | 锁定值 |
| --- | --- |
| OS / Python | Linux 6.8.0-136-generic x86_64 / Python 3.10.12 |
| YOLO-Master | 8.4.101，editable install 指向当前仓库 |
| PyTorch / torchvision | 2.13.0+cu130 / 0.28.0+cu130 |
| PEFT / OpenCV | 0.19.1 / 5.0.0 |
| GPU | 8 × NVIDIA GeForce RTX 4090；正式运行使用 GPU 0 |
| Driver / CUDA | 595.84；PyTorch CUDA build 13.0 |
| 数据 | [NEU-DET 官方页面](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)，6 类 |
| 原始归档 SHA-256 | `cf04bb0b05d364f23e05031b191b20b908a0e5eb11466df74ee7fba98ec835d5` |
| 5-shot 划分 | train/val/test = 30/180/180；train 62 个目标；seed 824 |
| 数据清单 SHA-256 | `6cdf14f723f1d3f9f05696a6834d27d832d1362e5a8171995d376db5071d4e34` |
| 预训练权重 | `yolo11n.pt`；SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| 公共训练条件 | `epochs=1`、`batch=1`、`imgsz=320`、`seed=824`、`workers=0`、`amp=false` |

## 3. 设计说明与代码边界

本次训练直接复用仓库已有的 `yolo train`、Trainer、LoRA 和 V-PEFT 实现；没有修改
`ultralytics/`、`tests/` 或原有 `scripts/`。新增内容仅位于 `smoke/c3/`，用于数据转换、运行封装、
证据采集和交付验证，不改变模型与训练算法。

| 方案 | 设置 |
| --- | --- |
| V-PEFT | rank 8、alpha 16、预算 2,100,000、AO 求解器、strict 模式 |
| Full-SFT | `lora_r=0`、`freeze=0` |
| 冻结主干 | `lora_r=0`、`freeze=11`，冻结模块 0～10 |

V-PEFT 预检发现首层 `0.conv` 输入通道为 3，不能承载 rank 8，因此配置显式排除该层，未关闭
strict，也未发生静默回退。最终规划状态为 `ACCEPT`，规划 59 个目标，PEFT 后端实际应用 52 个目标。

## 4. 复现命令

### 4.1 环境安装与数据准备

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
```

### 4.2 GPU FP32 三方最小任务

```bash
python smoke/c3/tools/run_smoke.py --config smoke/c3/config/vpeft_smoke.yaml \
  --name neu_det_vpeft_gpu_fp32_seed824 --device 0 --amp false
python smoke/c3/tools/run_smoke.py --config smoke/c3/config/baselines/full_sft_smoke.yaml \
  --name neu_det_full_sft_gpu_fp32_seed824 --device 0 --amp false
python smoke/c3/tools/run_smoke.py --config smoke/c3/config/baselines/frozen_backbone_smoke.yaml \
  --name neu_det_frozen_backbone_gpu_fp32_seed824 --device 0 --amp false

python smoke/c3/tools/summarize_runs.py
python smoke/c3/tools/validate_delivery.py \
  --output smoke/c3/evidence/static_validation.json
```

运行器拒绝覆盖同名目录；复跑必须使用新的 `--name`。实际参数以各运行目录中的
`resolved_config.yaml` 为准。

## 5. 结果证据

| 方案 | 可训练参数/总参数 | 速度 | 总耗时 | 观测峰值设备显存 | mAP50-95 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| V-PEFT | 613,602/2,772,770 | 9.4 it/s | 18.315 s | 712 MiB | 0.00501 | `PASS` |
| Full-SFT | 2,590,994/2,591,010 | 11.6 it/s | 13.215 s | 726 MiB | 0.00267 | `PASS` |
| 冻结主干 | 1,225,522/2,591,010 | 12.4 it/s | 13.847 s | 630 MiB | 0.02806 | `PASS` |

三次正式运行均一次完成、退出码为 0，且没有自动数值恢复。V-PEFT 使用实际 `peft` 后端，
独立适配器参数量为 181,760，导出成功。上述单 epoch、单 seed 指标只证明最小闭环跑通，
不用于评价收敛、稳定收益或方案优劣。

| 检查 | 结果 |
| --- | --- |
| GPU 设备枚举与 CUDA 张量分配 | `PASS` |
| V-PEFT 定向测试 | `39 passed` |
| C3 配置、数据、日志、证据和发布路径 | `PASS` |
| 论文锚点检查 / `yolo checks` | 退出码均为 0 |

## 6. 完整日志与证据索引

| 内容 | 仓库相对路径 |
| --- | --- |
| 完整复现指南 | `smoke/c3/PEFT_RUN_GUIDE.md` |
| 配置文件 | `smoke/c3/config/` |
| GPU 结构化结果 | `smoke/c3/evidence/gpu_smoke_comparison.json` |
| 环境与 CUDA 证据 | `smoke/c3/evidence/environment.json` |
| 静态验收结果 | `smoke/c3/evidence/static_validation.json` |
| NEU-DET 数据清单 | `smoke/c3/evidence/neu_det_fewshot_manifest.json` |
| V-PEFT 正式日志 | `smoke/c3/logs/neu_det_vpeft_gpu_fp32_seed824/` |
| Full-SFT 正式日志 | `smoke/c3/logs/neu_det_full_sft_gpu_fp32_seed824/` |
| 冻结主干正式日志 | `smoke/c3/logs/neu_det_frozen_backbone_gpu_fp32_seed824/` |

每个正式运行目录均包含实际命令、完整训练日志、最终配置、指标、资源采样、耗时、退出码和产物
SHA-256。所有公开文本只使用仓库相对路径，不包含本机用户绝对路径。

## 7. 风险与降级

| 风险 | 本次状态 | 降级与恢复规则 |
| --- | --- | --- |
| GPU 不可用 | 已解除 | 设备文件、`nvidia-smi`、CUDA PyTorch、设备枚举和张量分配必须全部通过，否则停止 GPU 验收 |
| AMP 非有限值 | 不在本次范围 | 本次固定 `amp=false`；历史 AMP 结果不得混入正式 FP32 结果 |
| GPU OOM | 未发生 | 保留失败记录；统一降低三种方案的 `imgsz` 后使用新 ID 全部重跑 |
| V-PEFT 回退 | 未发生 | strict 失败、后端非 `peft` 或实际目标数为 0时立即判定失败 |
| rank 越界 | 已解除 | 显式排除非法首层，并在最终配置中保留排除项，不关闭 strict 掩盖错误 |
| 数据来源 | 已确认 | 本次只使用官方页面公开下载且 SHA-256 已锁定的 NEU-DET，并按官方要求保留引用 |
| 结果过度解读 | 已控制 | 最小任务不发布 SOTA、收敛或稳定提升结论 |
| 日志或版本漂移 | 已控制 | 不覆盖运行；提交、依赖、数据、权重、配置和命令全部留证 |

## 8. 最终结论

本次规定范围内的环境安装、基线/最小任务、复现命令、配置文件、完整日志、结果证据、设计说明
和风险与降级均已完成，正式 GPU FP32 三方最小任务全部为 `PASS`。验收使用原有
YOLO-Master/V-PEFT 核心代码，新增工具只负责复现与证据交付。

## 9. 下一阶段 P0（不纳入本次验收）

下一目标是在 **NEU-DET** 与 **MVTec AD 派生检测任务** 上各跑通一次 V-PEFT GPU 最小闭环。
MVTec 必须先由有权提供信息的人通过[官方页面](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
取得数据并确认 CC BY-NC-SA 4.0 非商业许可；不使用未核验镜像。随后两个数据集统一使用 GPU 0、
`amp=false`、`epochs=1`、`batch=1`、`imgsz=320`、`seed=824`，并分别保存数据清单、完整日志、
最终配置、指标、资源、退出码、checkpoint 与独立适配器 SHA-256。两次运行均退出 0、strict 规划为
`ACCEPT/ADAPT`、实际后端为 `peft` 且无静默回退后，双数据集 P0 才标记为 `PASS`。
