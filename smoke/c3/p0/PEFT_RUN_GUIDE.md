# C3 P0 V-PEFT 复现指南

所有命令均从仓库根目录执行，公开材料不依赖本机用户绝对路径。公共基线为 `acce839c7e895d6b179de7f7093fa879e237cc7b`；本轮起点为 `bad9294f4217773dcf0ba0666c64de121a324fda`。

## 1. 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt -e .
```

实测环境为 Python 3.10.12、PyTorch 2.13.0+cu130、PEFT 0.19.1、RTX 4090；正式运行使用 GPU 0。

## 2. 数据准备

NEU-DET 已完成，不需要重跑。其配置和清单位于 `config/datasets/neu_det_fewshot.yaml` 与 `evidence/neu_det_fewshot_manifest.json`。

DeepPCB 使用原作者公开仓库 `https://github.com/tangsanli5201/DeepPCB`，锁定 source commit `08e98c4db5922613fb97176eb3d6497d48260cb1`：

```bash
git clone https://github.com/tangsanli5201/DeepPCB.git datasets/raw/DeepPCB

python smoke/c3/p0/tools/prepare_deeppcb.py \
  --source datasets/raw/DeepPCB \
  --output datasets/deeppcb_yolo \
  --manifest-output smoke/c3/p0/evidence/deeppcb_full_manifest.json \
  --seed 824 --overwrite

python smoke/c3/p0/tools/prepare_deeppcb.py \
  --source datasets/raw/DeepPCB \
  --output datasets/deeppcb_fewshot_yolo \
  --manifest-output smoke/c3/p0/evidence/deeppcb_manifest.json \
  --seed 824 --train-shots-per-class 5 --overwrite

python smoke/c3/p0/tools/validate_deeppcb_data.py \
  --data smoke/c3/p0/config/datasets/deeppcb_fewshot.yaml \
  --manifest smoke/c3/p0/evidence/deeppcb_manifest.json \
  --output smoke/c3/p0/evidence/deeppcb_data_validation.json \
  --imgsz 320 --batch 1
```

转换保留官方 test 500 张；官方 trainval 1000 张按 seed 824 固定划分为 train 800 / val 200。few-shot 训练集从 train 中确定性选择最少正样本，使每类至少覆盖 5 张图像；得到 train/val/test = 8/200/500。原始数据只读，类别 1～6 映射为 YOLO 类别 0～5。

## 3. Official C3 P0 运行

两次正式任务均固定为 GPU 0、`amp=false`、epochs 1、batch 1、imgsz 320、workers 0、seed 824、rank 8、alpha 16、AO solver、planner backend `vpeft`、strict `true`。

```bash
python smoke/c3/p0/tools/run_smoke.py \
  --config smoke/c3/p0/config/vpeft_smoke.yaml \
  --name neu_det_vpeft_gpu_fp32_seed824 \
  --device 0 --amp false

python smoke/c3/p0/tools/run_smoke.py \
  --config smoke/c3/p0/config/vpeft_smoke.yaml \
  --data smoke/c3/p0/config/datasets/deeppcb_fewshot.yaml \
  --name deeppcb_vpeft_gpu_fp32_seed824 \
  --device 0 --amp false
```

运行器拒绝覆盖同名目录。NEU-DET 结果已经 PASS，不应重复执行；任何失败复跑必须保留原目录并使用新名称。

## 4. 汇总与验收

```bash
python smoke/c3/p0/tools/summarize_runs.py
python smoke/c3/p0/tools/validate_delivery.py \
  --output smoke/c3/p0/evidence/static_validation.json
```

验收器检查数据与类别、真实 GPU、固定配置、strict planner、实际 PEFT 后端、目标数、有限 loss、无数值恢复、完整日志、参数量、显存、耗时、checkpoint/adapter 及产物 SHA-256。

## 5. 结果边界

Official C3 P0 只包含 NEU-DET 与 DeepPCB 的 V-PEFT 闭环。NEU-DET Full-SFT、冻结主干及历史 CPU/AMP 结果仅作 preliminary smoke / 排障证据；MVTec AD 是未来可选扩展。所有单 epoch、单 seed 指标均不得用于判断收敛或方法优劣。
