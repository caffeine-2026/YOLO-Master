# C3｜V-PEFT 工业缺陷小样本统一验收

## 1. 验收结论

Official C3 P0 = **NEU-DET + DeepPCB 各完成一次 V-PEFT**。两次均使用 RTX 4090 GPU 0、FP32、1 epoch，并满足 strict planner、实际 PEFT backend、checkpoint、独立 adapter 与完整证据要求。

| Dataset | Planner | Actual backend | Planned / applied | Exit | Adapter | Status |
| --- | --- | --- | ---: | ---: | --- | --- |
| NEU-DET | `ACCEPT` | `peft` | 59 / 52 | 0 | yes | `PASS` |
| DeepPCB | `ACCEPT` | `peft` | 59 / 52 | 0 | yes | `PASS` |

## 2. 环境安装

独立 `.venv` 使用 Python 3.10.12、PyTorch 2.13.0+cu130、PEFT 0.19.1；CUDA 枚举和张量分配已通过，正式运行设备为 NVIDIA GeForce RTX 4090 GPU 0。环境证据见 `smoke/c3/evidence/environment.json`。

## 3. 基线/最小任务

- `BASE_REF`: `acce839c7e895d6b179de7f7093fa879e237cc7b`
- `START_REF`: `bad9294f4217773dcf0ba0666c64de121a324fda`
- 公共条件：epochs 1、batch 1、imgsz 320、workers 0、seed 824、`amp=false`
- V-PEFT：rank 8、alpha 16、AO solver、planner backend `vpeft`、strict `true`

## 4. 复现命令

```bash
python smoke/c3/tools/run_smoke.py \
  --config smoke/c3/config/vpeft_smoke.yaml \
  --data smoke/c3/config/datasets/deeppcb_fewshot.yaml \
  --name deeppcb_vpeft_gpu_fp32_seed824 \
  --device 0 --amp false

python smoke/c3/tools/summarize_runs.py
python smoke/c3/tools/validate_delivery.py \
  --output smoke/c3/evidence/static_validation.json
```

NEU-DET 已 PASS，未在本轮重跑；其原运行命令保存在对应日志目录的 `command.txt`。

## 5. 配置文件

- 公共训练配置：`smoke/c3/config/vpeft_smoke.yaml`
- 数据配置：`smoke/c3/config/datasets/neu_det_fewshot.yaml`、`smoke/c3/config/datasets/deeppcb_fewshot.yaml`
- 每次运行的最终参数：对应日志目录的 `resolved_config.yaml`

## 6. 完整日志

- NEU-DET：`smoke/c3/logs/neu_det_vpeft_gpu_fp32_seed824/`
- DeepPCB：`smoke/c3/logs/deeppcb_vpeft_gpu_fp32_seed824/`

每个目录均含命令、stdout/stderr 完整日志、resolved config、metrics、资源采样、耗时、退出码、runtime metadata 和产物索引。

## 7. 结果证据

结构化结论见 `smoke/c3/evidence/c3_p0_summary.json`，最终报告见 `smoke/c3/docs/C3_P0_FINAL_REPORT.md`，静态验收见 `smoke/c3/evidence/static_validation.json`。DeepPCB 原始数据为 1500 张 tested image、1500 份 annotation、6 类；官方 trainval/test = 1000/500，source commit 为 `08e98c4db5922613fb97176eb3d6497d48260cb1`。

## 8. 设计说明

训练复用原有 `run_smoke.py`、`vpeft_smoke.yaml` 和 YOLO-Master/V-PEFT 核心实现；本轮没有修改 `ultralytics/`、`tests/` 或 `scripts/`。新增工具只负责 DeepPCB 只读转换、数据检查、证据汇总和验收。原始数据与权重不提交。

NEU Full-SFT 与冻结主干保留为 preliminary smoke evidence，不属于 P1 结论。单 epoch、单 seed 只证明闭环可运行，不评价收敛或方法优劣。MVTec AD 仅为未来可选扩展，不属于 Official C3 P0。

## 9. 风险与降级

| 风险 | 本次状态 | 处理规则 |
| --- | --- | --- |
| 数据来源或许可 | 已核验 | 只使用原作者仓库锁定提交；遵守 README 的 research-use 说明 |
| 缺图、缺标或非法 bbox | 已通过 | 数据 smoke 不 PASS 则禁止训练 |
| GPU/FP32 条件漂移 | 未发生 | 设备、AMP 或固定参数不符即 FAIL |
| planner/PEFT 静默回退 | 未发生 | strict 非 ACCEPT/合法 ADAPT、后端非 peft、目标数为 0 即 FAIL |
| 非有限值或自动恢复 | 未发生 | 保留失败运行，正式结果不得混入恢复任务 |
| OOM 或运行失败 | 未发生 | 保留证据并用新 run name 重试，不覆盖历史目录 |
| 指标过度解读 | 已控制 | 不以 1 epoch、单 seed 结果比较方法优劣 |

## 10. 最终结论

NEU-DET 与 DeepPCB 均为 `PASS`，Overall C3 P0 = `PASS`。最终提交 SHA 在提交完成后以 `git rev-parse HEAD` 记录于交付回复；提交自身无法预先包含自己的 SHA。差异可用 `git diff acce839c7e895d6b179de7f7093fa879e237cc7b..HEAD -- smoke/c3` 复核。
