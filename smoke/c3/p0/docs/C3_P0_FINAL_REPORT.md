# C3 P0 最终报告

## Official C3 P0

| Dataset | Planner | Actual backend | Targets | GPU | Exit | Adapter | Status |
| --- | --- | --- | ---: | --- | ---: | --- | --- |
| NEU-DET | `ACCEPT` | `peft` | 52 | RTX 4090 | 0 | yes | `PASS` |
| DeepPCB | `ACCEPT` | `peft` | 52 | RTX 4090 | 0 | yes | `PASS` |

两次 V-PEFT 均规划 59 个目标，经模型兼容性安全过滤后实际应用 52 个；固定 GPU 0、`amp=false`、epochs 1、batch 1、imgsz 320、workers 0、seed 824。两次均无数值恢复或静默回退。

## DeepPCB 数据核验

- 来源：原作者公开仓库 `https://github.com/tangsanli5201/DeepPCB`
- source commit：`08e98c4db5922613fb97176eb3d6497d48260cb1`
- 原始数据：1500 张 tested image、1500 份 annotation、6 类；官方 trainval/test = 1000/500
- 类别：open、short、mousebite、spur、copper、pin-hole
- few-shot：seed 824；train/val/test = 8/200/500；各类训练正样本图像不少于 5 张
- 检查：图标配对、bbox 边界、空标签、类别 ID、split 重叠和真实 dataloader batch 均通过
- 使用边界：原 README 声明数据仅限 research purpose；原始数据未进入主仓库

## V-PEFT 证据

| Dataset | Trainable params | Adapter params | Peak GPU memory | Elapsed | Exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| NEU-DET | 613,602 | 181,760 | 712 MiB | 18.315 s | 0 |
| DeepPCB | 613,602 | 181,760 | 712 MiB | 17.075 s | 0 |

日志目录分别为 `smoke/c3/p0/logs/neu_det_vpeft_gpu_fp32_seed824/` 与 `smoke/c3/p0/logs/deeppcb_vpeft_gpu_fp32_seed824/`。每个目录包含完整日志、resolved config、Planner/PEFT 元数据、metrics、显存采样、耗时、退出码和 checkpoint/adapter SHA-256。

## Extra Smoke Evidence

NEU-DET 的 V-PEFT、Full-SFT 与 Frozen Backbone 三方 GPU FP32 运行保留为 **preliminary smoke evidence only / not P1 conclusion**。单 epoch、单 seed 指标不能用于评价收敛或方法优劣。

## 版本与验收

- `BASE_REF`: `acce839c7e895d6b179de7f7093fa879e237cc7b`
- `START_REF`: `bad9294f4217773dcf0ba0666c64de121a324fda`
- `FINAL_REF`: 提交完成后由 `git rev-parse HEAD` 固化在交付回复
- 差异：`git diff acce839c7e895d6b179de7f7093fa879e237cc7b..HEAD -- smoke/c3`
- 结构化汇总：`smoke/c3/p0/evidence/c3_p0_summary.json`
- 静态验收：`smoke/c3/p0/evidence/static_validation.json`

结论：NEU-DET `PASS`，DeepPCB `PASS`，Overall C3 P0 `PASS`。MVTec AD 仅为未来可选扩展。
