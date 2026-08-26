# C3 P1 Final Report — Three-seed 100-Epoch Protocol

## 1. Research Question

在两个工业缺陷小样本数据集上，V-PEFT 能否在显著降低可训练参数的同时，跨 seed 保持 Full-SFT 的准确率并改善 GPU 资源效率？

## 2. Final Protocol

YOLO11n (`yolo11n.pt`)，每数据集固定 100 张训练图，100 epochs，batch=8，imgsz=640，AdamW，lr0=0.001，weight decay=0.0005，cosine scheduler，FP32，seeds=824/825/826。seed824 未重跑；seed825/826 只改变训练 seed。

## 3. Datasets and Splits

NEU-DET 与 DeepPCB 均复用 seed824 固定训练成员列表和原有 val/test split。三种方法、三个 seed 的 sample IDs 完全一致。

## 4. Three Training Strategies

- Full-SFT：全模型训练。
- Frozen Backbone：冻结 `model.0`–`model.10`，训练其余层。
- V-PEFT：rank=8、alpha=16、strict AO planner、backend=`vpeft`，运行时 actual backend=`peft`，不允许 silent fallback。

## 5. 18-run Experiment Matrix

2 datasets × 3 methods × 3 seeds = 18 runs。NEU 9/9、DeepPCB 9/9 均为 PASS；12 个新增 run 各自独占 GPU 和输出目录，seed824 六个正式结果保持冻结。

## 6. Multi-seed Accuracy Results

| Dataset | Method | mAP50-95 mean ± SD | 95% CI | mAP50 mean ± SD | 95% CI | Precision mean ± SD | Recall mean ± SD | Peak GPU MiB | Time (s) | GPU-hours |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NEU-DET | Full-SFT | 0.3329 ± 0.0039 | [0.3233, 0.3425] | 0.6378 ± 0.0020 | [0.6328, 0.6427] | 0.5957 ± 0.0459 | 0.6175 ± 0.0475 | 2652.2 | 356.2 | 0.09894 |
| NEU-DET | Frozen Backbone | 0.2935 ± 0.0069 | [0.2764, 0.3106] | 0.5769 ± 0.0040 | [0.5668, 0.5869] | 0.5232 ± 0.0287 | 0.5779 ± 0.0149 | 1699.8 | 325.1 | 0.09031 |
| NEU-DET | V-PEFT | 0.3203 ± 0.0081 | [0.3002, 0.3404] | 0.6110 ± 0.0129 | [0.5791, 0.6430] | 0.6668 ± 0.0472 | 0.5361 ± 0.0439 | 2621.4 | 402.6 | 0.11184 |
| DeepPCB | Full-SFT | 0.6486 ± 0.0129 | [0.6164, 0.6808] | 0.9226 ± 0.0026 | [0.9161, 0.9290] | 0.8996 ± 0.0191 | 0.8659 ± 0.0102 | 2665.8 | 373.8 | 0.10383 |
| DeepPCB | Frozen Backbone | 0.4844 ± 0.0149 | [0.4473, 0.5214] | 0.8060 ± 0.0171 | [0.7634, 0.8485] | 0.7842 ± 0.0296 | 0.7542 ± 0.0052 | 1706.7 | 340.1 | 0.09448 |
| DeepPCB | V-PEFT | 0.5166 ± 0.0142 | [0.4813, 0.5519] | 0.7958 ± 0.0161 | [0.7559, 0.8358] | 0.7790 ± 0.0199 | 0.7303 ± 0.0111 | 2631.7 | 423.7 | 0.11771 |

## 7. Resource Efficiency Results

资源指标为三 seed 均值。排序只描述对应维度，不合并成单一 winner：

- NEU-DET accuracy: Full-SFT > V-PEFT > Frozen Backbone
- NEU-DET parameter efficiency: V-PEFT > Frozen Backbone > Full-SFT
- NEU-DET memory efficiency: Frozen Backbone > V-PEFT > Full-SFT
- NEU-DET time efficiency: Frozen Backbone > Full-SFT > V-PEFT
- DeepPCB accuracy: Full-SFT > V-PEFT > Frozen Backbone
- DeepPCB parameter efficiency: V-PEFT > Frozen Backbone > Full-SFT
- DeepPCB memory efficiency: Frozen Backbone > V-PEFT > Full-SFT
- DeepPCB time efficiency: Frozen Backbone > Full-SFT > V-PEFT

## 8. V-PEFT Trade-off

- NEU-DET：parameter reduction=76.32%，accuracy retention=96.20%，accuracy drop=0.0126，memory saving=1.16%，training time change=+13.04%，GPU-hour change=+13.04%。
- DeepPCB：parameter reduction=76.32%，accuracy retention=79.65%，accuracy drop=0.1320，memory saving=1.28%，training time change=+13.37%，GPU-hour change=+13.37%。

V-PEFT 的 76.32% 可训练参数减少在两个数据集和三个 seed 上是结构常数。是否带来准确率、显存或时间收益必须分别判断，不能由参数量直接推断。

## 9. Paired Seed Analysis

- NEU-DET V-PEFT − Full-SFT：seed deltas=(-0.0012, -0.0161, -0.0207)，mean=-0.0126，95% CI=[-0.0378, +0.0126]，方向=0/3 positive; 3/3 negative。
- DeepPCB V-PEFT − Full-SFT：seed deltas=(-0.1230, -0.1274, -0.1456)，mean=-0.1320，95% CI=[-0.1618, -0.1022]，方向=0/3 positive; 3/3 negative。

n=3 很小，因此主要报告配对差值、spread、CI 与方向一致性，不用 p-value 支撑强结论。

## 10. Qualitative Comparison

每数据集从按文件名排序的固定 test split 中等间距选取 10 张，选择发生在推理前且与模型结果无关。四栏固定为 GT / Full-SFT / Frozen / V-PEFT，均使用预先指定的 seed824 100-epoch `best.pt`，confidence threshold=0.25，imgsz=640；未针对方法调阈值。

## 11. Planner Analysis

六个 V-PEFT runs 均满足 strict=true、planner status=ACCEPT 或合法 ADAPT、planner backend=vpeft、actual backend=peft、applied targets>0、adapter export 成功，且无 silent fallback。

## 12. Negative / Positive Findings

- Positive：V-PEFT 的可训练参数减少跨 seed 稳定为 76.32%。
- Accuracy：NEU accuracy retention=96.20%；DeepPCB=79.65%。数据集间 trade-off 明显不同。
- Resource：V-PEFT 的 memory saving 为 NEU 1.16%、DeepPCB 1.28%；time change 为 NEU +13.04%、DeepPCB +13.37%。这说明当前实现的参数减少没有自动转化为同比显存或时间减少。

对预先提出的跨 seed 观察逐项复核：

- A — NEU：V-PEFT 三个 seed 均低于对应 Full-SFT；mean drop=0.0126、retention=96.20%，配对 CI 跨 0。数据支持“均值接近但未达到 parity”，不支持“无损”。
- B — DeepPCB：三个配对差值均为负，mean drop=0.1320，配对 95% CI 完全低于 0；较大性能损失跨 seed 持续存在。
- C — 参数：76.32% reduction 是所有 V-PEFT run 的结构常数，稳定。
- D — 显存：NEU 各 seed saving 为 1.16%/1.16%/1.16%；DeepPCB 为 1.15%/1.16%/1.53%。近乎为零的 saving 稳定。
- E — 时间：NEU 各 seed change 为 +13.00%/+13.96%/+12.18%；DeepPCB 为 +11.69%/+12.02%/+16.35%。六个配对均为 overhead。

## 13. Limitations

每组只有三个 seed，t-based 95% CI 很宽且对单个 run 敏感；每数据集仅用 100 张训练图；结论限于 YOLO11n、当前冻结边界、V-PEFT rank/planner 与 RTX 4090 FP32 实现。并行运行使用相同型号独占 GPU，但 seed824 与新增 runs 的系统时段不同。qualitative comparison 使用预先指定 seed824，不代表跨 seed 集成模型。

## 14. Final P1 Conclusion

Overall C3 P1 = PASS：18/18 runs 可追踪，protocol fairness、统计汇总、准确率/资源比较、trade-off、配对分析、qualitative comparison 与 artifact integrity 均完成。数据支持 V-PEFT 具有显著且稳定的 trainable-parameter efficiency；其准确率保留具有数据集依赖性，且当前实现下不能声称显存或训练时间随参数量同比下降。
