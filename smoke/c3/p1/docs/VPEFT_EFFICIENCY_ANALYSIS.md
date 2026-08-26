# V-PEFT 效率差距分析

本分析保留 30-epoch pilot 的低侵入 profiler 证据，并用统一从零训练的 50/75/100-epoch run 重新核验端到端现象。微型分析只用于定位开销，不能替代真实训练的端到端资源数据。

## 100-Epoch Final Single-Seed Candidate

### Observed Fact

| Dataset | Trainable parameter reduction | Peak memory saving | Training time change | GPU-hour change | Accuracy retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| NEU-DET | 76.32% | 1.16% | +13.00% | +13.00% | 99.62% |
| DeepPCB | 76.32% | 1.15% | +11.69% | +11.69% | 80.62% |

- 在相同 model、split、100-image budget、batch、imgsz、optimizer、scheduler、seed、GPU 和 100 epochs 下，V-PEFT 的可训练参数均从 Full-SFT 的 2,590,994 降至 613,602，减少 76.32%。
- trainer 同口径 peak reserved memory 仅从 2652.16/2672.64 MiB 降至 2621.44/2641.92 MiB，NEU-DET/DeepPCB 分别只节省 1.16%/1.15%。
- V-PEFT wall time 分别为 395.15s/412.14s，高于 Full-SFT 的 349.68s/369.01s；GPU-hours 同比例增加 13.00%/11.69%。
- 因此，“trainable params 大幅降低、memory 几乎不降低、time 不降低且增加”是 P1 的正式 observed result。

### Supported Explanation

- 既有同 batch 微型 profiler 显示 optimizer state 从 19.77 MiB 降至 4.68 MiB，但 peak allocated 只从 1857 MiB 降至 1762 MiB；这支持“参数与 optimizer state 不是本训练峰值显存的主要组成部分”。
- 同一 profiler 测得 V-PEFT forward 约 38.9 ms，高于 Full-SFT 的 24.3–25.2 ms；这支持未融合 adapter 分支具有额外计算开销，并支持端到端时间增加的方向。
- 上述证据只支持方向与局部开销，不足以把完整 wall-time 差值全部归因于某一个 kernel 或训练阶段。

### Hypothesis

- V-PEFT optimizer-step 的额外时间可能来自更多小 adapter tensor、parameter group 或 kernel launch；尚无逐 kernel trace，不能作为事实。
- 端到端 reserved-memory saving 小于微型 profiler 差异，可能与 allocator history、验证阶段和完整 trainer 生命周期有关；需要阶段性 CUDA memory snapshot 才能验证。
- 数据加载和验证所占 wall time 可能影响最终时间比例；当前没有逐阶段计时，不能据此完成机制归因。

## 75-Epoch Revalidation — Observed facts

| Dataset | Trainable parameter reduction | Peak memory saving | Training time change | GPU-hour change | Accuracy retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| NEU-DET | 76.32% | 1.16% | +12.24% | +12.24% | 90.48% |
| DeepPCB | 76.32% | 1.15% | +10.88% | +10.88% | 77.86% |

- 75 epoch 再次确认：V-PEFT 可训练参数减少 76.32%，但完整训练的 peak reserved memory 仅节省约 1.15%。
- 75 epoch 再次确认：V-PEFT wall time 没有下降，NEU-DET/DeepPCB 分别增加 12.24%/10.88%。
- 这些是相同 batch、imgsz、GPU、split、optimizer 与 epoch 下的端到端观测事实。75 epoch 尚未冻结为最终预算，不把 accuracy retention 作为最终 P1 结论。

## 50-Epoch Historical Check — Observed facts

| Dataset | Trainable parameter reduction | Peak memory saving | Training time change | GPU-hour change | Accuracy retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| NEU-DET | 76.32% | 1.16% | +16.09% | +16.09% | 82.02% |
| DeepPCB | 76.32% | 1.15% | +9.26% | +9.26% | 68.21% |

- 50 epoch 仍确认“可训练参数显著减少、完整训练峰值 reserved memory 仅约 1% 节省”。
- 50 epoch 仍确认 V-PEFT wall time 高于 Full-SFT，但增幅随数据集变化；不能把单一百分比外推为固定开销。
- 上述 accuracy retention 是 seed824、50-epoch candidate 的观测值；由于 4/6 曲线尚未平台化，不作为最终 P1 结论。

## 30-Epoch Historical Pilot — Observed facts

- NEU-DET: V-PEFT 可训练参数减少 76.32%，实际 pilot 峰值 reserved memory 仅减少 1.16%，训练时间变化 +12.60%。
- DeepPCB: V-PEFT 可训练参数减少 76.32%，实际 pilot 峰值 reserved memory 仅减少 1.15%，训练时间变化 +14.83%。

| Dataset | Method | Trainable reduction | Pilot memory saving | Pilot time change | Profile F/B/Step (ms) | Profile allocated/reserved (MiB) | Optimizer state (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NEU-DET | Full-SFT | 0.00% | 0.00% | +0.00% | 24.28/27.35/0.46 | 1857/2352 | 19.77 |
| NEU-DET | Frozen Backbone | 52.70% | 35.91% | -7.57% | 22.34/9.47/2.18 | 877/996 | 9.35 |
| NEU-DET | V-PEFT | 76.32% | 1.16% | +12.60% | 38.81/30.68/2.94 | 1762/2002 | 4.68 |
| DeepPCB | Full-SFT | 0.00% | 0.00% | +0.00% | 25.19/28.61/0.46 | 1857/2354 | 19.77 |
| DeepPCB | Frozen Backbone | 52.70% | 36.02% | -5.76% | 22.83/10.23/2.27 | 877/994 | 9.35 |
| DeepPCB | V-PEFT | 76.32% | 1.15% | +14.83% | 38.98/22.75/2.83 | 1762/2004 | 4.68 |

Trainer 日志中的 `GPU_mem` 来自 `torch.cuda.memory_reserved()`；上述 pilot memory saving 采用这一共同口径。微型分析另行 reset CUDA peak statistics，两个口径不能混用。

## Supported explanations

- NEU-DET: 固定 batch 微型分析中，optimizer state 从 19.77 MiB 降至 4.68 MiB，但 peak allocated 仅从 1857 MiB 降至 1762 MiB；参数/优化器状态不是总 GPU 内存的主要部分。
- DeepPCB: 固定 batch 微型分析中，optimizer state 从 19.77 MiB 降至 4.68 MiB，但 peak allocated 仅从 1857 MiB 降至 1762 MiB；参数/优化器状态不是总 GPU 内存的主要部分。

- 固定 batch 下 V-PEFT forward 时间约 38.9 ms，而 Full-SFT 为 24.3–25.2 ms；adapter 分支没有融合，确实带来额外 forward 计算。该测量支持实际训练时间增加的方向，但不证明全部端到端差值都由 adapter 计算造成。
- V-PEFT 的 FP32 adapter 参数本身约 0.69 MiB，总参数内存反而比 Full-SFT 多约 0.69 MiB；节省主要来自 gradient 和 optimizer state，而不是 base parameter storage。

## Hypotheses requiring further instrumentation

- V-PEFT optimizer-step 微型时间高于 Full-SFT，可能与更多小 adapter 张量、参数组或 kernel-launch 开销有关；当前 profiler 没有逐 kernel 归因，因此这只是待验证假设。
- 微型分析的 reserved-memory saving 大于完整 pilot 的约 1%，可能与 allocator history、验证阶段和端到端生命周期不同有关；需要 trainer 内阶段性 memory snapshot 才能归因。
- 数据加载和验证占用会影响完整 wall time；当前没有把端到端 wall time逐阶段拆开，不能将全部时间差解释为 forward/backward。
