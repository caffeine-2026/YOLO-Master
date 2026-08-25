# V-PEFT 效率差距分析

本分析读取既有 30-epoch pilot，并对其 checkpoint 做固定 batch、FP32、3 次 warmup + 10 次测量的内存不落盘微型分析。微型分析只用于定位开销，不能替代真实 pilot 的端到端资源数据。

## Observed facts

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
