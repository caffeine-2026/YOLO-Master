# V-PEFT Multi-seed Efficiency Analysis

## Observed Fact

- NEU-DET: trainable parameter reduction=76.32%; mean peak-memory saving=1.16%; mean training-time change=+13.04%; mean GPU-hour change=+13.04%. Per-seed memory savings=(1.16%, 1.16%, 1.16%); per-seed time changes=(+13.00%, +13.96%, +12.18%).
- DeepPCB: trainable parameter reduction=76.32%; mean peak-memory saving=1.28%; mean training-time change=+13.37%; mean GPU-hour change=+13.37%. Per-seed memory savings=(1.15%, 1.16%, 1.53%); per-seed time changes=(+11.69%, +12.02%, +16.35%).

这些数值是当前代码、FP32、RTX 4090 与锁定 P1 protocol 下的三 seed 观测值。可训练参数大幅减少没有转化为同比的峰值显存下降。

## Supported Explanation

训练器报告的峰值显存包含参数以外的激活、梯度相关缓冲、优化器状态、验证与框架开销；因此 trainable parameter count 与 total peak memory 不是同一个量。V-PEFT 还执行 adapter/planner 相关计算，故参数减少本身不足以推出 wall-clock 加速。这里的解释只说明为什么两类指标可以不同，不声称已分解各项显存或耗时占比。

## Hypothesis

adapter 注入、额外张量操作、当前 kernel 路径或低利用率小数据训练可能贡献时间开销；激活或固定框架开销可能主导峰值显存。验证这些机制需要独立 profiler、算子级时间线和显存分解实验，本 P1 数据不能把任一机制确认为事实。
