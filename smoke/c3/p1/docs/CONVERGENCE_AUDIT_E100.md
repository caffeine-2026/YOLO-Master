# C3 P1 100-Epoch 收敛审计

## 1. Six-run status

seed824 六组 100-epoch run 均为 PASS。执行策略为 `restart_all`：六个 e75 checkpoint 虽有 optimizer/epoch state，但没有可独立恢复的 scheduler state 和 random/RNG state，无法证明完整 resume；同时 cosine schedule 由最终 epoch budget 参数化。因此六组均从同一 `yolo11n.pt` restart，没有混用 resume 与 restart。原 30/50/75 结果均未覆盖。

## 2. Epoch-wise curves

逐 epoch 数据：`../results/convergence_epochwise_e100.csv`。

- `smoke/c3/p1/visualizations/convergence_e100/neu_map5095.png`
- `smoke/c3/p1/visualizations/convergence_e100/neu_map50.png`
- `smoke/c3/p1/visualizations/convergence_e100/neu_loss.png`
- `smoke/c3/p1/visualizations/convergence_e100/deeppcb_map5095.png`
- `smoke/c3/p1/visualizations/convergence_e100/deeppcb_map50.png`
- `smoke/c3/p1/visualizations/convergence_e100/deeppcb_loss.png`

## 3. Fixed convergence rule

沿用阈值 0.01：`delta = mean(epoch 96-100) - mean(epoch 91-95)`；`delta > 0.01` 为 `NOT_CONVERGED`，否则为 `CONVERGED_OR_PLATEAU`。未按方法或结果调整阈值。

## 4. Per-run result

| Dataset | Method | Epoch 91-95 mean | Epoch 96-100 mean | Delta | Best epoch | Best | Last | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.316820 | 0.315564 | -0.001256 | 85 | 0.328660 | 0.316170 | CONVERGED_OR_PLATEAU |
| NEU-DET | Frozen Backbone | 0.254322 | 0.261958 | +0.007636 | 79 | 0.269400 | 0.266420 | CONVERGED_OR_PLATEAU |
| NEU-DET | V-PEFT | 0.307594 | 0.303062 | -0.004532 | 87 | 0.317860 | 0.303940 | CONVERGED_OR_PLATEAU |
| DeepPCB | Full-SFT | 0.588734 | 0.555780 | -0.032954 | 75 | 0.634960 | 0.547630 | CONVERGED_OR_PLATEAU |
| DeepPCB | Frozen Backbone | 0.473090 | 0.482846 | +0.009756 | 63 | 0.486680 | 0.478640 | CONVERGED_OR_PLATEAU |
| DeepPCB | V-PEFT | 0.404170 | 0.391320 | -0.012850 | 70 | 0.517760 | 0.379310 | CONVERGED_OR_PLATEAU |

## 5. Epoch decision

`KEEP_100`。6/6 组达到 plateau 判据，0/6 组仍超过固定阈值。`FINAL_SINGLE_SEED_EPOCH = 100`。100 epoch 已冻结为 multi-seed 的统一最终预算。

## 6. Fairness

`Protocol fairness: PASS`。除 epoch 75→100 外，模型、预训练权重、split、100-image 预算、batch、imgsz、optimizer、lr、weight decay、scheduler、augmentation、seed、freeze 边界与 V-PEFT 配置均保持一致。

## 7. 30/50/75/100

方法排序状态为“75→100 稳定”；全历史状态为“30/50 与 75/100 不完全一致”。

- NEU-DET: e30=Full-SFT > Frozen Backbone > V-PEFT; e50=Full-SFT > Frozen Backbone > V-PEFT; e75=Full-SFT > V-PEFT > Frozen Backbone; e100=Full-SFT > V-PEFT > Frozen Backbone
- DeepPCB: e30=Full-SFT > Frozen Backbone > V-PEFT; e50=Full-SFT > Frozen Backbone > V-PEFT; e75=Full-SFT > V-PEFT > Frozen Backbone; e100=Full-SFT > V-PEFT > Frozen Backbone

因此，排序在最近两个 convergence candidate（75/100）之间已稳定，但并非四个预算从一开始就不变。这只说明当前 seed 的排序状态，不构成 multi-seed 方法结论。详见 `../results/e30_e50_e75_e100.csv`。

## 8. Multi-seed gate

`MULTISEED_READY = YES`。本轮未运行 seed825/826。
