# C3 P1 75-Epoch 收敛审计

## 1. Six-run status

seed824 六组 75-epoch run 均为 PASS。执行策略为 `restart_all`：50-epoch cosine schedule 的前 50 步不等价于 75-epoch schedule，因此没有混用 resume 与 restart。原 30/50-epoch 结果均未覆盖。

## 2. Epoch-wise curves

逐 epoch 数据：`../results/convergence_epochwise_e75.csv`。

- `smoke/c3/p1/visualizations/convergence_e75/neu_map5095.png`
- `smoke/c3/p1/visualizations/convergence_e75/neu_map50.png`
- `smoke/c3/p1/visualizations/convergence_e75/neu_loss.png`
- `smoke/c3/p1/visualizations/convergence_e75/deeppcb_map5095.png`
- `smoke/c3/p1/visualizations/convergence_e75/deeppcb_map50.png`
- `smoke/c3/p1/visualizations/convergence_e75/deeppcb_loss.png`

## 3. Fixed convergence rule

沿用阈值 0.01：`delta = mean(epoch 71-75) - mean(epoch 66-70)`；`delta > 0.01` 为 `NOT_CONVERGED`，否则为 `CONVERGED_OR_PLATEAU`。未按方法或结果调整阈值。

## 4. Per-run result

| Dataset | Method | Epoch 66-70 mean | Epoch 71-75 mean | Delta | Best epoch | Best | Last | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.321914 | 0.322986 | +0.001072 | 64 | 0.326280 | 0.321680 | CONVERGED_OR_PLATEAU |
| NEU-DET | Frozen Backbone | 0.277736 | 0.280250 | +0.002514 | 62 | 0.284140 | 0.279660 | CONVERGED_OR_PLATEAU |
| NEU-DET | V-PEFT | 0.281754 | 0.271284 | -0.010470 | 66 | 0.294180 | 0.270360 | CONVERGED_OR_PLATEAU |
| DeepPCB | Full-SFT | 0.566432 | 0.585302 | +0.018870 | 58 | 0.608530 | 0.576220 | NOT_CONVERGED |
| DeepPCB | Frozen Backbone | 0.441126 | 0.462256 | +0.021130 | 74 | 0.463270 | 0.463220 | NOT_CONVERGED |
| DeepPCB | V-PEFT | 0.376926 | 0.371006 | -0.005920 | 62 | 0.460170 | 0.363560 | CONVERGED_OR_PLATEAU |

## 5. Epoch decision

`EXTEND_ALL_TO_100`。4/6 组达到 plateau 判据，2/6 组仍超过固定阈值。75 epoch 仍是 convergence candidate，不能称为 final P1 结果；下一阶段须六种条件统一扩展到 100 epoch。

## 6. Fairness

`Protocol fairness: PASS`。除 epoch 50→75 外，模型、预训练权重、split、100-image 预算、batch、imgsz、optimizer、lr、weight decay、scheduler、augmentation、seed、freeze 边界与 V-PEFT 配置均保持一致。

## 7. 30/50/75

30/50/75 三个预算下的锁定 test mAP50-95 方法排序为“未稳定”。这只说明当前 seed 的排序状态，不构成 multi-seed 方法结论。详见 `../results/e30_e50_e75.csv`。

## 8. Multi-seed gate

`MULTISEED_READY = NO`。本轮未运行 seed825/826。
