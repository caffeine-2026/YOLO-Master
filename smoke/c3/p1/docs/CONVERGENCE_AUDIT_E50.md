# C3 P1 50-Epoch 收敛审计

## 1. Six-run status

seed824 六组 50-epoch run 均为 PASS。执行策略为 `restart_all`：30-epoch cosine schedule 的前 30 步不等价于 50-epoch schedule，因此没有混用 resume 与 restart。原 30-epoch pilot 未覆盖。

## 2. Epoch-wise curves

逐 epoch 数据：`../results/convergence_epochwise_e50.csv`。

- `smoke/c3/p1/visualizations/convergence_e50/neu_map5095.png`
- `smoke/c3/p1/visualizations/convergence_e50/neu_map50.png`
- `smoke/c3/p1/visualizations/convergence_e50/neu_loss.png`
- `smoke/c3/p1/visualizations/convergence_e50/deeppcb_map5095.png`
- `smoke/c3/p1/visualizations/convergence_e50/deeppcb_map50.png`
- `smoke/c3/p1/visualizations/convergence_e50/deeppcb_loss.png`

## 3. Fixed convergence rule

沿用阈值 0.01：`delta = mean(epoch 46-50) - mean(epoch 41-45)`；`delta > 0.01` 为 `NOT_CONVERGED`，否则为 `CONVERGED_OR_PLATEAU`。未按方法或结果调整阈值。

## 4. Per-run result

| Dataset | Method | Epoch 41-45 mean | Epoch 46-50 mean | Delta | Best epoch | Best | Last | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.263518 | 0.286780 | +0.023262 | 50 | 0.292300 | 0.292300 | NOT_CONVERGED |
| NEU-DET | Frozen Backbone | 0.239738 | 0.255286 | +0.015548 | 50 | 0.259790 | 0.259790 | NOT_CONVERGED |
| NEU-DET | V-PEFT | 0.253876 | 0.258444 | +0.004568 | 48 | 0.260200 | 0.257910 | CONVERGED_OR_PLATEAU |
| DeepPCB | Full-SFT | 0.481210 | 0.483938 | +0.002728 | 45 | 0.507880 | 0.470700 | CONVERGED_OR_PLATEAU |
| DeepPCB | Frozen Backbone | 0.390756 | 0.404088 | +0.013332 | 47 | 0.405300 | 0.404210 | NOT_CONVERGED |
| DeepPCB | V-PEFT | 0.296432 | 0.313152 | +0.016720 | 39 | 0.354620 | 0.309640 | NOT_CONVERGED |

## 5. Epoch decision

`EXTEND_ALL_TO_75`。共有 4/6 组仍超过固定阈值，不能将 50 epoch 称为 final P1 结果；下一阶段仍须六种条件统一预算。

## 6. Fairness

`Protocol fairness: PASS`。除 epoch 30→50 外，模型、预训练权重、split、100-image 预算、batch、imgsz、optimizer、lr、weight decay、scheduler、augmentation、seed、freeze 边界与 V-PEFT 配置均保持一致。

## 7. 30 vs 50

两个数据集的单 seed 方法顺序在锁定 test mAP50-95 上均保持 Full-SFT > Frozen Backbone > V-PEFT；这只说明当前 seed 的排序未反转，不构成多 seed 方法结论。详见 `../results/e30_vs_e50.csv`。

## 8. Multi-seed gate

`MULTISEED_READY = NO`。先完成统一 75 epoch 候选并重新审计；未运行 seed825/826。
