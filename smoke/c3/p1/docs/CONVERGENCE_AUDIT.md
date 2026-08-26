# C3 P1 30-Epoch 收敛审计

## 1. Six-run status

既有 seed824 六组 pilot 均为 PASS，本审计未重跑或覆盖任何训练结果。

## 2. Epoch-wise curves

逐 epoch 数据：`../results/convergence_epochwise.csv`。生成的六张图：

- `smoke/c3/p1/visualizations/convergence/neu_map5095.png`
- `smoke/c3/p1/visualizations/convergence/neu_map50.png`
- `smoke/c3/p1/visualizations/convergence/neu_loss.png`
- `smoke/c3/p1/visualizations/convergence/deeppcb_map5095.png`
- `smoke/c3/p1/visualizations/convergence/deeppcb_map50.png`
- `smoke/c3/p1/visualizations/convergence/deeppcb_loss.png`

## 3. Convergence rule

统一使用 mAP50-95：`delta = mean(epoch 26–30) - mean(epoch 21–25)`。`delta > 0.01` 为 `NOT_CONVERGED`，否则为 `CONVERGED_OR_PLATEAU`。阈值对六组一次性固定，不按方法调整。

Epoch 决策采用顺序扩展：全部 plateau 才 `KEEP_30`；任一 run 未收敛则先 `EXTEND_ALL_TO_50` 并在 50 epoch 再审计；只有共同 50-epoch 审计仍未收敛时才考虑 `EXTEND_ALL_TO_100`。

## 4. Per-run convergence result

| Dataset | Method | Previous 5 mean | Last 5 mean | Delta | Best epoch | Best | Last | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.198128 | 0.244008 | +0.045880 | 30 | 0.245750 | 0.245750 | NOT_CONVERGED |
| NEU-DET | Frozen Backbone | 0.196452 | 0.212562 | +0.016110 | 30 | 0.217020 | 0.217020 | NOT_CONVERGED |
| NEU-DET | V-PEFT | 0.176708 | 0.189238 | +0.012530 | 30 | 0.191230 | 0.191230 | NOT_CONVERGED |
| DeepPCB | Full-SFT | 0.336930 | 0.400296 | +0.063366 | 30 | 0.410670 | 0.410670 | NOT_CONVERGED |
| DeepPCB | Frozen Backbone | 0.274108 | 0.295814 | +0.021706 | 30 | 0.300650 | 0.300650 | NOT_CONVERGED |
| DeepPCB | V-PEFT | 0.191970 | 0.205454 | +0.013484 | 30 | 0.206590 | 0.206590 | NOT_CONVERGED |

## 5. Final epoch decision

`EXTEND_ALL_TO_50`。六组 delta 均大于 0.01，且最佳 mAP50-95 均出现在 epoch 30。所有方法必须统一延长，不能只增加 V-PEFT 预算。

## 6. Fairness audit

`Protocol fairness: PASS`。相同模型、split、100-image 预算、epoch、batch、imgsz、优化器、学习率、weight decay、增强、scheduler、seed 和 test evaluator 均已逐字段核验；允许差异仅为训练策略。

## 7. V-PEFT efficiency observations

- NEU-DET: trainable reduction=76.32%, memory saving=1.16%, training time change=+12.60%.
- DeepPCB: trainable reduction=76.32%, memory saving=1.15%, training time change=+14.83%.

详细事实、支持性解释和假设见 `VPEFT_EFFICIENCY_ANALYSIS.md`。

## 8. Whether multi-seed can start

`MULTISEED_READY = NO`。最终共同预算已决定为 50 epochs，但现有 seed824 仅完成 30 epochs；必须先以不覆盖方式让六组 seed824 达到共同 50-epoch protocol 并重新审计，之后才能生成或启动 seed825/826 计划。
