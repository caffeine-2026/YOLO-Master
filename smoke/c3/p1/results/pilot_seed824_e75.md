# C3 P1 seed824 75-Epoch Candidate

30 epoch 结果仅为历史 pilot；本表为当前 75 epoch convergence candidate。

| Dataset | Method | mAP50-95 | mAP50 | Precision | Recall | Trainable / Total | Trainable Ratio | Peak GPU Mem | Memory Saving | Time | GPU-hours | Time Ratio | Checkpoint | Adapter | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.3242 | 0.6025 | 0.5384 | 0.5825 | 2,590,994 / 2,591,010 | 100.00% | 2652 MiB | 0.00% | 265.8s | 0.07385 | 1.000x | 5.24 MiB | 0.00 MiB | PASS |
| NEU-DET | Frozen Backbone | 0.2870 | 0.5621 | 0.4984 | 0.5442 | 1,225,522 / 2,591,010 | 47.30% | 1700 MiB | 35.91% | 239.7s | 0.06658 | 0.902x | 5.24 MiB | 0.00 MiB | PASS |
| NEU-DET | V-PEFT | 0.2933 | 0.5792 | 0.6980 | 0.5128 | 613,602 / 2,772,770 | 22.13% | 2621 MiB | 1.16% | 298.4s | 0.08288 | 1.122x | 5.75 MiB | 0.73 MiB | PASS |
| DeepPCB | Full-SFT | 0.5951 | 0.9029 | 0.8870 | 0.8311 | 2,590,994 / 2,591,010 | 100.00% | 2673 MiB | 0.00% | 283.2s | 0.07866 | 1.000x | 5.24 MiB | 0.00 MiB | PASS |
| DeepPCB | Frozen Backbone | 0.4534 | 0.7712 | 0.7575 | 0.7192 | 1,225,522 / 2,591,010 | 47.30% | 1710 MiB | 36.02% | 254.0s | 0.07056 | 0.897x | 5.24 MiB | 0.00 MiB | PASS |
| DeepPCB | V-PEFT | 0.4633 | 0.7318 | 0.6960 | 0.6938 | 613,602 / 2,772,770 | 22.13% | 2642 MiB | 1.15% | 314.0s | 0.08721 | 1.109x | 5.75 MiB | 0.73 MiB | PASS |
