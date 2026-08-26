# C3 P1 seed824 100-Epoch Candidate

30 epoch 结果仅为历史 pilot；本表为当前 100 epoch convergence candidate。

| Dataset | Method | mAP50-95 | mAP50 | Precision | Recall | Trainable / Total | Trainable Ratio | Peak GPU Mem | Memory Saving | Time | GPU-hours | Time Ratio | Checkpoint | Adapter | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.3293 | 0.6401 | 0.5927 | 0.6448 | 2,590,994 / 2,591,010 | 100.00% | 2652 MiB | 0.00% | 349.7s | 0.09713 | 1.000x | 5.24 MiB | 0.00 MiB | PASS |
| NEU-DET | Frozen Backbone | 0.2901 | 0.5736 | 0.4913 | 0.5862 | 1,225,522 / 2,591,010 | 47.30% | 1700 MiB | 35.91% | 317.4s | 0.08817 | 0.908x | 5.24 MiB | 0.00 MiB | PASS |
| NEU-DET | V-PEFT | 0.3280 | 0.6239 | 0.6162 | 0.5839 | 613,602 / 2,772,770 | 22.13% | 2621 MiB | 1.16% | 395.1s | 0.10976 | 1.130x | 5.76 MiB | 0.73 MiB | PASS |
| DeepPCB | Full-SFT | 0.6345 | 0.9208 | 0.9040 | 0.8558 | 2,590,994 / 2,591,010 | 100.00% | 2673 MiB | 0.00% | 369.0s | 0.10250 | 1.000x | 5.24 MiB | 0.00 MiB | PASS |
| DeepPCB | Frozen Backbone | 0.4754 | 0.7938 | 0.7525 | 0.7587 | 1,225,522 / 2,591,010 | 47.30% | 1710 MiB | 36.02% | 335.6s | 0.09322 | 0.909x | 5.24 MiB | 0.00 MiB | PASS |
| DeepPCB | V-PEFT | 0.5115 | 0.7794 | 0.7591 | 0.7177 | 613,602 / 2,772,770 | 22.13% | 2642 MiB | 1.15% | 412.1s | 0.11448 | 1.117x | 5.76 MiB | 0.73 MiB | PASS |
