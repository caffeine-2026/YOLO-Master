# C3 P1 seed824 50-Epoch Candidate

30 epoch 结果仅为历史 pilot；本表为当前 50 epoch convergence candidate。

| Dataset | Method | mAP50-95 | mAP50 | Precision | Recall | Trainable / Total | Trainable Ratio | Peak GPU Mem | Memory Saving | Time | GPU-hours | Time Ratio | Checkpoint | Adapter | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.2901 | 0.5580 | 0.7520 | 0.4788 | 2,590,994 / 2,591,010 | 100.00% | 2652 MiB | 0.00% | 182.3s | 0.05065 | 1.000x | 5.24 MiB | 0.00 MiB | PASS |
| NEU-DET | Frozen Backbone | 0.2708 | 0.5410 | 0.6636 | 0.4983 | 1,225,522 / 2,591,010 | 47.30% | 1700 MiB | 35.91% | 166.9s | 0.04635 | 0.915x | 5.24 MiB | 0.00 MiB | PASS |
| NEU-DET | V-PEFT | 0.2379 | 0.4827 | 0.7208 | 0.4158 | 613,602 / 2,772,770 | 22.13% | 2621 MiB | 1.16% | 211.7s | 0.05879 | 1.161x | 5.75 MiB | 0.73 MiB | PASS |
| DeepPCB | Full-SFT | 0.5324 | 0.8027 | 0.7696 | 0.7563 | 2,590,994 / 2,591,010 | 100.00% | 2673 MiB | 0.00% | 195.8s | 0.05439 | 1.000x | 5.24 MiB | 0.00 MiB | PASS |
| DeepPCB | Frozen Backbone | 0.4045 | 0.6861 | 0.6586 | 0.6521 | 1,225,522 / 2,591,010 | 47.30% | 1710 MiB | 36.02% | 180.7s | 0.05020 | 0.923x | 5.24 MiB | 0.00 MiB | PASS |
| DeepPCB | V-PEFT | 0.3631 | 0.5784 | 0.5217 | 0.6320 | 613,602 / 2,772,770 | 22.13% | 2642 MiB | 1.15% | 213.9s | 0.05942 | 1.093x | 5.75 MiB | 0.73 MiB | PASS |
