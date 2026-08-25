# C3 P1 Pilot 三方对照（seed 824）

以下为统一协议、固定 test split 的单 seed pilot，不构成方法优劣或多 seed 结论。

| Dataset | Method | mAP50-95 | mAP50 | Precision | Recall | Trainable / Total | Trainable vs Full | Peak GPU Mem | Memory Saving | Time | GPU-hours | Time Ratio | Checkpoint | Adapter | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NEU-DET | Full-SFT | 0.2574 | 0.4919 | 0.4574 | 0.5392 | 2,590,994 / 2,591,010 | 100.00% | 2652 MiB | 0.00% | 115.3s | 0.03201 | 1.000× | 5.23 MiB | 0.00 MiB | PASS |
| NEU-DET | Frozen Backbone | 0.2384 | 0.4922 | 0.6109 | 0.4761 | 1,225,522 / 2,591,010 | 47.30% | 1700 MiB | 35.91% | 106.5s | 0.02959 | 0.924× | 5.23 MiB | 0.00 MiB | PASS |
| NEU-DET | V-PEFT | 0.2043 | 0.3875 | 0.3991 | 0.3922 | 613,602 / 2,772,770 | 23.68% | 2621 MiB | 1.16% | 129.8s | 0.03605 | 1.126× | 5.75 MiB | 0.73 MiB | PASS |
| DeepPCB | Full-SFT | 0.4294 | 0.6606 | 0.5914 | 0.6741 | 2,590,994 / 2,591,010 | 100.00% | 2673 MiB | 0.00% | 119.6s | 0.03322 | 1.000× | 5.23 MiB | 0.00 MiB | PASS |
| DeepPCB | Frozen Backbone | 0.3057 | 0.5264 | 0.4857 | 0.5893 | 1,225,522 / 2,591,010 | 47.30% | 1710 MiB | 36.02% | 112.7s | 0.03131 | 0.942× | 5.23 MiB | 0.00 MiB | PASS |
| DeepPCB | V-PEFT | 0.2117 | 0.3877 | 0.3575 | 0.5029 | 613,602 / 2,772,770 | 23.68% | 2642 MiB | 1.15% | 137.3s | 0.03815 | 1.148× | 5.75 MiB | 0.73 MiB | PASS |
