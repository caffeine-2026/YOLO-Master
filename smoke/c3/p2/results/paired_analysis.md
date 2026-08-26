# C3 P2 Paired Seed Analysis

Delta is method B minus method A. Intervals use n=3 paired deltas; no p-value claim is made.

| Dataset | Images | A | B | Δ824 / Δ825 / Δ826 | Mean Δ | 95% CI | Direction |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| NEU-DET | 10 | Full-SFT | V-PEFT | -0.0161 / +0.0182 / -0.0082 | -0.0020 | [-0.0466, +0.0426] | mixed |
| NEU-DET | 10 | Full-SFT | Frozen Backbone | +0.0218 / +0.0223 / -0.0102 | +0.0113 | [-0.0349, +0.0575] | mixed |
| NEU-DET | 10 | Frozen Backbone | V-PEFT | -0.0379 / -0.0041 / +0.0020 | -0.0133 | [-0.0666, +0.0400] | mixed |
| NEU-DET | 50 | Full-SFT | V-PEFT | -0.0557 / +0.0038 / -0.0025 | -0.0181 | [-0.0993, +0.0630] | mixed |
| NEU-DET | 50 | Full-SFT | Frozen Backbone | -0.0557 / -0.0434 / -0.0484 | -0.0492 | [-0.0644, -0.0339] | all_negative |
| NEU-DET | 50 | Frozen Backbone | V-PEFT | -0.0000 / +0.0473 / +0.0459 | +0.0310 | [-0.0358, +0.0979] | mixed |
| NEU-DET | 100 | Full-SFT | V-PEFT | -0.0012 / -0.0161 / -0.0207 | -0.0126 | [-0.0378, +0.0126] | all_negative |
| NEU-DET | 100 | Full-SFT | Frozen Backbone | -0.0391 / -0.0355 / -0.0436 | -0.0394 | [-0.0495, -0.0294] | all_negative |
| NEU-DET | 100 | Frozen Backbone | V-PEFT | +0.0379 / +0.0195 / +0.0230 | +0.0268 | [+0.0025, +0.0511] | all_positive |
| NEU-DET | 500 | Full-SFT | V-PEFT | -0.0052 / -0.0156 / -0.0051 | -0.0087 | [-0.0237, +0.0064] | all_negative |
| NEU-DET | 500 | Full-SFT | Frozen Backbone | -0.0097 / -0.0304 / -0.0276 | -0.0226 | [-0.0505, +0.0053] | all_negative |
| NEU-DET | 500 | Frozen Backbone | V-PEFT | +0.0045 / +0.0148 / +0.0225 | +0.0139 | [-0.0085, +0.0364] | all_positive |
| DeepPCB | 10 | Full-SFT | V-PEFT | -0.0584 / -0.1270 / -0.1165 | -0.1006 | [-0.1924, -0.0089] | all_negative |
| DeepPCB | 10 | Full-SFT | Frozen Backbone | -0.0424 / -0.0840 / -0.0828 | -0.0697 | [-0.1286, -0.0109] | all_negative |
| DeepPCB | 10 | Frozen Backbone | V-PEFT | -0.0160 / -0.0429 / -0.0337 | -0.0309 | [-0.0649, +0.0031] | all_negative |
| DeepPCB | 50 | Full-SFT | V-PEFT | -0.2015 / -0.2021 / -0.1845 | -0.1960 | [-0.2208, -0.1713] | all_negative |
| DeepPCB | 50 | Full-SFT | Frozen Backbone | -0.1774 / -0.1843 / -0.1696 | -0.1771 | [-0.1955, -0.1587] | all_negative |
| DeepPCB | 50 | Frozen Backbone | V-PEFT | -0.0241 / -0.0178 / -0.0150 | -0.0190 | [-0.0306, -0.0074] | all_negative |
| DeepPCB | 100 | Full-SFT | V-PEFT | -0.1230 / -0.1274 / -0.1456 | -0.1320 | [-0.1618, -0.1022] | all_negative |
| DeepPCB | 100 | Full-SFT | Frozen Backbone | -0.1591 / -0.1585 / -0.1751 | -0.1642 | [-0.1877, -0.1408] | all_negative |
| DeepPCB | 100 | Frozen Backbone | V-PEFT | +0.0361 / +0.0311 / +0.0295 | +0.0323 | [+0.0237, +0.0408] | all_positive |
| DeepPCB | 500 | Full-SFT | V-PEFT | -0.0435 / -0.0441 / -0.0451 | -0.0442 | [-0.0462, -0.0423] | all_negative |
| DeepPCB | 500 | Full-SFT | Frozen Backbone | -0.1121 / -0.0869 / -0.1094 | -0.1028 | [-0.1372, -0.0684] | all_negative |
| DeepPCB | 500 | Frozen Backbone | V-PEFT | +0.0686 / +0.0428 / +0.0643 | +0.0585 | [+0.0242, +0.0929] | all_positive |
