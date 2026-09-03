# C3 Research Evidence Delivery

이 디렉터리는 웹 데모가 아니라 DeepPCB/NEU-DET C3 연구 실험의 최종 증거 인덱스다. 기존의 유효한 실험은 재실행하지 않고 원본 full log, resolved config, `args.yaml`, epoch CSV, locked-test JSON, resource/timing JSON, artifact SHA-256, checkpoint를 교차검증했다. 기초 통합 validator 결과는 `PASS`이며 72/72 training cells와 72/72 `best.pt` load 검사를 통과했다. 이후 수행한 native MIP, Planner 3분기, learned LOVO, ≤10% 파라미터 실험과 데이터 증강 소거 실험은 각각 [completion report](../completion/docs/C3_COMPLETION_REPORT.md)와 [augmentation report](../augmentation/docs/AUGMENTATION_ABLATION_REPORT.md)에 연결한다.

## 1. P1 three-way comparison

모든 값은 seed 824/825/826, 100 images, 100 epochs, batch 8, imgsz 640, AdamW, FP32, 동일 augmentation 및 고정 test split의 평균이다. 대괄호는 mAP의 two-sided 95% Student-t CI (n=3, df=2)다. Accuracy retention은 같은 데이터셋 Full-SFT mAP50-95 평균 대비 비율이다.

| Dataset | Method | mAP50-95 mean [95% CI] | mAP50 mean [95% CI] | Trainable / total params | Peak MiB | Time s / GPU-h | Retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NEU-DET | Full-SFT | 0.332935 [0.323344, 0.342525] | 0.637783 [0.632831, 0.642735] | 2,590,994 / 2,591,010 | 2,652.16 | 356.168 / 0.098935 | 100.00% |
| NEU-DET | Frozen Backbone | 0.293499 [0.276374, 0.310623] | 0.576870 [0.566847, 0.586892] | 1,225,522 / 2,591,010 | 1,699.84 | 325.120 / 0.090311 | 88.15% |
| NEU-DET | V-PEFT | 0.320288 [0.300222, 0.340353] | 0.611046 [0.579120, 0.642973] | 613,602 / 2,772,770 | 2,621.44 | 402.629 / 0.111841 | 96.20% |
| DeepPCB | Full-SFT | 0.648599 [0.616436, 0.680763] | 0.922556 [0.916093, 0.929019] | 2,590,994 / 2,591,010 | 2,665.81 | 373.784 / 0.103829 | 100.00% |
| DeepPCB | Frozen Backbone | 0.484355 [0.447318, 0.521391] | 0.805972 [0.763410, 0.848533] | 1,225,522 / 2,591,010 | 1,706.67 | 340.116 / 0.094477 | 74.68% |
| DeepPCB | V-PEFT | 0.516607 [0.481275, 0.551939] | 0.795849 [0.755903, 0.835795] | 613,602 / 2,772,770 | 2,631.68 | 423.744 / 0.117707 | 79.65% |

V-PEFT는 Full-SFT보다 trainable parameters가 76.32% 적지만 이 구현에서 GPU memory 절감은 NEU 1.16%, DeepPCB 1.28%에 그쳤고 training time은 각각 13.04%, 13.37% 늘었다. 100-shot에서 V-PEFT−Frozen mAP50-95 paired mean은 NEU +0.02679 (95% CI [0.00249, 0.05109]), DeepPCB +0.03225 ([0.02369, 0.04082])다. 따라서 V-PEFT를 보편적 우승자로 표현하지 않고 정확도·학습 가능 파라미터·메모리·시간의 trade-off로 해석한다.

원본/통계: [P1 all runs](../p1/results/p1_all_runs.csv), [P1 summary](../p1/results/p1_summary.csv), [paired analysis](../p1/results/paired_full_vs_vpeft.csv), [P1 report](../p1/docs/C3_P1_REPORT.md).

## 2. P2 scaling curve

아래는 seed 824/825/826 mAP50-95 평균이다. 100-shot cells는 검증 후 P1에서 재사용했고, 나머지 54 cells도 기존의 완료된 실험을 검증해 재사용했다.

| Dataset | Images | Full-SFT | Frozen Backbone | V-PEFT | V-PEFT retention vs Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| NEU-DET | 10 | 0.1212 | 0.1325 | 0.1192 | 98.33% |
| NEU-DET | 50 | 0.2687 | 0.2195 | 0.2505 | 93.26% |
| NEU-DET | 100 | 0.3329 | 0.2935 | 0.3203 | 96.20% |
| NEU-DET | 500 | 0.3995 | 0.3769 | 0.3909 | 97.83% |
| DeepPCB | 10 | 0.2903 | 0.2206 | 0.1897 | 65.34% |
| DeepPCB | 50 | 0.5669 | 0.3898 | 0.3709 | 65.42% |
| DeepPCB | 100 | 0.6486 | 0.4844 | 0.5166 | 79.65% |
| DeepPCB | 500 | 0.7006 | 0.5979 | 0.6564 | 93.69% |

- CSV: [72-cell raw table](../p2/results/p2_all_runs.csv), [mean/95% CI](../p2/results/p2_summary.csv), [retention](../p2/results/retention_multiseed.csv), [paired deltas](../p2/results/paired_analysis.csv).
- Curves: [NEU mAP50-95](../p2/visualizations/final/neu_map5095_multiseed.png), [NEU mAP50](../p2/visualizations/final/neu_map50_multiseed.png), [DeepPCB mAP50-95](../p2/visualizations/final/deeppcb_map5095_multiseed.png), [DeepPCB mAP50](../p2/visualizations/final/deeppcb_map50_multiseed.png).
- Efficiency: [NEU accuracy/params](../p2/visualizations/final/neu_accuracy_vs_params_multiseed.png), [DeepPCB accuracy/params](../p2/visualizations/final/deeppcb_accuracy_vs_params_multiseed.png), [P2 report](../p2/docs/C3_P2_REPORT.md).

## 3. Planner and solver evidence

Planner 흐름도는 [Planner flow and solver audit](../p0/docs/PLANNER_FLOW_AND_SOLVER_AUDIT_20260831.md)에 있다.

| Dataset | Requested → effective | Decision | Budget | Planned / applied modules | Ranks | Result |
| --- | --- | --- | ---: | ---: | --- | --- |
| NEU-DET | AO → AO | ACCEPT | 2,100,000 | 59 / 52 | 8 | completed |
| DeepPCB | AO → AO | ACCEPT | 2,100,000 | 59 / 52 | 8 | completed |
| NEU-DET | DCO → DCO | ACCEPT | 2,100,000 | 59 / 52 | 8/16/32/48/64 | completed after fix |
| DeepPCB | DCO → DCO | ACCEPT | 2,100,000 | 59 / 52 | 8/16/32/48/64 | completed after fix |
| NEU-DET | MIPR → AO | ACCEPT | 2,100,000 | 59 / 52 | 8 | OR-Tools ImportError fallback |
| NEU-DET | MIP → MIP | ACCEPT | 2,100,000 | 59 planned | 8/16/32/64 | native SCIP, OPTIMAL |
| DeepPCB | MIP → MIP | ACCEPT | 2,100,000 | 59 planned | 8/16/32/64 | native SCIP, OPTIMAL |

기초 구조화 근거는 [solver audit JSON](../p0/evidence/solver_audit_20260831.json), 최신 native MIP 및 Planner 분기 근거는 [solver comparison](../completion/evidence/solvers/solver_comparison.json)과 [Planner branches](../completion/evidence/planner_branches/planner_branches.json)에 있다. 실패·수정·재실행 기록은 [FAILURE_REPAIR_RERUN.md](FAILURE_REPAIR_RERUN.md)에 있다.

## 4. LOVO learned calibration과 경계

기초 감사 당시 값 `predicted_delta=0.06602954545454547`, confidence `0`, state `cold_start`, source `default_prior`는 측정 ΔmAP도 V-PEFT 향상 증거도 아니다. 이후 locked test를 사용하지 않고 6개 calibration 단위와 2개 held-out 단위로 learned regression을 실행했다. 최신 결과는 predicted ΔmAP50-95 `-0.16002`, confidence `0.01667`, held-out RMSE `0.10020`, prediction interval `[-0.40821, 0.08816]`이다.

따라서 LOVO 구현과 정식 calibration 증거는 완료됐지만, 작은 표본, design rank 1/12와 넓은 구간 때문에 **low-confidence limited evidence**로만 해석한다. P1/P2의 역사 metadata는 당시 값을 원본 그대로 유지한다. 근거는 [LOVO calibration report](../completion/evidence/lovo/lovo_calibration_report.json)에 있다.

## 5. Evidence index

- 실제 GPU preflight: [gpu_preflight.json](evidence/gpu_preflight.json).
- 79개 원본 command files 통합본: [raw_command_manifest.json](evidence/raw_command_manifest.json). 각 run의 `command.txt` SHA-256과 원문을 보존한다.
- 원본 full logs/config/metrics/resources/time: 각 run의 `smoke/c3/p1/logs/<run_id>/` 또는 `smoke/c3/p2/logs/<run_id>/`; 경로는 [통합 validation JSON](evidence/research_delivery_validation.json)에 기록돼 있다.
- P0 full train logs: [P0 logs index](../p0/logs/README.md).
- P1 seed별 결과와 95% CI: [P1 all runs](../p1/results/p1_all_runs.csv), [P1 summary](../p1/results/p1_summary.csv).
- P2 seed별 결과와 95% CI: [P2 all runs](../p2/results/p2_all_runs.csv), [P2 summary](../p2/results/p2_summary.csv).
- Checkpoint/adapter SHA-256: 각 run의 `artifact_manifest.json`; 통합 validator가 72개 `best.pt`를 실제 load했다.
- 실행·검증 명령: [EXECUTED_COMMANDS.md](EXECUTED_COMMANDS.md).
- GitHub 게시용 중국어 보고서: [GITHUB_PROGRESS_ZH.md](GITHUB_PROGRESS_ZH.md).
- completion 검증과 최신 보강 결과: [completion report](../completion/docs/C3_COMPLETION_REPORT.md).
- augmentation 검증과 소거 결과: [augmentation report](../augmentation/docs/AUGMENTATION_ABLATION_REPORT.md).

## 6. Completion boundary and limitations

- P0: 완료. 두 데이터셋 V-PEFT, AO/DCO/native MIP, MIP 의존성 누락 시 AO fallback, 실패 로그와 DCO capacity fix 근거를 확인했다.
- P1: 완료. 두 데이터셋 × 세 전략 × 세 seed = 18/18, 동일 protocol, test 분리, mean/95% CI를 확인했다.
- P2: 완료. 두 데이터셋 × 4 scales × 3 strategies × 3 seeds = 72/72, nested split과 curve/CSV 일치를 확인했다.
- Planner 분기: 완료. `ACCEPT / ADAPT / REFUSE`의 실제 입력, 구조화 출력과 회귀 테스트를 확인했다.
- LOVO: 구현 및 calibration 증거 완료. 6 calibration + 2 held-out이지만 confidence 0.01667인 제한적 결과다.
- 파라미터 효율 보강: 완료. 195,410 trainable parameters로 Full-SFT의 7.54%를 달성했지만 평균 정확도는 하락했다.
- 데이터 증강: 완료. DeepPCB medium은 locked test에서 개선됐고 NEU-DET mild는 역사적 baseline 대비 개선으로 판단하지 않는다.
- 통계는 seed 3개라 CI가 넓을 수 있고, 100 fixed epochs는 scale별 optimizer update 수를 동일하게 만들지 않는다.
- V-PEFT의 trainable-parameter 절감은 크지만 이 구현의 memory/time 이점은 확인되지 않았다.

기초 72-cell 기계 검증 결과: [research_delivery_validation.json](evidence/research_delivery_validation.json). 이 JSON은 completion 및 augmentation 이전의 역사적 검증 범위이며, 최신 전체 상태는 [C3 root evidence index](../README.md)와 함께 확인한다.

관련 LoRA/MoLoRA/PEFT/Planner pytest 전체 결과: 295 passed, 7 skipped, 2 warnings, 0 failed. 상세 명령은 [EXECUTED_COMMANDS.md](EXECUTED_COMMANDS.md)에 있다.
