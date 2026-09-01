# C3 V-PEFT 데이터 증강 ablation 보고서

## 결론

기준 코드는 `bf6c7c508635dec0be849aedaa3eac5d88ed220d`, 기능 브랜치는 `codex/c3-augmentation-ablation-20260831`이다. 기존 P0/P1/P2, checkpoint, CSV, JSON은 수정하거나 덮어쓰지 않았다. 정확도 최적화 기준은 613,602 trainable / 2,772,770 total parameter V-PEFT이며, 195,410-parameter(Full-SFT의 10% 이하) 설정은 efficiency-first negative result로만 참조했다.

100-shot validation-only 탐색에서 NEU-DET는 `mild`, DeepPCB는 `medium`이 선택됐다. 최종 locked test에서 DeepPCB medium은 기존 accuracy-first V-PEFT보다 mAP50-95가 **+0.10960**, paired 95% CI **[0.05316, 0.16605]**로 개선됐다. NEU-DET mild는 기존 baseline보다 **-0.01433**, CI **[-0.06441, 0.03575]**였으므로 개선으로 해석하지 않는다.

## 누수 방지와 고정 조건

[사전 protocol](../config/augmentation_protocol.yaml)은 train/validation만 읽는 YAML을 사용하고 모든 search metrics의 `test`를 null로 강제했다. seed 824의 고정 train membership을 seed 824/825/826 학습에서 공유했다. epoch 100, image size 640, batch 8, AdamW, AMP false, deterministic true, V-PEFT rank 8, full head policy, AO planner 조건을 고정했다.

[최종 선택](../results/frozen_selection.json)은 2026-08-31T15:33:23Z에 validation 결과만으로 동결됐으며 `test_access_before_selection=false`, `test_metrics_used_for_selection=false`이다. 동결 뒤 test 결과를 보고 정책을 다시 선택하지 않았다.

## 영상 특성과 정책

[dataset audit](../evidence/dataset_characteristics.json)에서 두 데이터셋 모두 channel difference가 0인 grayscale/binary 영상임을 확인했다. NEU-DET 100-shot은 200×200, box area 중앙값 0.1258이었다. DeepPCB는 99/100장이 640×640 단일 채널이고 box area 중앙값이 0.002988이라 작은 결함 손실 위험이 더 컸다.

따라서 hue/saturation, mosaic, mixup, cutmix, copy-paste, shear, perspective를 전 정책에서 0으로 두었다. NEU-DET에는 제한된 회전·이동·scale·양방향 flip과 grayscale-safe brightness/contrast/blur/noise를 허용했다. DeepPCB는 rotation/translation/scale/blur/noise를 더 약하게 하고 vertical flip은 mild/medium에서 껐다. 새 photometric 변환은 세 채널에 같은 값을 적용하며 annotation을 변경하지 않는다. flip/translation annotation은 회귀 테스트로 좌표를 검증했다.

## Validation-only 탐색과 포화

초기 seed 824 validation mAP50-95는 다음과 같다.

| Dataset | baseline | mild | medium | strong | 선택 |
|---|---:|---:|---:|---:|---|
| NEU-DET | 0.25082 | **0.28728** | 0.26237 | 0.26723 | mild |
| DeepPCB | 0.53848 | 0.63361 | **0.63636** | 0.62200 | medium |

세 seed confirmation에서 validation paired delta는 NEU-DET mild +0.05284, 95% CI [-0.01974, 0.12542], DeepPCB medium +0.09975, CI [0.05450, 0.14500]이었다. 사전 scaling trigger가 “mean ≥ 0.005 및 CI lower > 0”이므로 DeepPCB만 10/50/100/500-shot으로 확장했다.

NEU-DET는 mild 이후 medium/strong에서 하락했다. DeepPCB는 mild→medium 추가 이득이 0.00275에 그쳤고 strong에서 하락했다. 즉 테스트한 강도 범위에서 두 데이터셋 모두 포화 후 악화가 관찰됐다. 이는 **테스트한 증강 범위에서의 경험적 상한**이며 이론적 상한이 아니다. 원본 행과 그래프 입력 해시는 [initial_search.csv](../results/initial_search.csv) 및 [figure_manifest.json](../figures/figure_manifest.json)에 있다.

## 100-shot locked-test 결과

괄호는 seed 824/825/826의 sample standard deviation이다.

| Dataset / 정책 | mAP50-95 | mAP50 | Precision | Recall | Full-SFT retention |
|---|---:|---:|---:|---:|---:|
| NEU no augmentation | 0.22739 (0.01890) | 0.47575 | 0.49858 | 0.48765 | 68.30% |
| NEU mild | **0.30596 (0.01461)** | **0.60061** | **0.59058** | **0.60675** | **91.90%** |
| NEU existing accuracy-first V-PEFT | 0.32029 | 0.61105 | — | — | 96.20% |
| NEU Full-SFT reference | 0.33293 | 0.63778 | — | — | 100% |
| DeepPCB no augmentation | 0.53644 (0.00598) | 0.71753 | 0.71992 | 0.68182 | 82.71% |
| DeepPCB medium | **0.62621 (0.00865)** | **0.83303** | **0.81365** | **0.79355** | **96.55%** |
| DeepPCB existing accuracy-first V-PEFT | 0.51661 | 0.79585 | — | — | 79.65% |
| DeepPCB Full-SFT reference | 0.64860 | 0.92256 | — | — | 100% |

새 no-augmentation과의 paired mAP50-95 delta는 NEU +0.07857, CI [-0.00463, 0.16177]로 유의하지 않았고 DeepPCB +0.08977, CI [0.06140, 0.11814]로 유의했다. 사용자가 지정한 **기존** accuracy-first baseline과의 delta는 각각 -0.01433 및 +0.10960이다. 통계 원본은 [paired_test_statistics.csv](../results/paired_test_statistics.csv), [historical_baseline_paired_statistics.csv](../results/historical_baseline_paired_statistics.csv), seed별 값은 [locked_test_runs.csv](../results/locked_test_runs.csv)에 있다.

## 자원 trade-off

| Dataset / 정책 | trainable / total | Peak GPU MiB | 평균 학습 초 | 평균 GPU-hours |
|---|---:|---:|---:|---:|
| NEU no augmentation | 613,602 / 2,772,770 | 2,621.44 | 336.331 | 0.09343 |
| NEU mild | 613,602 / 2,772,770 | 2,621.44 | 526.754 | 0.14632 |
| DeepPCB no augmentation | 613,602 / 2,772,770 | 2,621.44 | 347.343 | 0.09648 |
| DeepPCB medium | 613,602 / 2,772,770 | 2,621.44 | 546.054 | 0.15168 |

증강은 trainable parameter와 측정 peak GPU memory를 늘리지 않았지만, CPU photometric/geometric 처리 때문에 100-shot 학습 시간이 NEU 56.6%, DeepPCB 57.2% 증가했다. 기존/신규/Full-SFT/≤10% 설정 전체 비교는 [reference_comparison.csv](../results/reference_comparison.csv)에 있다.

## DeepPCB scaling

| Shots | Full-SFT 참조 | V-PEFT no-aug | V-PEFT medium | paired delta (95% CI) |
|---:|---:|---:|---:|---:|
| 10 | 0.29034 | 0.20000 | **0.29082** | +0.09082 [0.01594, 0.16569] |
| 50 | 0.56690 | 0.45055 | **0.54399** | +0.09344 [0.07225, 0.11463] |
| 100 | 0.64860 | 0.53644 | **0.62621** | +0.08977 [0.06140, 0.11814] |
| 500 | 0.70065 | 0.69124 | **0.73171** | +0.04047 [0.01963, 0.06130] |

네 scale 모두 paired CI lower가 0보다 컸다. 그러나 delta가 500-shot에서 약 +0.0405로 줄어 데이터가 늘수록 증강 이득이 감소하는 포화 경향이 있다. Full-SFT는 이전 고정 실험의 참고 upper baseline이며 새 V-PEFT와 동일 augmentation run이 아니므로, 500-shot의 104.43% retention을 보편적 우위로 해석하지 않는다. 원본은 [scaling_comparison.csv](../results/scaling_comparison.csv)와 [DeepPCB scaling 그래프](../figures/deeppcb_scaling_accuracy.png)에 있다. NEU scaling은 trigger 미충족으로 실행하지 않았다.

## Per-class 분석

100-shot AP50-95 paired delta에서 NEU는 crazing +0.03178, inclusion +0.11648, patches +0.09631, rolled-in_scale +0.07411의 CI가 양수였다. pitted_surface +0.11908와 scratches +0.03365는 CI가 0을 포함했다. DeepPCB는 open +0.09891, short +0.14224, mousebite +0.11215, spur +0.14909가 유의했고, 이미 높은 copper +0.00912와 pin-hole +0.02711은 유의하지 않았다. 어떤 클래스도 95% CI에서 유의한 하락은 없었다. 전체 AP50/precision/recall과 seed별 값은 [per_class_test_runs.csv](../results/per_class_test_runs.csv), [per_class_paired_statistics.csv](../results/per_class_paired_statistics.csv), [그래프](../figures/per_class_ap_comparison.png)에 있다.

## 원본 증거

- 모든 leaf 명령, resolved config, full stdout/stderr, metrics, resource/time: [logs](../logs)
- locked-test overall/per-class JSON: [evaluations](../evaluations)
- checkpoint와 학습 시각화: [artifacts](../artifacts); gitignored local files이며 각 run의 `logs/<phase>/<run>/artifact_manifest.json`과 `metrics.json`에 size/SHA-256이 있다.
- search/confirmation/scaling scheduler: [logs](../logs)
- 실패·수정·재실행: [failures](../failures), [locked-test equivalence audit](../evidence/locked_test_retry_equivalence.json)
- 실제 명령 목록: [EXECUTED_COMMANDS.md](EXECUTED_COMMANDS.md)

## 실패와 제한

1. 새 annotation 회귀 테스트 fixture의 잘못된 기대값과 통계 테스트 상수 오타는 첫 실패 로그를 보존하고 테스트 기대값만 수정했다.
2. 최초 100-shot test orchestration이 중단된 뒤 DeepPCB medium 세 child가 완료됐다. full stdout/timing이 부족해 동일 checkpoint를 재평가했으며 결과·per-class·checkpoint hash가 모두 동일했다. 이 세 checkpoint의 물리적 test 접근은 2회라는 protocol 예외를 [audit](../evidence/locked_test_retry_equivalence.json)에 명시했다. 첫 결과 이후 선택/튜닝은 없었다.
3. scaling locked-test 첫 호출은 sandbox `nvidia-smi` preflight에서 평가 시작 전에 실패했다. 호스트에서 GPU idle을 확인한 뒤 같은 명령으로 18/18 완료했다.
4. release pytest의 첫 sandbox 실행은 localhost socket 금지로 425 passed/1 failed/17 skipped였다. 호스트 재실행은 426 passed/17 skipped였다. 두 결과 모두 [failures](../failures)와 [release evidence](../evidence/release_checks)에 있다.
5. Python CSV writer의 CRLF가 `git diff --check`에서 trailing whitespace로 판정됐다. generator를 LF 고정으로 수정하고 신규 CSV 줄끝만 정규화했다. 그래프를 다시 생성했으며 plotted rows와 PNG SHA-256은 전후 동일하고 source-file SHA-256만 줄끝 변경을 반영했다. 이전 manifest는 [failure bundle](../failures/figures_pre_lf_normalization)에 보존했다.
6. seed가 3개뿐이므로 paired t CI(df=2)가 넓다. NEU의 기존 baseline 대비 정확도 개선은 입증되지 않았다.
7. 새 checkpoint 약 1.2 GiB는 로컬에 보존되나 저장소 정책상 gitignored이다. delivery validator가 실제 load/hash/config/seed/epoch를 검사한다.

## 검증 상태

P0/P1/P2와 통합 연구 validator 5개는 기존 보호 파일 hash 불변 상태로 PASS했다. bf6c completion validator는 분리 worktree에서 24 runs/72 checkpoints를 load해 PASS했다. 관련 pytest 426 passed/17 skipped, Ruff/format/critical legacy/diff gate가 PASS했다. 최종 augmentation delivery validator 결과는 [evidence](../evidence)에 저장한다.
