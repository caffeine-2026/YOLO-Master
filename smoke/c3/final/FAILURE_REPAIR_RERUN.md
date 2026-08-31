# Failure, repair, and rerun record

실패 기록은 삭제하지 않았고 숫자를 임의 보정하지 않았다.

## 1. DCO rank exceeds layer capacity

- 재현 run: `neu_det_vpeft_dco_gpu_fp32_seed824`, `deeppcb_vpeft_dco_gpu_fp32_seed824`.
- 원본 오류: `ValueError: PlacementPlan rank 64 for '1.conv' exceeds layer capacity 16`.
- 원본 로그: `smoke/c3/p0/logs/neu_det_vpeft_dco_gpu_fp32_seed824/train.log`, `smoke/c3/p0/logs/deeppcb_vpeft_dco_gpu_fp32_seed824/train.log`.
- 원인: hard constraint registry가 `rank <= min(in_channels, out_channels)`를 solver projection 전에 강제하지 않았다.
- 수정 코드: `ultralytics/vpeft/constraints.py`의 `VariantModuleCompatibilityConstraint`; runner는 실패 시 checkpoint가 없는데 adapter export를 재시도하던 동작도 `smoke/c3/p0/tools/run_smoke.py`에서 중단하도록 수정됐다.
- 회귀 테스트: `tests/test_vpeft.py`, `tests/test_vpeft_lora_e2e.py`.
- 수정 후 run: `neu_det_vpeft_dco_fixed_gpu_fp32_seed824`, `deeppcb_vpeft_dco_fixed_gpu_fp32_seed824`; 두 run 모두 exit 0. `1.conv` rank는 16이며 실제 training/adapter export가 완료됐다.

## 2. MIPR dependency fallback

- run: `neu_det_vpeft_mip_fallback_gpu_fp32_seed824`.
- 원인: OR-Tools 미설치로 `MIPRelaxationSolver`가 `ImportError`를 발생시켰다.
- 처리: AO로 명시적 fallback.
- 감사 필드: requested `mip`, effective `ao`, exception type `ImportError`, OR-Tools 원인이 runtime metadata와 solver audit JSON에 모두 남아 있다.

## 3. Delivery validator failures found in this audit

### P1 stale P0 path

- 실패 명령: `.venv/bin/python smoke/c3/p1/scripts/validate_p1.py --epochs 100 --output smoke/c3/p1/evidence/revalidation_20260831.json`.
- 실패: P0가 `smoke/c3/p0/`로 정리됐지만 validator가 `smoke/c3/evidence/c3_p0_summary.json`을 읽어 `FileNotFoundError`.
- 수정: 현재 P0 경로를 사용하고, 정리 직후 P0 experiment evidence 기준 ref `f4b1af0e5aa5669bd14dee2660cb0b8286bb293d`에 대해 immutable run evidence만 비교.
- 재실행: `overall_status=PASS`.

### P2 archived gate schema mismatch

- 실패 명령: `.venv/bin/python smoke/c3/p2/tools/validate_p2.py`.
- 실패: final multiseed protocol에서 제거된 `training.seeds_this_stage`를 읽어 `KeyError`.
- 수정: final schema의 `immutable_reused_seed`, `final_seeds`, `immutable_seed824_p2_runs`로 역사 seed-824 gate를 검증. 이후 seed 825/826가 존재하더라도 당시 manifest를 검증하도록 변경.
- 재실행: `P2_SEED824_STAGE=PASS`, `MULTISEED_READY=YES`.

### P2 final gate over-broad immutability scope

- 실패 명령: `.venv/bin/python smoke/c3/p2/tools/validate_p2_final.py`.
- 실패: 72/72 run, 24/24 V-PEFT, 14/14 figures는 모두 통과했지만 `p1_history_unmodified=false`였다. 검증기가 `smoke/c3/p1` 전체를 비교해 이번 P1 validator 복구와 새 revalidation JSON까지 원본 실험 변경으로 오인했다.
- 수정: 불변 범위를 P1 원본 logs, artifacts, split/multiseed manifests, raw metrics CSV로 제한했다. 코드와 새 감사 결과는 실험 이력이 아니다.
- 재실행: `P2_FINAL_VALIDATION=PASS`, `OVERALL_C3_P2=PASS`.

### Hygiene scope false positives

- 첫 통합 검증은 ignored `node_modules`의 JSONC/README와 raw model artifacts의 생성기 `save_dir`까지 publishable log로 오인해 FAIL했다.
- 수정: JSON/YAML parsing은 tracked/non-ignored delivery 전체에 적용하고, privacy scan은 제출 대상 logs/docs/config/evidence/results/source에 적용했다. raw artifacts는 별도로 manifest SHA-256, config cross-check, checkpoint load로 검증한다.
- 재실행: hygiene `PASS`.

### Manual LoRA fallback base parameters remained trainable

- 최초 관련 suite 결과: 59 passed, 2 failed. 한 건은 sandbox loopback 제한이고, 다른 한 건은 `test_planner_adapter_full_lifecycle`에서 non-target base convolution이 trainable로 남은 실제 회귀였다.
- 원인: fallback wrapper가 선택된 Conv2d 내부만 동결하고 전체 base model을 먼저 동결하지 않았다.
- 수정: `ultralytics/utils/lora/fallback.py::apply_manual_lora`가 adapter 삽입 전에 base parameters 전체를 동결하고, 이후 LoRA tensors와 알려진 detection head만 선택적으로 trainable로 만든다.
- 재실행: lifecycle/fallback 집중 테스트 17/17 PASS; 핵심 V-PEFT/LoRA/P0 suite 65/65 PASS; 관련 LoRA/MoLoRA/PEFT/Planner pytest 전체 295 passed / 7 skipped / 2 warnings.
- sandbox의 CPU Gloo 실패는 실제 호스트에서 동일 테스트를 재실행해 1/1 PASS했다.

## 4. Experiment rerun decision

P1/P2 72 cells는 모두 100 epochs, locked test, 원본 CSV, resource/time, artifact hashes, checkpoint load, seed/config 조건을 통과했다. 따라서 새로운 training run은 실행하지 않았다. 이는 유효한 기존 실험을 재사용하고 누락·오류·조건 불일치 때만 재실행한다는 원칙에 따른 결정이다.
