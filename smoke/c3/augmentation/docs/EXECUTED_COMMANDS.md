# 실제 실행 명령

아래 명령은 저장소 루트에서 실제 실행했다. 각 학습의 완전히 해석된 leaf 명령은 [search logs](../logs/search), [confirmation logs](../logs/confirm), [scaling logs](../logs/scaling)의 `command.txt`에 있다.

## 환경과 데이터 확인

```bash
git switch -c codex/c3-augmentation-ablation-20260831 bf6c7c508635dec0be849aedaa3eac5d88ed220d
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/analyze_dataset_characteristics.py
```

## Validation-only search, confirmation, freeze

```bash
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/run_suite.py --phase search --devices 1,3,5
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/select_initial.py
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/run_suite.py --phase confirm --devices 1,3,5
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/freeze_selection.py
```

## Locked test와 scaling

```bash
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/run_locked_test_suite.py --scope 100 --devices 1,3,5
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/run_locked_test_suite.py --scope retry-100-medium --devices 1,3,5
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/audit_locked_test_retry.py
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/run_scaling_suite.py --devices 1,3,5
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/run_locked_test_suite.py --scope scaling --devices 1,3,5
```

`--scope scaling`의 첫 sandbox 호출은 GPU preflight에서 exit 1이었고 평가를 시작하지 않았다. 동일 명령의 호스트 재실행은 18/18 PASS였다. 원본은 [preflight failure](../evidence/scaling_locked_test_sandbox_preflight_failure.log)와 [scaling scheduler](../logs/locked_test_scaling_scheduler.jsonl)에 있다.

## 집계와 그래프

```bash
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/summarize_results.py
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/plot_results.py
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/sanitize_scheduler_logs.py
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/sanitize_release_logs.py
```

## 기존 결과 및 release 검증

```bash
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/run_historical_validators.py
git worktree add --detach /tmp/c3-bf6c-completion-validation bf6c7c508635dec0be849aedaa3eac5d88ed220d
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/record_base_completion_validation.py --worktree /tmp/c3-bf6c-completion-validation
git worktree remove --force /tmp/c3-bf6c-completion-validation
../YOLO-Master/.venv/bin/python -m ruff check smoke/c3/augmentation/tools tests/test_c3_augmentation_policy.py
../YOLO-Master/.venv/bin/python -m ruff format --check smoke/c3/augmentation/tools tests/test_c3_augmentation_policy.py
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/run_release_checks.py
../YOLO-Master/.venv/bin/python smoke/c3/augmentation/tools/validate_augmentation_delivery.py
git diff --check
```

release gate의 첫 sandbox run은 CPU Gloo socket 권한으로 1건 실패했으며, 같은 명령을 호스트에서 재실행해 PASS했다. 실패와 성공 stdout/stderr는 각각 [failure bundle](../failures/release_checks_sandbox_socket_failure)과 [release checks](../evidence/release_checks)에 있다.
