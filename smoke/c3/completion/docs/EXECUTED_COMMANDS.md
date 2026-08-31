# C3 executed commands

아래 명령은 integration worktree의 저장소 root에서 실제 실행했다. 가상환경은 인접한 기존 `../YOLO-Master/.venv`를 재사용했다. 문서에는 토큰이나 사용자별 절대 경로를 복제하지 않기 위해 저장소 기준 상대 경로로 표기한다.

## GPU와 환경 확인

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
../YOLO-Master/.venv/bin/python -m pip install 'ortools==9.15.6755'
../YOLO-Master/.venv/bin/python -m pip show ortools protobuf numpy pandas
../YOLO-Master/.venv/bin/python -m pip check
```

GPU 4에는 다른 사용자의 프로세스가 있어 제외했다. 어떤 프로세스도 종료하지 않았다. 탐색에는 preflight에서 비어 있던 GPU 1/5/7, 최종 matrix에는 GPU 0/1/2/3/5/6/7을 사용했다.

## Solver, Planner, 탐색과 학습

```bash
../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/compare_solvers.py
../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/generate_planner_branch_evidence.py
../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/run_search_suite.py --devices 1,5,7
../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/select_efficiency.py
../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/run_final_matrix.py --devices 0,1,2,3,5,6,7
../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/summarize_completion.py
../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/calibrate_lovo.py
../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/capture_training_provenance.py
```

각 학습 subprocess의 완전한 argv는 search/final run별 `command.txt`와 scheduler JSONL에 있고, resolved config는 각 run의 `resolved_config.yaml`에 있다. stdout/stderr, epoch metrics, CUDA memory sample, timing과 checkpoint manifest도 같은 run 디렉터리에 보존했다.

## 테스트와 validator

```bash
../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/run_release_checks.py
../YOLO-Master/.venv/bin/python -m pytest -q tests/test_p0_system_gates.py::test_cpu_gloo_two_rank_routed_continuous_training
../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/run_validators.py
../YOLO-Master/.venv/bin/python smoke/c3/final/tools/validate_research_delivery.py --output smoke/c3/final/evidence/completion_revalidation_20260831.json
../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/validate_completion.py
../YOLO-Master/.venv/bin/ruff check smoke/c3/completion/tools tests/test_c3_completion_evidence.py
../YOLO-Master/.venv/bin/ruff format --check smoke/c3/completion/tools tests/test_c3_completion_evidence.py
../YOLO-Master/.venv/bin/ruff check --select E9,F63,F7,F82 ultralytics/vpeft/solver.py ultralytics/utils/lora ultralytics/cfg/__init__.py smoke/c3/p1/scripts/run_p1.py tests/test_lora_fallback_effective_config.py tests/test_planner.py tests/test_planner_enhancement.py tests/test_vpeft_lora_e2e.py
git diff --check
```

첫 validator suite는 pytest warning에 포함된 절대 가상환경 경로를 hygiene finding으로 정확히 탐지해 실패했다. 출력의 의미는 바꾸지 않고 경로만 `<venv>`로 비식별화한 뒤 전체 suite를 다시 실행했다. 최종 기록은 `evidence/validation_rerun/validation_suite.json`이며 P0, P1, P2 seed-824, P2 final, 기존 통합 validator, 새 completion validator가 모두 PASS다.

## GitHub 상태 확인

```bash
gh auth status
```

이 명령은 설정된 GitHub CLI token이 유효하지 않다고 반환했다. 따라서 Issue #2에는 완료 댓글을 게시하지 않았으며, 게시용 중국어 본문은 `GITHUB_ISSUE_2_FOLLOWUP_ZH.md`에 보존했다.
