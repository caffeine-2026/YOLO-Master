# Executed commands

이 파일은 이번 감사에서 실제 실행한 핵심 명령과 결과를 기록한다. 79개 원본 training/evaluation command 전문은 `evidence/raw_command_manifest.json` 및 각 run의 `command.txt`에 있다.

## Host/GPU preflight

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pstate --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
.venv/bin/python smoke/c3/final/tools/capture_gpu_preflight.py
```

결과: 8 × RTX 4090, GPU 0 occupied, GPU 1–7 available, CUDA available. 다른 사용자의 프로세스는 중단하지 않았다.

## Evidence regeneration and validators

```bash
.venv/bin/python smoke/c3/p1/scripts/analyze_multiseed.py
.venv/bin/python smoke/c3/p2/tools/summarize_multiseed.py
.venv/bin/python smoke/c3/p0/tools/validate_delivery.py
.venv/bin/python smoke/c3/p1/scripts/validate_p1.py --epochs 100 --output smoke/c3/p1/evidence/revalidation_20260831.json
.venv/bin/python smoke/c3/p2/tools/validate_p2.py
.venv/bin/python smoke/c3/p2/tools/validate_p2_final.py
PYTHONUNBUFFERED=1 .venv/bin/python smoke/c3/final/tools/validate_research_delivery.py
```

결과: P0 PASS, P1 PASS, archived P2 seed-824 gate PASS, final P2 PASS, integrated delivery PASS.

main 기반 clean worktree에는 Git ignored dataset과 P0 runtime artifact가 없었으므로, 원본을 변경하지 않는 hard-link 작업 사본을 만든 뒤 P0 validator를 재실행했다. 첫 missing-dataset 실패와 symbolic-link 보안 거부는 [FAILURE_REPAIR_RERUN.md](FAILURE_REPAIR_RERUN.md)에 보존했다.

## Checkpoint audit

통합 validator가 72개 `best.pt`에 대해 `torch.load(..., map_location="cpu", weights_only=False)`를 실행하고 model parameter count, seed, epoch metadata를 metrics/resolved config와 대조했다. 72/72 PASS.

## Tests and release gates

```bash
.venv/bin/pytest tests/test_p0_system_gates.py::test_cpu_gloo_two_rank_routed_continuous_training -q
.venv/bin/pytest tests/test_p0_system_gates.py::test_planner_adapter_full_lifecycle tests/test_vpeft_lora_e2e.py tests/test_lora_fallback_effective_config.py -q
.venv/bin/pytest tests/test_vpeft.py tests/test_vpeft_lora_e2e.py tests/test_vpeft_rl_rank_allocator.py tests/test_placement_plan_schema.py tests/test_molora_vpeft_integration.py tests/test_p0_system_gates.py tests/test_yolo_peft_paper_anchor.py tests/test_lora_fallback_effective_config.py -q
.venv/bin/pytest tests/test_lora_fallback_effective_config.py tests/test_lora_moe_ddp_control_paths.py tests/test_lora_selective_ema_lifecycle.py tests/test_lora_training_strategy.py tests/test_moe_aware_peft.py tests/test_molora.py tests/test_molora_backend_roundtrip.py tests/test_molora_dtype.py tests/test_molora_merge_publishability.py tests/test_molora_merge_semantics.py tests/test_molora_routing_aware_merge.py tests/test_molora_sparse_dispatch.py tests/test_molora_supplementary.py tests/test_molora_vpeft_integration.py tests/test_p0_system_gates.py tests/test_peft_adapters.py tests/test_peft_optimizer_policy.py tests/test_placement_plan_schema.py tests/test_vpeft.py tests/test_vpeft_lora_e2e.py tests/test_vpeft_rl_rank_allocator.py tests/test_yolo_peft_paper_anchor.py -q
```

결과: host Gloo 1/1 PASS, 집중 재검증 17/17 PASS, 핵심 suite 65/65 PASS, 관련 LoRA/MoLoRA/PEFT/Planner pytest 전체 295 passed / 7 skipped / 2 warnings. skip은 조건부 backend/device 케이스이고 실패는 없다.

최종 release gate:

```bash
.venv/bin/ruff check smoke/c3/p1/scripts/validate_p1.py smoke/c3/p2/tools/validate_p2.py smoke/c3/p2/tools/validate_p2_final.py smoke/c3/final/tools
.venv/bin/ruff check ultralytics/utils/lora/fallback.py --ignore I001,PLR0402,FA100,RUF022
.venv/bin/ruff format --check smoke/c3/p1/scripts/validate_p1.py smoke/c3/p2/tools/validate_p2.py smoke/c3/p2/tools/validate_p2_final.py smoke/c3/final/tools
git diff --check
```

결과: 모두 PASS. `fallback.py`의 import ordering, module alias, future annotation, `__all__` ordering 네 lint 범주는 이번 동작 수정 전부터 존재한 파일 수준 부채라 명시적으로 제외했다. 최종 main 기준 worktree에서 JSON/YAML 934개 파싱, Markdown local link 72개, publishable file 1,430개 privacy 검사와 14개 그래프 provenance 검사는 통합 validator에서 PASS했다.
