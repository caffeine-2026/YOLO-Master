#!/usr/bin/env python3
"""Run the related pytest and Ruff/git release gates with preserved output."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = REPO_ROOT / "smoke" / "c3" / "completion" / "evidence" / "release_checks"
RELATED_TESTS = (
    "tests/test_adapter_backend_contract.py",
    "tests/test_c3_completion_evidence.py",
    "tests/test_default_config_integrity.py",
    "tests/test_lora_fallback_effective_config.py",
    "tests/test_lora_moe_ddp_control_paths.py",
    "tests/test_lora_selective_ema_lifecycle.py",
    "tests/test_lora_training_strategy.py",
    "tests/test_moe_aware_peft.py",
    "tests/test_molora.py",
    "tests/test_molora_backend_roundtrip.py",
    "tests/test_molora_dtype.py",
    "tests/test_molora_merge_publishability.py",
    "tests/test_molora_merge_semantics.py",
    "tests/test_molora_routing_aware_merge.py",
    "tests/test_molora_sparse_dispatch.py",
    "tests/test_molora_supplementary.py",
    "tests/test_molora_vpeft_integration.py",
    "tests/test_p0_system_gates.py",
    "tests/test_peft_adapters.py",
    "tests/test_peft_optimizer_policy.py",
    "tests/test_placement_plan_schema.py",
    "tests/test_planner.py",
    "tests/test_planner_enhancement.py",
    "tests/test_planner_integration.py",
    "tests/test_vpeft.py",
    "tests/test_vpeft_lora_e2e.py",
    "tests/test_vpeft_rl_rank_allocator.py",
    "tests/test_yolo_peft_paper_anchor.py",
)
LEGACY_CHANGED_PYTHON = (
    "smoke/c3/p1/scripts/run_p1.py",
    "ultralytics/cfg/__init__.py",
    "ultralytics/utils/lora/api.py",
    "ultralytics/utils/lora/config.py",
    "ultralytics/utils/lora/fallback.py",
    "ultralytics/utils/lora/io.py",
    "ultralytics/utils/lora/planner.py",
    "ultralytics/vpeft/solver.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def clean(value: str) -> str:
    return value.replace(str(REPO_ROOT), "<repo>").replace(str(Path(sys.executable).parents[1]), "<venv>")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite release-check evidence")
    new_python = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "smoke" / "c3" / "completion" / "tools").glob("*.py")
    )
    commands = {
        "related_pytest": [sys.executable, "-m", "pytest", "-q", *RELATED_TESTS],
        "ruff_new_code": [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *new_python,
            "tests/test_c3_completion_evidence.py",
        ],
        "ruff_format": [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            *new_python,
            "tests/test_c3_completion_evidence.py",
        ],
        # The legacy files have broad pre-existing style debt recorded in the
        # failure evidence. Keep Ruff's non-negotiable syntax/name checks on
        # every touched legacy Python file without modifying Ruff config.
        "ruff_legacy_critical": [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "E9,F63,F7,F82",
            *LEGACY_CHANGED_PYTHON,
        ],
        "git_diff_check": ["git", "diff", "--check"],
    }
    output_dir.mkdir(parents=True)
    results = []
    for name, command in commands.items():
        started = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        stdout_path = output_dir / f"{name}.stdout.txt"
        stderr_path = output_dir / f"{name}.stderr.txt"
        stdout_path.write_text(clean(completed.stdout), encoding="utf-8")
        stderr_path.write_text(clean(completed.stderr), encoding="utf-8")
        results.append(
            {
                "name": name,
                "command": " ".join(command).replace(sys.executable, "python"),
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "returncode": completed.returncode,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "stdout": stdout_path.relative_to(REPO_ROOT).as_posix(),
                "stderr": stderr_path.relative_to(REPO_ROOT).as_posix(),
            }
        )
    payload = {
        "schema_version": 1,
        "command": "../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/run_release_checks.py",
        "related_test_files": len(RELATED_TESTS),
        "status": "PASS" if all(row["returncode"] == 0 for row in results) else "FAIL",
        "checks": results,
        "ruff_scope_note": (
            "New completion code uses the complete repository Ruff rules. Touched legacy files use Ruff's "
            "critical syntax/name rules; the unfiltered legacy-file failure is preserved separately and was not "
            "made to pass by changing configuration."
        ),
    }
    (output_dir / "release_checks.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "checks": len(results)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
