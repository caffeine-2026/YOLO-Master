#!/usr/bin/env python3
"""Rerun historical P0/P1/P2/integrated gates without overwriting their JSON evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "smoke" / "c3" / "augmentation"
REDIRECT = ROOT / "tools" / "redirect_validator_output.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value: str) -> str:
    return value.replace(str(REPO_ROOT), "<repo>").replace(str(Path(sys.executable).parents[1]), "<venv>")


def main() -> int:
    output = ROOT / "evidence" / "historical_validators"
    if output.exists():
        raise FileExistsError("Refusing to overwrite historical validator rerun")
    output.mkdir(parents=True)
    protected = [
        REPO_ROOT / "smoke/c3/p0/evidence/completion_revalidation_20260831.json",
        REPO_ROOT / "smoke/c3/p1/evidence/completion_revalidation_20260831.json",
        REPO_ROOT / "smoke/c3/p2/evidence/p2_seed824_validation.json",
        REPO_ROOT / "smoke/c3/p2/evidence/p2_final_validation.json",
        REPO_ROOT / "smoke/c3/final/evidence/raw_command_manifest.json",
        REPO_ROOT / "smoke/c3/final/evidence/completion_revalidation_20260831.json",
        REPO_ROOT / "smoke/c3/completion/evidence/integration_validation.json",
    ]
    before = {path.relative_to(REPO_ROOT).as_posix(): sha256(path) for path in protected}
    commands = {
        "p0": [
            sys.executable,
            str(REDIRECT),
            "--module",
            "smoke.c3.p0.tools.validate_delivery",
            "--source",
            "smoke/c3/p0/evidence/augmentation_revalidation_20260831.json",
            "--destination",
            "smoke/c3/augmentation/evidence/historical_validators/p0_validation.json",
            "--",
            "--output",
            "smoke/c3/p0/evidence/augmentation_revalidation_20260831.json",
        ],
        "p1": [
            sys.executable,
            str(REDIRECT),
            "--module",
            "smoke.c3.p1.scripts.validate_p1",
            "--source",
            "smoke/c3/p1/evidence/augmentation_revalidation_20260831.json",
            "--destination",
            "smoke/c3/augmentation/evidence/historical_validators/p1_validation.json",
            "--",
            "--epochs",
            "100",
            "--output",
            "smoke/c3/p1/evidence/augmentation_revalidation_20260831.json",
        ],
        "p2_seed824": [
            sys.executable,
            str(REDIRECT),
            "--module",
            "smoke.c3.p2.tools.validate_p2",
            "--source",
            "smoke/c3/p2/evidence/p2_seed824_validation.json",
            "--destination",
            "smoke/c3/augmentation/evidence/historical_validators/p2_seed824_validation.json",
        ],
        "p2_final": [
            sys.executable,
            str(REDIRECT),
            "--module",
            "smoke.c3.p2.tools.validate_p2_final",
            "--source",
            "smoke/c3/p2/evidence/p2_final_validation.json",
            "--destination",
            "smoke/c3/augmentation/evidence/historical_validators/p2_final_validation.json",
        ],
        "integrated_research": [
            sys.executable,
            str(REDIRECT),
            "--module",
            "smoke.c3.final.tools.validate_research_delivery",
            "--source",
            "smoke/c3/final/evidence/raw_command_manifest.json",
            "--destination",
            "smoke/c3/augmentation/evidence/historical_validators/raw_command_manifest.json",
            "--",
            "--output",
            "smoke/c3/final/evidence/augmentation_revalidation_20260831.json",
        ],
    }
    results = []
    for name, command in commands.items():
        started = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        stdout = output / f"{name}.stdout.log"
        stderr = output / f"{name}.stderr.log"
        stdout.write_text(clean(completed.stdout), encoding="utf-8")
        stderr.write_text(clean(completed.stderr), encoding="utf-8")
        results.append(
            {
                "name": name,
                "command": clean(" ".join(["python", *command[1:]])),
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "returncode": completed.returncode,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "stdout": stdout.relative_to(REPO_ROOT).as_posix(),
                "stderr": stderr.relative_to(REPO_ROOT).as_posix(),
            }
        )
    after = {path.relative_to(REPO_ROOT).as_posix(): sha256(path) for path in protected}
    unchanged = {path: before[path] == after[path] for path in before}
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(row["returncode"] == 0 for row in results) and all(unchanged.values()) else "FAIL",
        "validators": results,
        "protected_existing_outputs": {path: {"sha256": before[path], "unchanged": unchanged[path]} for path in before},
    }
    (output / "validation_suite.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": payload["status"], "validators": len(results), "protected_unchanged": all(unchanged.values())}
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
