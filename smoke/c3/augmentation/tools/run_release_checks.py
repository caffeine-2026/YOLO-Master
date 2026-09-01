#!/usr/bin/env python3
"""Run and preserve the augmentation-related pytest, Ruff, and diff gates."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ROOT = REPO_ROOT / "smoke" / "c3" / "augmentation"


def clean(value: str) -> str:
    return (
        value.replace(str(REPO_ROOT), "<repo>")
        .replace(str(Path(sys.executable).parents[1]), "<venv>")
        .replace(str(Path.home()), "<user-home>")
    )


def main() -> int:
    output = ROOT / "evidence" / "release_checks"
    if output.exists():
        raise FileExistsError("Refusing to overwrite release checks")
    output.mkdir(parents=True)
    from smoke.c3.completion.tools.run_release_checks import RELATED_TESTS

    tests = [*RELATED_TESTS, "tests/test_c3_augmentation_policy.py"]
    new_python = sorted(path.relative_to(REPO_ROOT).as_posix() for path in (ROOT / "tools").glob("*.py")) + [
        "tests/test_c3_augmentation_policy.py"
    ]
    commands = {
        "related_pytest": [sys.executable, "-m", "pytest", "-q", *tests],
        "ruff_new_code": [sys.executable, "-m", "ruff", "check", *new_python],
        "ruff_format": [sys.executable, "-m", "ruff", "format", "--check", *new_python],
        "ruff_touched_legacy_critical": [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "E9,F63,F7,F82",
            "ultralytics/cfg/__init__.py",
            "ultralytics/data/augment.py",
        ],
        "git_diff_check": ["git", "diff", "--check"],
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
                "command": clean(" ".join(command).replace(sys.executable, "python")),
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "returncode": completed.returncode,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "stdout": stdout.relative_to(REPO_ROOT).as_posix(),
                "stderr": stderr.relative_to(REPO_ROOT).as_posix(),
            }
        )
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(row["returncode"] == 0 for row in results) else "FAIL",
        "related_test_files": len(tests),
        "checks": results,
        "ruff_scope_note": "New augmentation code uses full Ruff rules; touched legacy files use non-negotiable syntax/name rules without changing Ruff configuration.",
    }
    (output / "release_checks.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": len(results)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
