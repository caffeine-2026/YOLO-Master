#!/usr/bin/env python3
"""Run and preserve the existing P0/P1/P2 and integrated C3 validators."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = REPO_ROOT / "smoke" / "c3" / "completion" / "evidence" / "validation"


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
        raise FileExistsError("Refusing to overwrite validator evidence")
    commands = {
        "p0": [
            sys.executable,
            "smoke/c3/p0/tools/validate_delivery.py",
            "--output",
            "smoke/c3/p0/evidence/completion_revalidation_20260831.json",
        ],
        "p1": [
            sys.executable,
            "smoke/c3/p1/scripts/validate_p1.py",
            "--epochs",
            "100",
            "--output",
            "smoke/c3/p1/evidence/completion_revalidation_20260831.json",
        ],
        "p2_seed824": [sys.executable, "smoke/c3/p2/tools/validate_p2.py"],
        "p2_final": [sys.executable, "smoke/c3/p2/tools/validate_p2_final.py"],
        "existing_integrated": [
            sys.executable,
            "smoke/c3/final/tools/validate_research_delivery.py",
            "--output",
            "smoke/c3/final/evidence/completion_revalidation_20260831.json",
        ],
        "new_completion": [sys.executable, "smoke/c3/completion/tools/validate_completion.py"],
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
                "command": " ".join(["python", *command[1:]]),
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
        "command": "../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/run_validators.py",
        "status": "PASS" if all(row["returncode"] == 0 for row in results) else "FAIL",
        "validators": results,
    }
    (output_dir / "validation_suite.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "validators": len(results)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
