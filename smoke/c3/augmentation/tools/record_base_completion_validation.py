#!/usr/bin/env python3
"""Run the immutable completion gate in a detached base worktree and record its evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_COMMIT = "bf6c7c508635dec0be849aedaa3eac5d88ed220d"
OUTPUT = REPO_ROOT / "smoke/c3/augmentation/evidence/base_completion_validation"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    args = parser.parse_args()
    worktree = args.worktree.resolve()
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT.relative_to(REPO_ROOT)}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=worktree, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != BASE_COMMIT:
        raise RuntimeError(f"Expected detached base {BASE_COMMIT}, got {head}")
    command = [sys.executable, "smoke/c3/completion/tools/validate_completion.py"]
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=worktree, capture_output=True, text=True, check=False)
    finished = datetime.now(timezone.utc).isoformat()
    source = worktree / "smoke/c3/completion/evidence/integration_validation.json"
    OUTPUT.mkdir(parents=True)
    venv_root = str(Path(sys.executable).parents[1])
    clean = lambda value: value.replace(str(worktree), "<base-worktree>").replace(venv_root, "<venv>")
    (OUTPUT / "stdout.log").write_text(clean(completed.stdout), encoding="utf-8")
    (OUTPUT / "stderr.log").write_text(clean(completed.stderr), encoding="utf-8")
    validation = json.loads(source.read_text(encoding="utf-8")) if source.is_file() else None
    payload = {
        "schema_version": 1,
        "status": "PASS" if completed.returncode == 0 and validation and validation.get("status") == "PASS" else "FAIL",
        "base_commit": BASE_COMMIT,
        "detached_head_verified": head == BASE_COMMIT,
        "command": "<venv>/bin/python smoke/c3/completion/tools/validate_completion.py",
        "started_utc": started,
        "finished_utc": finished,
        "returncode": completed.returncode,
        "source_validation_sha256": sha256(source) if source.is_file() else None,
        "validation": validation,
        "artifact_links": {
            "completion_artifacts": (worktree / "smoke/c3/completion/artifacts").is_symlink(),
            "initial_weight": (worktree / "yolo11n.pt").is_symlink(),
        },
    }
    (OUTPUT / "validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "returncode": completed.returncode, "base": head}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
