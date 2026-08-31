#!/usr/bin/env python3
"""Capture the exact dirty-worktree code and configuration used by C3 training."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "smoke" / "c3" / "completion" / "evidence" / "provenance"
TRACKED_INPUTS = (
    "requirements.txt",
    "pyproject.toml",
    "ultralytics/cfg/__init__.py",
    "ultralytics/cfg/default.yaml",
    # ``yolo11n.pt`` resolves its scale from the canonical family YAML.
    "ultralytics/cfg/models/11/yolo11.yaml",
    "ultralytics/utils/lora/api.py",
    "ultralytics/utils/lora/config.py",
    "ultralytics/utils/lora/fallback.py",
    "ultralytics/utils/lora/io.py",
    "ultralytics/utils/lora/planner.py",
    "ultralytics/vpeft/solver.py",
    "smoke/c3/p1/scripts/run_p1.py",
    "smoke/c3/completion/tools/run_efficiency.py",
    "smoke/c3/completion/config/efficiency_search_protocol.yaml",
    "smoke/c3/p2/config/runs/neu/vpeft.yaml",
    "smoke/c3/p2/config/runs/deeppcb/vpeft.yaml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def main() -> int:
    json_path = OUTPUT_DIR / "training_code_snapshot.json"
    patch_path = OUTPUT_DIR / "training_code.patch"
    if json_path.exists() or patch_path.exists():
        raise FileExistsError("Refusing to overwrite immutable training provenance")
    paths = [REPO_ROOT / relative for relative in TRACKED_INPUTS]
    missing = [path.relative_to(REPO_ROOT).as_posix() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing training inputs: {missing}")
    diff = git("diff", "--", *TRACKED_INPUTS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(diff + ("\n" if diff else ""), encoding="utf-8")
    weight_path = REPO_ROOT / "yolo11n.pt"
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "command": "../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/capture_training_provenance.py",
        "git_branch_at_launch": "codex/c3-lovo-mipr-planner-efficiency-20260831",
        "git_head_at_launch": git("rev-parse", "HEAD"),
        "dirty_training_diff_path": patch_path.relative_to(REPO_ROOT).as_posix(),
        "dirty_training_diff_sha256": sha256(patch_path),
        "tracked_inputs": {
            path.relative_to(REPO_ROOT).as_posix(): {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths
        },
        "initial_weight": {
            "path": weight_path.relative_to(REPO_ROOT).as_posix(),
            "size_bytes": weight_path.stat().st_size,
            "sha256": sha256(weight_path),
        },
        "note": (
            "The experiments ran before the feature commit existed. This source-hash snapshot and patch identify "
            "the executed dirty-worktree state; the final commit must contain the same tracked-input hashes."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"snapshot": json_path.relative_to(REPO_ROOT).as_posix(), "files": len(paths)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
