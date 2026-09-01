#!/usr/bin/env python3
"""Remove absolute workspace/user paths from scheduler evidence and audit the change."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "smoke" / "c3" / "augmentation"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    audit_path = ROOT / "evidence" / "scheduler_privacy_sanitization.json"
    if audit_path.exists():
        raise FileExistsError("Refusing to overwrite sanitization audit")
    paths = sorted({*(ROOT / "logs").glob("*_scheduler.jsonl"), *(ROOT / "failures").rglob("*scheduler*.jsonl")})
    rows = []
    replacements = ((str(REPO_ROOT), "<repo>"), (str(Path.home()), "<user-home>"))
    for path in paths:
        before = path.read_bytes()
        text = before.decode("utf-8")
        counts = {}
        for source, target in replacements:
            counts[source] = text.count(source)
            text = text.replace(source, target)
        after = text.encode("utf-8")
        path.write_bytes(after)
        rows.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "before_sha256": sha256_bytes(before),
                "after_sha256": sha256_bytes(after),
                "replacement_counts": {
                    "repository_root": counts[str(REPO_ROOT)],
                    "user_home": counts[str(Path.home())],
                },
                "semantic_change": "none; path redaction only",
            }
        )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "reason": "privacy scan requires logs not to expose absolute user paths",
                "scheduler_logs": rows,
                "training_stdout_stderr_modified": False,
                "metrics_modified": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "files": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
