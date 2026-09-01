#!/usr/bin/env python3
"""Redact absolute user paths from preserved release-test stdout and audit hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "smoke/c3/augmentation"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    audit = ROOT / "evidence/release_log_privacy_sanitization.json"
    if audit.exists():
        raise FileExistsError(f"Refusing to overwrite {audit.relative_to(REPO_ROOT)}")
    paths = [
        ROOT / "evidence/release_checks/related_pytest.stdout.log",
        ROOT / "failures/release_checks_sandbox_socket_failure/related_pytest.stdout.log",
    ]
    rows = []
    for path in paths:
        before = path.read_bytes()
        source = str(Path.home())
        text = before.decode("utf-8")
        count = text.count(source)
        after = text.replace(source, "<user-home>").encode("utf-8")
        path.write_bytes(after)
        rows.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "before_sha256": digest(before),
                "after_sha256": digest(after),
                "replacement_count": count,
                "semantic_change": "none; absolute user path redaction only",
            }
        )
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(row["replacement_count"] > 0 for row in rows) else "FAIL",
        "files": rows,
        "test_outcomes_modified": False,
    }
    audit.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "files": len(rows)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
