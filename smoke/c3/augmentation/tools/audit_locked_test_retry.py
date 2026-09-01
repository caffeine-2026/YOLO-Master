#!/usr/bin/env python3
"""Audit the unavoidable locked-test retry without discarding either access record."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "smoke" / "c3" / "augmentation"


def main() -> int:
    output = ROOT / "evidence" / "locked_test_retry_equivalence.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite retry audit")
    rows = []
    for seed in (824, 825, 826):
        name = f"test_deeppcb_100_medium_seed{seed}.json"
        first_path = ROOT / "failures" / "locked_test_100_interrupted" / "evaluations" / name
        retry_path = ROOT / "evaluations" / name
        first = json.loads(first_path.read_text(encoding="utf-8"))
        retry = json.loads(retry_path.read_text(encoding="utf-8"))
        checks = {
            "overall_metrics_identical": first["overall"] == retry["overall"],
            "per_class_metrics_identical": first["per_class"] == retry["per_class"],
            "source_checkpoint_identical": first["source_checkpoint"]["sha256"] == retry["source_checkpoint"]["sha256"],
            "selection_frozen_for_both": first["selection_frozen_before_test"] is True
            and retry["selection_frozen_before_test"] is True,
            "test_not_used_for_selection": first["test_metrics_used_for_selection"] is False
            and retry["test_metrics_used_for_selection"] is False,
        }
        rows.append(
            {
                "seed": seed,
                "first": first_path.relative_to(REPO_ROOT).as_posix(),
                "retry": retry_path.relative_to(REPO_ROOT).as_posix(),
                "checks": checks,
            }
        )
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(all(row["checks"].values()) for row in rows) else "FAIL",
        "physical_locked_test_accesses_per_affected_checkpoint": 2,
        "reason": "The first evaluation children completed after their orchestration parent was interrupted, but full stdout/timing was incomplete; the delivery rerun was required for complete evidence.",
        "selection_or_tuning_after_first_test": False,
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "affected": len(rows)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
