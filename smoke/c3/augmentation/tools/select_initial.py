#!/usr/bin/env python3
"""Select one preregistered non-baseline augmentation per dataset using validation only."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "smoke" / "c3" / "augmentation"
DATASETS = ("neu", "deeppcb")
POLICIES = ("baseline", "mild", "medium", "strong")


def main() -> int:
    output = ROOT / "results" / "initial_selection.json"
    csv_output = ROOT / "results" / "initial_search.csv"
    if output.exists() or csv_output.exists():
        raise FileExistsError("Refusing to overwrite initial selection evidence")
    rows = []
    for dataset in DATASETS:
        for policy in POLICIES:
            run_id = f"search_{dataset}_100_{policy}_seed824_e100"
            metrics_path = ROOT / "logs" / "search" / run_id / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("status") != "PASS" or metrics.get("test") is not None:
                raise ValueError(f"Invalid validation-only search run: {run_id}")
            if not metrics["checks"].get("test_not_accessed"):
                raise ValueError(f"Test access guard failed: {run_id}")
            rows.append(
                {
                    "dataset": dataset,
                    "policy": policy,
                    "strength": metrics["policy"]["strength"],
                    "run_id": run_id,
                    "best_validation_map50_95": metrics["validation"]["best_map50_95"],
                    "best_validation_map50": metrics["validation"]["best_map50"],
                    "best_epoch": metrics["validation"]["best_epoch"],
                    "trainable_parameters": metrics["parameters"]["trainable_parameters"],
                    "peak_gpu_memory_mib": metrics["resources"]["peak_gpu_memory_mib"],
                    "training_seconds": metrics["timing"]["training_seconds"],
                }
            )
    selections = {}
    for dataset in DATASETS:
        candidates = [row for row in rows if row["dataset"] == dataset and row["policy"] != "baseline"]
        top = min(
            candidates,
            key=lambda row: (
                -float(row["best_validation_map50_95"]),
                -float(row["best_validation_map50"]),
                int(row["strength"]),
            ),
        )
        baseline = next(row for row in rows if row["dataset"] == dataset and row["policy"] == "baseline")
        selections[dataset] = {
            "top_augmentation": top["policy"],
            "top_run_id": top["run_id"],
            "top_validation_map50_95": top["best_validation_map50_95"],
            "baseline_validation_map50_95": baseline["best_validation_map50_95"],
            "initial_delta": top["best_validation_map50_95"] - baseline["best_validation_map50_95"],
        }
    with csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "status": "INITIAL_SELECTION_LOCKED",
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "metric_split": "validation",
        "test_access_before_selection": False,
        "selection_rule": "highest primary, then secondary, then lower strength among mild/medium/strong",
        "datasets": selections,
        "source_csv": csv_output.relative_to(REPO_ROOT).as_posix(),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
