#!/usr/bin/env python3
"""Apply the preregistered validation-only rule to the completed efficiency search."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smoke.c3.completion.tools.run_efficiency import CANDIDATES, DATASETS, ROOT, TRAINABLE_LIMIT
from smoke.c3.p1.scripts.run_p1 import read_learning_curve


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def full_validation(dataset: str) -> dict:
    curve = REPO_ROOT / "smoke" / "c3" / "p1" / "logs" / f"{dataset}_full_seed824_e100" / "learning_curve.csv"
    rows, finite = read_learning_curve(curve)
    values = [float(row["metrics/mAP50-95(B)"]) for row in rows]
    if len(rows) != 100 or not finite:
        raise ValueError(f"Invalid Full-SFT validation curve: {curve.relative_to(REPO_ROOT)}")
    return {
        "path": curve.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256(curve),
        "best_map50_95": max(values),
    }


def main() -> int:
    output = ROOT / "results" / "efficiency_selection.json"
    csv_output = ROOT / "results" / "efficiency_search.csv"
    if output.exists() or csv_output.exists():
        raise FileExistsError("Refusing to overwrite the locked efficiency selection")
    baselines = {dataset: full_validation(dataset) for dataset in DATASETS}
    rows = []
    for candidate in CANDIDATES:
        candidate_runs = []
        for dataset in DATASETS:
            run_id = f"search_{dataset}_100_{candidate}_seed824"
            path = ROOT / "logs" / "search" / run_id / "metrics.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("test") is not None or (path.parent / "test_metrics.json").exists():
                raise RuntimeError(f"Search contaminated by test access: {run_id}")
            candidate_runs.append((path, payload))
        trainable = [int(payload["parameters"]["trainable_parameters"]) for _, payload in candidate_runs]
        validation = [float(payload["validation"]["best_map50_95"]) for _, payload in candidate_runs]
        row = {
            "candidate": candidate,
            "eligible": all(
                payload["status"] == "PASS" and value <= TRAINABLE_LIMIT
                for value, (_, payload) in zip(trainable, candidate_runs)
            ),
            "mean_best_validation_map50_95": sum(validation) / len(validation),
            "mean_trainable_parameters": sum(trainable) / len(trainable),
            "mean_peak_gpu_memory_mib": sum(
                float(payload["resources"]["peak_gpu_memory_mib"]) for _, payload in candidate_runs
            )
            / len(candidate_runs),
            "mean_training_seconds": sum(float(payload["timing"]["training_seconds"]) for _, payload in candidate_runs)
            / len(candidate_runs),
            "dataset_results": {
                dataset: {
                    "best_validation_map50_95": validation[index],
                    "full_sft_best_validation_map50_95": baselines[dataset]["best_map50_95"],
                    "validation_delta": validation[index] - baselines[dataset]["best_map50_95"],
                    "validation_retention": (
                        validation[index] / baselines[dataset]["best_map50_95"]
                        if baselines[dataset]["best_map50_95"]
                        else None
                    ),
                    "metrics_path": candidate_runs[index][0].relative_to(REPO_ROOT).as_posix(),
                    "metrics_sha256": sha256(candidate_runs[index][0]),
                }
                for index, dataset in enumerate(DATASETS)
            },
        }
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("No candidate passed the preregistered eligibility gate")
    best_primary = max(row["mean_best_validation_map50_95"] for row in eligible)
    contenders = [row for row in eligible if best_primary - row["mean_best_validation_map50_95"] <= 0.001]
    selected = min(
        contenders,
        key=lambda row: (
            row["mean_trainable_parameters"],
            row["mean_peak_gpu_memory_mib"],
            row["mean_training_seconds"],
            row["candidate"],
        ),
    )
    payload = {
        "schema_version": 1,
        "status": "SELECTED",
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "command": "../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/select_efficiency.py",
        "protocol": "smoke/c3/completion/config/efficiency_search_protocol.yaml",
        "metric_split": "validation",
        "test_access_before_selection": False,
        "trainable_limit": TRAINABLE_LIMIT,
        "tie_threshold_map50_95": 0.001,
        "full_sft_validation_baselines": baselines,
        "selected_candidate": selected["candidate"],
        "selected_configuration": CANDIDATES[selected["candidate"]],
        "best_primary_value": best_primary,
        "tie_contenders": [row["candidate"] for row in contenders],
        "candidate_results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "candidate",
        "eligible",
        "mean_best_validation_map50_95",
        "mean_trainable_parameters",
        "mean_peak_gpu_memory_mib",
        "mean_training_seconds",
    ]
    with csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(json.dumps({"selected_candidate": selected["candidate"], "test_access_before_selection": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
