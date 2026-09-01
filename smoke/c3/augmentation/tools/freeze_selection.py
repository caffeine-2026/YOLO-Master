#!/usr/bin/env python3
"""Freeze augmentation policies from three validation seeds and no test evidence."""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "smoke" / "c3" / "augmentation"
DATASETS = ("neu", "deeppcb")
SEEDS = (824, 825, 826)
T_CRITICAL_95_DF2 = 4.30265273


def load_metrics(dataset: str, policy: str, seed: int) -> dict:
    phase = "search" if seed == 824 else "confirm"
    run_id = f"{phase}_{dataset}_100_{policy}_seed{seed}_e100"
    path = ROOT / "logs" / phase / run_id / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError(f"Run did not pass: {run_id}")
    if payload.get("test") is not None or not payload["checks"].get("test_not_accessed"):
        raise ValueError(f"Test evidence was accessed before policy freeze: {run_id}")
    return payload


def summary(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    half = T_CRITICAL_95_DF2 * std / math.sqrt(len(values))
    return {"mean": mean, "sample_std": std, "ci95_lower": mean - half, "ci95_upper": mean + half}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    results_dir = ROOT / "results"
    outputs = [
        results_dir / "confirmation_runs.csv",
        results_dir / "confirmation_summary.csv",
        results_dir / "paired_validation_statistics.csv",
        results_dir / "frozen_selection.json",
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite frozen selection evidence")
    initial = json.loads((results_dir / "initial_selection.json").read_text(encoding="utf-8"))
    if initial.get("test_access_before_selection") is not False:
        raise ValueError("Initial selection lacks the no-test guard")

    run_rows: list[dict] = []
    summary_rows: list[dict] = []
    paired_rows: list[dict] = []
    frozen: dict[str, dict] = {}
    for dataset in DATASETS:
        augmented = initial["datasets"][dataset]["top_augmentation"]
        by_policy: dict[str, list[dict]] = {}
        for policy in ("baseline", augmented):
            payloads = [load_metrics(dataset, policy, seed) for seed in SEEDS]
            by_policy[policy] = payloads
            for payload in payloads:
                run_rows.append(
                    {
                        "dataset": dataset,
                        "policy": policy,
                        "seed": payload["seed"],
                        "run_id": payload["run_id"],
                        "best_validation_map50_95": payload["validation"]["best_map50_95"],
                        "best_validation_map50": payload["validation"]["best_map50"],
                        "best_validation_precision": payload["validation"]["best_precision"],
                        "best_validation_recall": payload["validation"]["best_recall"],
                        "best_epoch": payload["validation"]["best_epoch"],
                        "trainable_parameters": payload["parameters"]["trainable_parameters"],
                        "total_parameters": payload["parameters"]["total_parameters"],
                        "peak_gpu_memory_mib": payload["resources"]["peak_gpu_memory_mib"],
                        "training_seconds": payload["timing"]["training_seconds"],
                        "gpu_hours": payload["timing"]["gpu_hours"],
                        "test_access": False,
                    }
                )
            for metric, source in (
                ("map50_95", ("validation", "best_map50_95")),
                ("map50", ("validation", "best_map50")),
                ("precision", ("validation", "best_precision")),
                ("recall", ("validation", "best_recall")),
                ("peak_gpu_memory_mib", ("resources", "peak_gpu_memory_mib")),
                ("training_seconds", ("timing", "training_seconds")),
                ("gpu_hours", ("timing", "gpu_hours")),
            ):
                values = [float(payload[source[0]][source[1]]) for payload in payloads]
                summary_rows.append({"dataset": dataset, "policy": policy, "metric": metric, "n": 3, **summary(values)})

        base_values = [float(p["validation"]["best_map50_95"]) for p in by_policy["baseline"]]
        aug_values = [float(p["validation"]["best_map50_95"]) for p in by_policy[augmented]]
        deltas = [aug - base for aug, base in zip(aug_values, base_values, strict=True)]
        paired = summary(deltas)
        scaling_trigger = paired["mean"] >= 0.005 and paired["ci95_lower"] > 0.0
        frozen_policy = augmented if paired["mean"] > 0.0 else "baseline"
        paired_rows.extend(
            {
                "dataset": dataset,
                "seed": seed,
                "baseline_policy": "baseline",
                "augmentation_policy": augmented,
                "baseline_validation_map50_95": base,
                "augmentation_validation_map50_95": aug,
                "paired_delta_map50_95": delta,
            }
            for seed, base, aug, delta in zip(SEEDS, base_values, aug_values, deltas, strict=True)
        )
        frozen[dataset] = {
            "candidate_policy": augmented,
            "frozen_policy": frozen_policy,
            "paired_delta_map50_95": {"n": 3, "values": deltas, **paired},
            "selection_rule_passed": paired["mean"] > 0.0,
            "scaling_trigger_passed": scaling_trigger,
            "scaling_trigger_rule": "mean >= 0.005 and paired t 95% CI lower > 0",
        }

    results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(outputs[0], run_rows)
    write_csv(outputs[1], summary_rows)
    write_csv(outputs[2], paired_rows)
    payload = {
        "schema_version": 1,
        "status": "FROZEN_FROM_VALIDATION",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "metric_split": "validation",
        "test_access_before_selection": False,
        "test_metrics_used_for_selection": False,
        "final_freeze_rule": "candidate if three-seed mean paired validation delta is positive, otherwise baseline",
        "t_critical_95_df2": T_CRITICAL_95_DF2,
        "datasets": frozen,
        "sources": [path.relative_to(REPO_ROOT).as_posix() for path in outputs[:3]],
    }
    outputs[3].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
