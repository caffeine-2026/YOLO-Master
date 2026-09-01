#!/usr/bin/env python3
"""Recompute augmentation test, resource, paired, and per-class tables from raw evidence."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "smoke" / "c3" / "augmentation"
T_CRITICAL_95_DF2 = 4.30265273


def stats(values: list[float]) -> dict[str, float]:
    if len(values) != 3:
        raise ValueError(f"Expected three seeds, got {len(values)}")
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    half = T_CRITICAL_95_DF2 * std / math.sqrt(3)
    return {"mean": mean, "sample_std": std, "ci95_lower": mean - half, "ci95_upper": mean + half}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty result table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_training(payload: dict) -> dict:
    path = REPO_ROOT / payload["source_training_metrics"]
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    outputs = {
        name: ROOT / "results" / name
        for name in (
            "locked_test_runs.csv",
            "locked_test_summary.csv",
            "paired_test_statistics.csv",
            "historical_baseline_paired_statistics.csv",
            "per_class_test_runs.csv",
            "per_class_test_summary.csv",
            "per_class_paired_statistics.csv",
            "reference_comparison.csv",
            "scaling_comparison.csv",
        )
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Refusing to overwrite aggregate result evidence")
    frozen = json.loads((ROOT / "results" / "frozen_selection.json").read_text(encoding="utf-8"))
    evaluations = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted((ROOT / "evaluations").glob("*.json"))
    ]
    if not evaluations or any(item.get("status") != "PASS" for item in evaluations):
        raise ValueError("Locked test evaluations are absent or incomplete")

    run_rows = []
    class_rows = []
    for item in evaluations:
        training = source_training(item)
        run_rows.append(
            {
                "dataset": item["dataset"],
                "sample_size": item["sample_size"],
                "policy": item["policy"],
                "seed": item["seed"],
                "map50_95": item["overall"]["map50_95"],
                "map50": item["overall"]["map50"],
                "precision": item["overall"]["precision"],
                "recall": item["overall"]["recall"],
                "trainable_parameters": training["parameters"]["trainable_parameters"],
                "total_parameters": training["parameters"]["total_parameters"],
                "peak_gpu_memory_mib": training["resources"]["peak_gpu_memory_mib"],
                "training_seconds": training["timing"]["training_seconds"],
                "gpu_hours": training["timing"]["gpu_hours"],
                "source_run_id": item["source_run_id"],
                "source_evaluation": f"smoke/c3/augmentation/evaluations/test_{item['dataset']}_{item['sample_size']}_{item['policy']}_seed{item['seed']}.json",
            }
        )
        for row in item["per_class"]:
            class_rows.append(
                {
                    "dataset": item["dataset"],
                    "sample_size": item["sample_size"],
                    "policy": item["policy"],
                    "seed": item["seed"],
                    **row,
                }
            )

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["dataset"], row["sample_size"], row["policy"])].append(row)
    summary_rows = []
    for (dataset, sample_size, policy), rows in sorted(grouped.items()):
        if sorted(row["seed"] for row in rows) != [824, 825, 826]:
            raise ValueError(f"Incomplete seeds for {dataset}/{sample_size}/{policy}")
        for metric in (
            "map50_95",
            "map50",
            "precision",
            "recall",
            "peak_gpu_memory_mib",
            "training_seconds",
            "gpu_hours",
        ):
            summary_rows.append(
                {
                    "dataset": dataset,
                    "sample_size": sample_size,
                    "policy": policy,
                    "metric": metric,
                    "n": 3,
                    **stats([float(row[metric]) for row in rows]),
                }
            )

    paired_rows = []
    for dataset, details in frozen["datasets"].items():
        augmented = details["frozen_policy"]
        if augmented == "baseline":
            continue
        sizes = [100] + ([10, 50, 500] if details["scaling_trigger_passed"] else [])
        for sample_size in sizes:
            for metric in ("map50_95", "map50", "precision", "recall"):
                base = {row["seed"]: float(row[metric]) for row in grouped[(dataset, sample_size, "baseline")]}
                aug = {row["seed"]: float(row[metric]) for row in grouped[(dataset, sample_size, augmented)]}
                deltas = [aug[seed] - base[seed] for seed in (824, 825, 826)]
                interval = stats(deltas)
                paired_rows.append(
                    {
                        "dataset": dataset,
                        "sample_size": sample_size,
                        "baseline": "baseline",
                        "augmentation": augmented,
                        "metric": metric,
                        "seed824_delta": deltas[0],
                        "seed825_delta": deltas[1],
                        "seed826_delta": deltas[2],
                        **interval,
                        "significant_positive_95ci": interval["ci95_lower"] > 0.0,
                    }
                )

    historical_paired = []
    for dataset, details in frozen["datasets"].items():
        augmented = details["frozen_policy"]
        old_prefix = "neu" if dataset == "neu" else "deeppcb"
        for metric, old_key in (
            ("map50_95", "map50_95"),
            ("map50", "map50"),
            ("precision", "precision"),
            ("recall", "recall"),
        ):
            current = {row["seed"]: float(row[metric]) for row in grouped[(dataset, 100, augmented)]}
            historical = {}
            sources = []
            for seed in (824, 825, 826):
                source = REPO_ROOT / f"smoke/c3/p1/logs/{old_prefix}_vpeft_seed{seed}_e100/metrics.json"
                payload = json.loads(source.read_text(encoding="utf-8"))
                historical[seed] = float(payload["test"][old_key])
                sources.append(source.relative_to(REPO_ROOT).as_posix())
            deltas = [current[seed] - historical[seed] for seed in (824, 825, 826)]
            interval = stats(deltas)
            historical_paired.append(
                {
                    "dataset": dataset,
                    "sample_size": 100,
                    "historical_policy": "existing accuracy-first V-PEFT (613602 params)",
                    "new_policy": augmented,
                    "metric": metric,
                    "seed824_delta": deltas[0],
                    "seed825_delta": deltas[1],
                    "seed826_delta": deltas[2],
                    **interval,
                    "significant_positive_95ci": interval["ci95_lower"] > 0.0,
                    "significant_negative_95ci": interval["ci95_upper"] < 0.0,
                    "historical_sources": ";".join(sources),
                }
            )

    class_grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in class_rows:
        class_grouped[(row["dataset"], row["sample_size"], row["policy"], row["class_id"], row["class_name"])].append(
            row
        )
    class_summary = []
    for (dataset, sample_size, policy, class_id, class_name), rows in sorted(class_grouped.items()):
        if len(rows) != 3:
            raise ValueError(f"Incomplete per-class seeds for {dataset}/{sample_size}/{policy}/{class_name}")
        for metric in ("ap50_95", "ap50", "precision", "recall"):
            class_summary.append(
                {
                    "dataset": dataset,
                    "sample_size": sample_size,
                    "policy": policy,
                    "class_id": class_id,
                    "class_name": class_name,
                    "metric": metric,
                    "n": 3,
                    **stats([float(row[metric]) for row in rows]),
                }
            )

    class_paired = []
    for dataset, details in frozen["datasets"].items():
        augmented = details["frozen_policy"]
        if augmented == "baseline":
            continue
        sizes = [100] + ([10, 50, 500] if details["scaling_trigger_passed"] else [])
        for sample_size in sizes:
            class_names = sorted(
                {
                    row["class_name"]
                    for row in class_rows
                    if row["dataset"] == dataset and row["sample_size"] == sample_size
                }
            )
            for class_name in class_names:
                for metric in ("ap50_95", "ap50", "precision", "recall"):
                    base = {
                        row["seed"]: float(row[metric])
                        for row in class_rows
                        if row["dataset"] == dataset
                        and row["sample_size"] == sample_size
                        and row["policy"] == "baseline"
                        and row["class_name"] == class_name
                    }
                    aug = {
                        row["seed"]: float(row[metric])
                        for row in class_rows
                        if row["dataset"] == dataset
                        and row["sample_size"] == sample_size
                        and row["policy"] == augmented
                        and row["class_name"] == class_name
                    }
                    deltas = [aug[seed] - base[seed] for seed in (824, 825, 826)]
                    interval = stats(deltas)
                    class_paired.append(
                        {
                            "dataset": dataset,
                            "sample_size": sample_size,
                            "class_name": class_name,
                            "baseline": "baseline",
                            "augmentation": augmented,
                            "metric": metric,
                            "seed824_delta": deltas[0],
                            "seed825_delta": deltas[1],
                            "seed826_delta": deltas[2],
                            **interval,
                            "significant_positive_95ci": interval["ci95_lower"] > 0.0,
                            "significant_negative_95ci": interval["ci95_upper"] < 0.0,
                        }
                    )

    reference_rows = []
    full_sft_scaling = []
    with (REPO_ROOT / "smoke/c3/completion/results/final_summary.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] == "DeepPCB" and row["method"] == "Full-SFT":
                full_sft_scaling.append(
                    {
                        "dataset": "deeppcb",
                        "sample_size": row["sample_size"],
                        "method": "Full-SFT reference",
                        "map50_95_mean": row["map50_95_mean"],
                        "map50_95_ci95_lower": row["map50_95_ci95_lower"],
                        "map50_95_ci95_upper": row["map50_95_ci95_upper"],
                        "accuracy_retention_vs_full": 1.0,
                        "source": "smoke/c3/completion/results/final_summary.csv",
                    }
                )
            if int(row["sample_size"]) == 100 and row["method"] == "Full-SFT":
                reference_rows.append(
                    {
                        "dataset": row["dataset"],
                        "sample_size": 100,
                        "method": "Full-SFT reference",
                        "augmentation": "historical locked protocol",
                        "map50_95_mean": row["map50_95_mean"],
                        "map50_mean": row["map50_mean"],
                        "trainable_parameters": row["trainable_parameters"],
                        "total_parameters": row["total_parameters"],
                        "peak_gpu_memory_mib_mean": row["peak_gpu_memory_mib_mean"],
                        "training_seconds_mean": row["training_seconds_mean"],
                        "gpu_hours_mean": row["gpu_hours_mean"],
                        "source": "smoke/c3/completion/results/final_summary.csv",
                    }
                )
    with (REPO_ROOT / "smoke/c3/completion/results/old_new_vpeft.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["sample_size"]) == 100:
                reference_rows.append(
                    {
                        "dataset": row["dataset"],
                        "sample_size": 100,
                        "method": "Existing accuracy-first V-PEFT",
                        "augmentation": "historical default policy; preserved, not mixed with new runs",
                        "map50_95_mean": row["old_map50_95_mean"],
                        "map50_mean": row["old_map50_mean"],
                        "trainable_parameters": row["old_trainable_parameters"],
                        "total_parameters": row["old_total_parameters"],
                        "peak_gpu_memory_mib_mean": row["old_peak_gpu_memory_mib_mean"],
                        "training_seconds_mean": row["old_training_seconds_mean"],
                        "gpu_hours_mean": row["old_gpu_hours_mean"],
                        "source": "smoke/c3/completion/results/old_new_vpeft.csv",
                    }
                )
                reference_rows.append(
                    {
                        "dataset": row["dataset"],
                        "sample_size": 100,
                        "method": "V-PEFT <=10% efficiency-first negative",
                        "augmentation": "historical locked protocol; not used for accuracy tuning",
                        "map50_95_mean": row["new_map50_95_mean"],
                        "map50_mean": row["new_map50_mean"],
                        "trainable_parameters": row["new_trainable_parameters"],
                        "total_parameters": row["new_total_parameters"],
                        "peak_gpu_memory_mib_mean": row["new_peak_gpu_memory_mib_mean"],
                        "training_seconds_mean": row["new_training_seconds_mean"],
                        "gpu_hours_mean": row["new_gpu_hours_mean"],
                        "source": "smoke/c3/completion/results/old_new_vpeft.csv",
                    }
                )
    for (dataset, sample_size, policy), rows in sorted(grouped.items()):
        if sample_size != 100:
            continue
        pretty = "NEU-DET" if dataset == "neu" else "DeepPCB"
        reference_rows.append(
            {
                "dataset": pretty,
                "sample_size": 100,
                "method": f"New accuracy-first V-PEFT ({policy})",
                "augmentation": "new preregistered policy",
                "map50_95_mean": stats([float(row["map50_95"]) for row in rows])["mean"],
                "map50_mean": stats([float(row["map50"]) for row in rows])["mean"],
                "trainable_parameters": rows[0]["trainable_parameters"],
                "total_parameters": rows[0]["total_parameters"],
                "peak_gpu_memory_mib_mean": stats([float(row["peak_gpu_memory_mib"]) for row in rows])["mean"],
                "training_seconds_mean": stats([float(row["training_seconds"]) for row in rows])["mean"],
                "gpu_hours_mean": stats([float(row["gpu_hours"]) for row in rows])["mean"],
                "source": "smoke/c3/augmentation/results/locked_test_runs.csv",
            }
        )

    full_100 = {
        row["dataset"]: float(row["map50_95_mean"]) for row in reference_rows if row["method"] == "Full-SFT reference"
    }
    for row in reference_rows:
        row["accuracy_retention_vs_full"] = float(row["map50_95_mean"]) / full_100[row["dataset"]]

    write_csv(outputs["locked_test_runs.csv"], run_rows)
    write_csv(outputs["locked_test_summary.csv"], summary_rows)
    write_csv(outputs["paired_test_statistics.csv"], paired_rows)
    write_csv(outputs["historical_baseline_paired_statistics.csv"], historical_paired)
    write_csv(outputs["per_class_test_runs.csv"], class_rows)
    write_csv(outputs["per_class_test_summary.csv"], class_summary)
    write_csv(outputs["per_class_paired_statistics.csv"], class_paired)
    write_csv(outputs["reference_comparison.csv"], reference_rows)
    full_scaling_map = {int(row["sample_size"]): float(row["map50_95_mean"]) for row in full_sft_scaling}
    new_scaling = [
        {
            "dataset": "deeppcb",
            "sample_size": row["sample_size"],
            "method": f"V-PEFT {row['policy']}",
            "map50_95_mean": row["mean"],
            "map50_95_ci95_lower": row["ci95_lower"],
            "map50_95_ci95_upper": row["ci95_upper"],
            "accuracy_retention_vs_full": float(row["mean"]) / full_scaling_map[int(row["sample_size"])],
            "source": "smoke/c3/augmentation/results/locked_test_summary.csv",
        }
        for row in summary_rows
        if row["dataset"] == "deeppcb" and row["metric"] == "map50_95"
    ]
    write_csv(outputs["scaling_comparison.csv"], [*full_sft_scaling, *new_scaling])
    print(json.dumps({"status": "PASS", "evaluations": len(evaluations), "outputs": list(outputs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
