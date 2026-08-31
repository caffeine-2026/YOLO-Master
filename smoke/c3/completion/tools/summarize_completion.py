#!/usr/bin/env python3
"""Rebuild the selected 3-way matrix, statistics, trade-offs, and scaling figures."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smoke.c3.completion.tools.run_efficiency import ROOT, SEEDS, SIZES, selected_candidate
from smoke.c3.p2.tools.summarize_multiseed import run_location, verify_manifest

DATASETS = (("neu", "NEU-DET"), ("deeppcb", "DeepPCB"))
METHODS = ("Full-SFT", "Frozen Backbone", "V-PEFT <=10%")
T95_DF2 = 4.302652729911275


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    margin = T95_DF2 * std / math.sqrt(3)
    return mean, std, mean - margin, mean + margin


def verify_old_csv_row(row: dict[str, str]) -> dict:
    dataset = "neu" if row["dataset"] == "NEU-DET" else "deeppcb"
    tag = {"Full-SFT": "full", "Frozen Backbone": "frozen", "V-PEFT": "vpeft"}[row["method"]]
    run_id, log_dir, _ = run_location(dataset, int(row["sample_size"]), tag, int(row["seed"]))
    metrics_path = log_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest_ok, _ = verify_manifest(log_dir / "artifact_manifest.json")
    expected = {
        "mAP50-95": float(metrics["test"]["map50_95"]),
        "mAP50": float(metrics["test"]["map50"]),
        "trainable_params": int(metrics["parameters"]["trainable_parameters"]),
        "total_params": int(metrics["parameters"]["total_parameters"]),
        "peak_gpu_memory": float(metrics["resources"]["peak_gpu_memory_mib"]),
        "elapsed_time": float(metrics["timing"]["training_seconds"]),
    }
    for key, value in expected.items():
        observed = float(row[key])
        if not math.isclose(observed, float(value), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Existing CSV/raw mismatch for {run_id}/{key}: {observed} != {value}")
    if metrics["status"] != "PASS" or not manifest_ok:
        raise ValueError(f"Existing run failed integrity validation: {run_id}")
    return {
        "dataset": row["dataset"],
        "sample_size": int(row["sample_size"]),
        "method": row["method"],
        "seed": int(row["seed"]),
        "map50_95": float(row["mAP50-95"]),
        "map50": float(row["mAP50"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "trainable_parameters": int(row["trainable_params"]),
        "total_parameters": int(row["total_params"]),
        "peak_gpu_memory_mib": float(row["peak_gpu_memory"]),
        "training_seconds": float(row["elapsed_time"]),
        "gpu_hours": float(row["gpu_hours"]),
        "run_id": run_id,
        "metrics_path": metrics_path.relative_to(REPO_ROOT).as_posix(),
        "metrics_sha256": sha256(metrics_path),
        "checkpoint_path": metrics["checkpoint"]["path"],
        "checkpoint_sha256": metrics["checkpoint"]["sha256"],
        "source": "reused_verified_existing_full_or_frozen",
        "status": "PASS",
    }


def load_old_methods() -> tuple[list[dict], list[dict]]:
    path = REPO_ROOT / "smoke" / "c3" / "p2" / "results" / "p2_all_runs.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    # Verify all 72 published cells against raw metrics and manifests before
    # reusing either baselines or the old V-PEFT values in a comparison.
    verified_rows = [verify_old_csv_row(row) for row in source_rows]
    baseline = [row for row in verified_rows if row["method"] in {"Full-SFT", "Frozen Backbone"}]
    old_vpeft = [row for row in source_rows if row["method"] == "V-PEFT"]
    if len(baseline) != 48 or len(old_vpeft) != 24:
        raise ValueError("Existing P2 matrix does not contain the expected 48+24 cells")
    return baseline, old_vpeft


def load_new_vpeft(candidate: str) -> list[dict]:
    rows = []
    for dataset, dataset_name in DATASETS:
        for size in SIZES:
            for seed in SEEDS:
                run_id = f"final_{dataset}_{size}_vpeft_{candidate}_seed{seed}_e100"
                log_dir = ROOT / "logs" / "final" / run_id
                metrics_path = log_dir / "metrics.json"
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                manifest_ok, _ = verify_manifest(log_dir / "artifact_manifest.json")
                if (
                    metrics["status"] != "PASS"
                    or metrics["exit_code"] != 0
                    or not all(metrics["checks"].values())
                    or not manifest_ok
                ):
                    raise ValueError(f"New V-PEFT run failed integrity validation: {run_id}")
                if metrics["test"] is None:
                    raise ValueError(f"Final run has no locked test result: {run_id}")
                rows.append(
                    {
                        "dataset": dataset_name,
                        "sample_size": size,
                        "method": "V-PEFT <=10%",
                        "seed": seed,
                        "map50_95": float(metrics["test"]["map50_95"]),
                        "map50": float(metrics["test"]["map50"]),
                        "precision": float(metrics["test"]["precision"]),
                        "recall": float(metrics["test"]["recall"]),
                        "trainable_parameters": int(metrics["parameters"]["trainable_parameters"]),
                        "total_parameters": int(metrics["parameters"]["total_parameters"]),
                        "peak_gpu_memory_mib": float(metrics["resources"]["peak_gpu_memory_mib"]),
                        "training_seconds": float(metrics["timing"]["training_seconds"]),
                        "gpu_hours": float(metrics["timing"]["gpu_hours"]),
                        "run_id": run_id,
                        "metrics_path": metrics_path.relative_to(REPO_ROOT).as_posix(),
                        "metrics_sha256": sha256(metrics_path),
                        "checkpoint_path": metrics["checkpoint"]["best"]["path"],
                        "checkpoint_sha256": metrics["checkpoint"]["best"]["sha256"],
                        "source": "new_selected_efficiency_matrix",
                        "status": "PASS",
                    }
                )
    if len(rows) != 24:
        raise ValueError("New matrix does not contain 24 cells")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for _, dataset_name in DATASETS:
        for size in SIZES:
            full_mean = statistics.fmean(
                row["map50_95"]
                for row in rows
                if row["dataset"] == dataset_name and row["sample_size"] == size and row["method"] == "Full-SFT"
            )
            for method in METHODS:
                group = [
                    row
                    for row in rows
                    if row["dataset"] == dataset_name and row["sample_size"] == size and row["method"] == method
                ]
                if sorted(row["seed"] for row in group) != list(SEEDS):
                    raise ValueError(f"Broken seed pairing: {dataset_name}/{size}/{method}")
                map_mean, map_std, map_lower, map_upper = ci95([row["map50_95"] for row in group])
                map50_mean, map50_std, map50_lower, map50_upper = ci95([row["map50"] for row in group])
                memory_mean, memory_std, memory_lower, memory_upper = ci95(
                    [row["peak_gpu_memory_mib"] for row in group]
                )
                time_mean, time_std, time_lower, time_upper = ci95([row["training_seconds"] for row in group])
                output.append(
                    {
                        "dataset": dataset_name,
                        "sample_size": size,
                        "method": method,
                        "n": 3,
                        "map50_95_mean": map_mean,
                        "map50_95_std": map_std,
                        "map50_95_ci95_lower": map_lower,
                        "map50_95_ci95_upper": map_upper,
                        "map50_mean": map50_mean,
                        "map50_std": map50_std,
                        "map50_ci95_lower": map50_lower,
                        "map50_ci95_upper": map50_upper,
                        "accuracy_retention_vs_full": map_mean / full_mean,
                        "trainable_parameters": group[0]["trainable_parameters"],
                        "total_parameters": group[0]["total_parameters"],
                        "trainable_ratio_vs_full": group[0]["trainable_parameters"] / 2_590_994,
                        "peak_gpu_memory_mib_mean": memory_mean,
                        "peak_gpu_memory_mib_std": memory_std,
                        "peak_gpu_memory_mib_ci95_lower": memory_lower,
                        "peak_gpu_memory_mib_ci95_upper": memory_upper,
                        "training_seconds_mean": time_mean,
                        "training_seconds_std": time_std,
                        "training_seconds_ci95_lower": time_lower,
                        "training_seconds_ci95_upper": time_upper,
                        "gpu_hours_mean": statistics.fmean(row["gpu_hours"] for row in group),
                    }
                )
    return output


def paired(rows: list[dict], old_vpeft: list[dict[str, str]]) -> list[dict]:
    output = []
    old_index = {
        (row["dataset"], int(row["sample_size"]), int(row["seed"])): float(row["mAP50-95"]) for row in old_vpeft
    }
    for _, dataset_name in DATASETS:
        for size in SIZES:
            index = {
                (row["method"], row["seed"]): row
                for row in rows
                if row["dataset"] == dataset_name and row["sample_size"] == size
            }
            comparisons = {
                "new_minus_full": [
                    index[("V-PEFT <=10%", seed)]["map50_95"] - index[("Full-SFT", seed)]["map50_95"] for seed in SEEDS
                ],
                "new_minus_frozen": [
                    index[("V-PEFT <=10%", seed)]["map50_95"] - index[("Frozen Backbone", seed)]["map50_95"]
                    for seed in SEEDS
                ],
                "new_minus_old_vpeft": [
                    index[("V-PEFT <=10%", seed)]["map50_95"] - old_index[(dataset_name, size, seed)] for seed in SEEDS
                ],
            }
            for comparison, deltas in comparisons.items():
                mean, std, lower, upper = ci95(deltas)
                output.append(
                    {
                        "dataset": dataset_name,
                        "sample_size": size,
                        "comparison": comparison,
                        "delta_seed824": deltas[0],
                        "delta_seed825": deltas[1],
                        "delta_seed826": deltas[2],
                        "mean_delta": mean,
                        "std_delta": std,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                    }
                )
    return output


def old_new_table(summary: list[dict], old_vpeft: list[dict[str, str]]) -> list[dict]:
    output = []
    for _, dataset_name in DATASETS:
        for size in SIZES:
            old = [row for row in old_vpeft if row["dataset"] == dataset_name and int(row["sample_size"]) == size]
            new = next(
                row
                for row in summary
                if row["dataset"] == dataset_name and row["sample_size"] == size and row["method"] == "V-PEFT <=10%"
            )
            output.append(
                {
                    "dataset": dataset_name,
                    "sample_size": size,
                    "old_map50_95_mean": statistics.fmean(float(row["mAP50-95"]) for row in old),
                    "new_map50_95_mean": new["map50_95_mean"],
                    "old_map50_mean": statistics.fmean(float(row["mAP50"]) for row in old),
                    "new_map50_mean": new["map50_mean"],
                    "old_trainable_parameters": int(old[0]["trainable_params"]),
                    "new_trainable_parameters": new["trainable_parameters"],
                    "old_total_parameters": int(old[0]["total_params"]),
                    "new_total_parameters": new["total_parameters"],
                    "new_fraction_of_full": new["trainable_ratio_vs_full"],
                    "old_peak_gpu_memory_mib_mean": statistics.fmean(float(row["peak_gpu_memory"]) for row in old),
                    "new_peak_gpu_memory_mib_mean": new["peak_gpu_memory_mib_mean"],
                    "old_training_seconds_mean": statistics.fmean(float(row["elapsed_time"]) for row in old),
                    "new_training_seconds_mean": new["training_seconds_mean"],
                    "old_gpu_hours_mean": statistics.fmean(float(row["gpu_hours"]) for row in old),
                    "new_gpu_hours_mean": new["gpu_hours_mean"],
                }
            )
    return output


def plot_scaling(summary: list[dict]) -> tuple[list[str], dict]:
    output_dir = ROOT / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    series = {}
    colors = {"Full-SFT": "#1f77b4", "Frozen Backbone": "#ff7f0e", "V-PEFT <=10%": "#2ca02c"}
    for _, dataset_name in DATASETS:
        series[dataset_name] = {}
        fig, axis = plt.subplots(figsize=(7, 4.5), dpi=160)
        for method in METHODS:
            group = [row for row in summary if row["dataset"] == dataset_name and row["method"] == method]
            series[dataset_name][method] = [
                {
                    "sample_size": row["sample_size"],
                    "mean": row["map50_95_mean"],
                    "ci95_lower": row["map50_95_ci95_lower"],
                    "ci95_upper": row["map50_95_ci95_upper"],
                }
                for row in group
            ]
            axis.errorbar(
                [row["sample_size"] for row in group],
                [row["map50_95_mean"] for row in group],
                yerr=[
                    [row["map50_95_mean"] - row["map50_95_ci95_lower"] for row in group],
                    [row["map50_95_ci95_upper"] - row["map50_95_mean"] for row in group],
                ],
                marker="o",
                capsize=3,
                label=method,
                color=colors[method],
            )
        axis.set_xscale("log")
        axis.set_xticks(SIZES, labels=[str(size) for size in SIZES])
        axis.set_xlabel("Training shots")
        axis.set_ylabel("Locked-test mAP50-95")
        axis.set_title(f"{dataset_name}: selected 3-way scaling")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        filename = f"{dataset_name.lower().replace('-', '_')}_scaling_map50_95.png"
        path = output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        paths.append(path.relative_to(REPO_ROOT).as_posix())
    return paths, series


def main() -> int:
    for path in (ROOT / "results" / "final_all_runs.csv", ROOT / "results" / "completion_summary.json"):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite final result: {path.relative_to(REPO_ROOT)}")
    candidate = selected_candidate()
    baseline, old_vpeft = load_old_methods()
    new_vpeft = load_new_vpeft(candidate)
    all_rows = sorted(
        baseline + new_vpeft, key=lambda row: (row["dataset"], row["sample_size"], row["method"], row["seed"])
    )
    if len(all_rows) != 72:
        raise RuntimeError("Selected comparison matrix must contain 72 rows")
    summary = summarize(all_rows)
    paired_rows = paired(all_rows, old_vpeft)
    old_new = old_new_table(summary, old_vpeft)
    write_csv(ROOT / "results" / "final_all_runs.csv", all_rows)
    write_csv(ROOT / "results" / "final_summary.csv", summary)
    write_csv(ROOT / "results" / "paired_deltas.csv", paired_rows)
    write_csv(ROOT / "results" / "old_new_vpeft.csv", old_new)
    figures, figure_series = plot_scaling(summary)
    payload = {
        "schema_version": 1,
        "selected_candidate": candidate,
        "matrix_rows": len(all_rows),
        "new_vpeft_runs": len(new_vpeft),
        "reused_verified_full_frozen_runs": len(baseline),
        "reused_verified_old_vpeft_runs_for_tradeoff_only": len(old_vpeft),
        "source_old_matrix": "smoke/c3/p2/results/p2_all_runs.csv",
        "source_old_matrix_sha256": sha256(REPO_ROOT / "smoke" / "c3" / "p2" / "results" / "p2_all_runs.csv"),
        "outputs": {
            "all_runs": "smoke/c3/completion/results/final_all_runs.csv",
            "summary": "smoke/c3/completion/results/final_summary.csv",
            "paired_deltas": "smoke/c3/completion/results/paired_deltas.csv",
            "old_new_vpeft": "smoke/c3/completion/results/old_new_vpeft.csv",
            "figures": figures,
        },
        "figure_series_derived_from_final_summary": figure_series,
    }
    (ROOT / "results" / "completion_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"matrix_rows": len(all_rows), "new_vpeft_runs": len(new_vpeft), "figures": figures}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
