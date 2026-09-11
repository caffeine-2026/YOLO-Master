"""Validated readers for immutable C3 P0/P1/P2 experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from utils.paths import P0_ROOT, P1_ROOT, P2_ROOT, relative_path

P1_FILES = {
    "all_runs": P1_ROOT / "results" / "p1_all_runs.csv",
    "summary": P1_ROOT / "results" / "p1_summary.csv",
    "tradeoff": P1_ROOT / "results" / "tradeoff_multiseed.csv",
}
P2_FILES = {
    "all_runs": P2_ROOT / "results" / "p2_all_runs.csv",
    "summary": P2_ROOT / "results" / "p2_summary.csv",
    "retention": P2_ROOT / "results" / "retention_multiseed.csv",
    "paired": P2_ROOT / "results" / "paired_analysis.csv",
}
METHODS = ("Full-SFT", "Frozen Backbone", "V-PEFT")
DATASETS = ("NEU-DET", "DeepPCB")
SAMPLE_SIZES = (10, 50, 100, 500)


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(relative_path(path))
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{relative_path(path)} missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{relative_path(path)} is empty")
    return frame


@lru_cache(maxsize=1)
def p1_all_runs() -> pd.DataFrame:
    return _read_csv(
        P1_FILES["all_runs"],
        {"dataset", "method", "seed", "mAP50-95", "mAP50", "run_id", "checkpoint_sha256", "status"},
    )


@lru_cache(maxsize=1)
def p1_summary() -> pd.DataFrame:
    return _read_csv(
        P1_FILES["summary"],
        {
            "dataset",
            "method",
            "map50_95_mean",
            "map50_mean",
            "precision_mean",
            "recall_mean",
            "trainable_params",
            "peak_gpu_memory_mib_mean",
            "elapsed_seconds_mean",
            "gpu_hours_mean",
        },
    )


@lru_cache(maxsize=1)
def p1_tradeoff() -> pd.DataFrame:
    return _read_csv(P1_FILES["tradeoff"], {"dataset", "parameter_reduction", "memory_saving", "training_time_change"})


@lru_cache(maxsize=1)
def p2_all_runs() -> pd.DataFrame:
    return _read_csv(
        P2_FILES["all_runs"],
        {
            "dataset",
            "sample_size",
            "method",
            "seed",
            "mAP50-95",
            "mAP50",
            "peak_gpu_memory",
            "elapsed_time",
            "gpu_hours",
            "status",
        },
    )


@lru_cache(maxsize=1)
def p2_summary() -> pd.DataFrame:
    return _read_csv(
        P2_FILES["summary"],
        {
            "dataset",
            "sample_size",
            "method",
            "map50_95_mean",
            "map50_mean",
            "precision_mean",
            "recall_mean",
            "peak_gpu_memory_mib_mean",
            "elapsed_seconds_mean",
            "gpu_hours_mean",
            "trainable_params",
            "total_params",
        },
    )


@lru_cache(maxsize=1)
def p2_retention() -> pd.DataFrame:
    return _read_csv(
        P2_FILES["retention"],
        {
            "dataset",
            "sample_size",
            "accuracy_retention",
            "trainable_parameter_reduction",
            "memory_saving",
            "training_time_change",
        },
    )


@lru_cache(maxsize=1)
def paired_analysis() -> pd.DataFrame:
    return _read_csv(P2_FILES["paired"], {"dataset", "sample_size", "method_a", "method_b", "mean_delta"})


def validate_matrix() -> dict[str, int | str]:
    frame = p2_all_runs()
    expected = len(DATASETS) * len(SAMPLE_SIZES) * len(METHODS) * 3
    passed = int((frame["status"] == "PASS").sum())
    return {"expected": expected, "passed": passed, "status": "PASS" if passed == expected == len(frame) else "FAIL"}


@lru_cache(maxsize=1)
def stage_status() -> dict[str, str | int]:
    p0 = json.loads((P0_ROOT / "evidence" / "c3_p0_summary.json").read_text(encoding="utf-8"))
    p1 = json.loads((P1_ROOT / "evidence" / "p1_final_validation.json").read_text(encoding="utf-8"))
    p2 = json.loads((P2_ROOT / "evidence" / "p2_final_validation.json").read_text(encoding="utf-8"))
    matrix = p2.get("run_validation", validate_matrix())
    return {
        "P0": p0.get("official_p0", {}).get("status", "UNKNOWN"),
        "P1": p1.get("P1 status", "UNKNOWN"),
        "P2": p2.get("P2 status", p2.get("Overall C3 P2", "UNKNOWN")),
        "matrix_pass": int(matrix.get("pass", validate_matrix()["passed"])),
        "matrix_expected": int(matrix.get("expected", validate_matrix()["expected"])),
    }


def overview_metrics() -> dict[str, object]:
    retention = p2_retention().sort_values(["dataset", "sample_size"])
    reduction = float(retention["trainable_parameter_reduction"].median())
    return {
        "parameter_reduction": reduction,
        "retention": retention[["dataset", "sample_size", "accuracy_retention"]].copy(),
        "memory_saving_min": float(retention["memory_saving"].min()),
        "memory_saving_max": float(retention["memory_saving"].max()),
        "training_time_min": float(retention["training_time_change"].min()),
        "training_time_max": float(retention["training_time_change"].max()),
    }


def comparison_table(dataset: str, sample_size: int) -> pd.DataFrame:
    if dataset not in DATASETS or int(sample_size) not in SAMPLE_SIZES:
        raise ValueError("Unsupported dataset or sample size")
    frame = p2_summary()
    selected = frame[(frame["dataset"] == dataset) & (frame["sample_size"] == int(sample_size))].copy()
    if set(selected["method"]) != set(METHODS):
        raise ValueError(f"Incomplete comparison cell: {dataset}/{sample_size}")
    selected["method"] = pd.Categorical(selected["method"], METHODS, ordered=True)
    selected = selected.sort_values("method")
    full_params = float(selected.loc[selected["method"] == "Full-SFT", "trainable_params"].iloc[0])
    return pd.DataFrame(
        {
            "Method": selected["method"].astype(str),
            "mAP50-95": selected["map50_95_mean"].round(4),
            "mAP50": selected["map50_mean"].round(4),
            "Precision": selected["precision_mean"].round(4),
            "Recall": selected["recall_mean"].round(4),
            "Trainable Params": selected["trainable_params"].astype(int),
            "Parameter Reduction": (1.0 - selected["trainable_params"] / full_params).map(lambda value: f"{value:.2%}"),
            "Peak GPU Memory (MiB)": selected["peak_gpu_memory_mib_mean"].round(2),
            "Training Time (s)": selected["elapsed_seconds_mean"].round(2),
            "GPU-hours": selected["gpu_hours_mean"].round(4),
        }
    )


def scaling_data(dataset: str, metric: str) -> pd.DataFrame:
    if dataset not in DATASETS:
        raise ValueError("Unsupported dataset")
    summary = p2_summary()
    subset = summary[summary["dataset"] == dataset].copy()
    metric_map = {
        "mAP50-95": ("map50_95_mean", "map50_95_ci95_lower", "map50_95_ci95_upper"),
        "mAP50": ("map50_mean", "map50_ci95_lower", "map50_ci95_upper"),
        "Peak GPU Memory": (
            "peak_gpu_memory_mib_mean",
            "peak_gpu_memory_mib_ci95_lower",
            "peak_gpu_memory_mib_ci95_upper",
        ),
        "GPU-hours": ("gpu_hours_mean", "gpu_hours_ci95_lower", "gpu_hours_ci95_upper"),
    }
    if metric == "Accuracy Retention":
        runs = p2_all_runs()
        runs = runs[runs["dataset"] == dataset].copy()
        baseline = runs[runs["method"] == "Full-SFT"][["sample_size", "seed", "mAP50-95"]].rename(
            columns={"mAP50-95": "baseline"}
        )
        joined = runs.merge(baseline, on=["sample_size", "seed"], validate="many_to_one")
        joined["value"] = joined["mAP50-95"] / joined["baseline"]
        grouped = joined.groupby(["sample_size", "method"], observed=True)["value"]
        result = grouped.agg(["mean", "std", "count"]).reset_index()
        result["half_ci"] = 4.302652729911275 * result["std"] / result["count"].pow(0.5)
        result["lower"] = result["mean"] - result["half_ci"]
        result["upper"] = result["mean"] + result["half_ci"]
    elif metric in metric_map:
        mean_col, lower_col, upper_col = metric_map[metric]
        result = subset[["sample_size", "method", mean_col, lower_col, upper_col]].rename(
            columns={mean_col: "mean", lower_col: "lower", upper_col: "upper"}
        )
    else:
        raise ValueError("Unsupported scaling metric")
    result["method"] = pd.Categorical(result["method"], METHODS, ordered=True)
    return result.sort_values(["method", "sample_size"]).reset_index(drop=True)


@lru_cache(maxsize=8)
def canonical_checkpoint(dataset: str, method: str) -> tuple[Path, dict[str, object]]:
    """Resolve the immutable seed-824 100-epoch P1 checkpoint from p1_all_runs.csv."""
    if dataset not in DATASETS or method not in METHODS:
        raise ValueError("Unsupported dataset or method")
    frame = p1_all_runs()
    rows = frame[(frame["dataset"] == dataset) & (frame["method"] == method) & (frame["seed"] == 824)]
    if len(rows) != 1 or rows.iloc[0]["status"] != "PASS":
        raise ValueError(f"Canonical checkpoint row not unique/PASS: {dataset}/{method}")
    row = rows.iloc[0].to_dict()
    run_id = str(row["run_id"])
    if "e100" not in run_id:
        raise ValueError(f"Canonical run is not a 100-epoch final run: {run_id}")
    checkpoint = P1_ROOT / "artifacts" / run_id / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(relative_path(checkpoint))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != str(row["checkpoint_sha256"]):
        raise ValueError(f"Checkpoint SHA-256 mismatch: {relative_path(checkpoint)}")
    return checkpoint, row


@lru_cache(maxsize=1)
def planner_data() -> dict[str, object]:
    _, row = canonical_checkpoint("NEU-DET", "V-PEFT")
    metrics_path = P1_ROOT / "logs" / str(row["run_id"]) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    adapter = metrics.get("adapter", {})
    parameters = metrics.get("parameters", {})
    required = {
        "status": adapter.get("planner_status"),
        "planner_backend": adapter.get("planner_backend"),
        "actual_backend": adapter.get("actual_backend"),
        "planned_targets": adapter.get("planned_targets"),
        "applied_targets": adapter.get("applied_targets"),
        "trainable_params": parameters.get("trainable_parameters"),
        "adapter_params": parameters.get("adapter_parameters"),
    }
    if any(value is None for value in required.values()):
        raise ValueError(f"Incomplete planner data: {relative_path(metrics_path)}")
    full = p1_summary()
    full_params = int(full[(full["dataset"] == "NEU-DET") & (full["method"] == "Full-SFT")]["trainable_params"].iloc[0])
    required["parameter_reduction"] = 1.0 - int(required["trainable_params"]) / full_params
    required["source"] = relative_path(metrics_path)
    return required


def evidence_catalog() -> pd.DataFrame:
    items: list[tuple[str, Path]] = [
        ("P0 report", P0_ROOT / "docs" / "C3_P0_FINAL_REPORT.md"),
        ("P1 report", P1_ROOT / "docs" / "C3_P1_REPORT.md"),
        ("P2 report", P2_ROOT / "docs" / "C3_P2_REPORT.md"),
        ("Protocol", P1_ROOT / "config" / "protocol.yaml"),
        ("Protocol", P2_ROOT / "config" / "protocol.yaml"),
        ("Split manifest", P1_ROOT / "evidence" / "neu_det_split_manifest.json"),
        ("Split manifest", P1_ROOT / "evidence" / "deeppcb_split_manifest.json"),
        ("Split manifest", P2_ROOT / "evidence" / "neu_scaling_split_manifest.json"),
        ("Split manifest", P2_ROOT / "evidence" / "deeppcb_scaling_split_manifest.json"),
        ("Validation JSON", P0_ROOT / "evidence" / "c3_p0_summary.json"),
        ("Validation JSON", P1_ROOT / "evidence" / "p1_final_validation.json"),
        ("Validation JSON", P2_ROOT / "evidence" / "p2_final_validation.json"),
    ]
    items.extend(("Summary CSV", path) for path in [*P1_FILES.values(), *P2_FILES.values()])
    items.extend(("Visualization", path) for path in sorted((P2_ROOT / "visualizations" / "final").glob("*.png")))
    records = []
    for category, path in items:
        records.append(
            {
                "Category": category,
                "Repository path": relative_path(path),
                "Status": "Available" if path.is_file() else "Missing",
                "Size (KiB)": round(path.stat().st_size / 1024, 1) if path.is_file() else None,
            }
        )
    return pd.DataFrame(records)


def source_manifest() -> dict[str, list[str]]:
    return {
        "P1": [relative_path(path) for path in P1_FILES.values()],
        "P2": [relative_path(path) for path in P2_FILES.values()],
        "Planner": [str(planner_data()["source"])],
    }
