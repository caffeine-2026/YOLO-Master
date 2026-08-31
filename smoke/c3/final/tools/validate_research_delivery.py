#!/usr/bin/env python3
"""Cross-check the complete C3 delivery against raw runs, checkpoints, and plots."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
C3_ROOT = REPO_ROOT / "smoke" / "c3"
FINAL_ROOT = C3_ROOT / "final"
P0_ROOT = C3_ROOT / "p0"
P1_ROOT = C3_ROOT / "p1"
P2_ROOT = C3_ROOT / "p2"
T95_DF2 = 4.302652729911275
SEEDS = (824, 825, 826)
SIZES = (10, 50, 100, 500)
METHODS = (
    ("full", "full_sft", "Full-SFT"),
    ("frozen", "frozen_backbone", "Frozen Backbone"),
    ("vpeft", "vpeft", "V-PEFT"),
)
LOCKED_KEYS = (
    "model",
    "data",
    "epochs",
    "batch",
    "imgsz",
    "workers",
    "seed",
    "deterministic",
    "amp",
    "cache",
    "val",
    "patience",
    "optimizer",
    "lr0",
    "lrf",
    "weight_decay",
    "momentum",
    "cos_lr",
    "warmup_epochs",
    "warmup_bias_lr",
    "close_mosaic",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "cutmix",
    "freeze",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_backend",
    "lora_planner_enabled",
    "lora_planner_backend",
    "lora_planner_solver",
    "lora_adapter_budget",
    "lora_vpeft_strict",
)
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".md", ".py", ".txt", ".yaml", ".yml"}
SECRET_PATTERNS = {
    "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "absolute_linux_user_path": re.compile(r"/home/[^\s/'\"<>]+/"),
    "absolute_macos_user_path": re.compile(r"/Users/[^\s/'\"<>]+/"),
    "absolute_windows_user_path": re.compile(r"[A-Za-z]:\\\\Users\\\\[^\s\\\"<>]+\\\\"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="smoke/c3/final/evidence/research_delivery_validation.json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def close(a: object, b: object, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def ci95(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    margin = T95_DF2 * std / math.sqrt(len(values))
    return mean, std, mean - margin, mean + margin


def verify_manifest(path: Path) -> tuple[bool, int, list[str]]:
    payload = load_json(path)
    errors = []
    for row in payload["artifacts"]:
        artifact = REPO_ROOT / row["path"]
        if not artifact.is_file():
            errors.append(f"missing:{row['path']}")
        elif artifact.stat().st_size != row["size_bytes"]:
            errors.append(f"size:{row['path']}")
        elif sha256(artifact) != row["sha256"]:
            errors.append(f"sha256:{row['path']}")
    return bool(payload["artifacts"]) and not errors, len(payload["artifacts"]), errors


def run_specs() -> list[dict[str, object]]:
    specs = []
    for dataset, dataset_name in (("neu", "NEU-DET"), ("deeppcb", "DeepPCB")):
        for size in SIZES:
            for tag, method, method_name in METHODS:
                for seed in SEEDS:
                    if size == 100:
                        run_id = f"{dataset}_{tag}_seed{seed}_e100"
                        stage_root = P1_ROOT
                        source = "reused_p1_100_multiseed"
                    else:
                        suffix = "" if seed == 824 else "_e100"
                        run_id = f"{dataset}_{size}_{tag}_seed{seed}{suffix}"
                        stage_root = P2_ROOT
                        source = "immutable_p2_seed824" if seed == 824 else "new_p2_multiseed"
                    specs.append(
                        {
                            "dataset": dataset,
                            "dataset_name": dataset_name,
                            "sample_size": size,
                            "tag": tag,
                            "method": method,
                            "method_name": method_name,
                            "seed": seed,
                            "run_id": run_id,
                            "stage_root": stage_root,
                            "source": source,
                        }
                    )
    return specs


def validate_training_matrix() -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    checkpoint_failures = []
    parameter_pairs: dict[str, set[tuple[int, int]]] = defaultdict(set)
    historical_lovo_null_runs = []
    for spec in run_specs():
        stage_root = spec["stage_root"]
        log_dir = stage_root / "logs" / spec["run_id"]
        artifact_dir = stage_root / "artifacts" / spec["run_id"]
        required = (
            "command.txt",
            "resolved_config.yaml",
            "stdout.log",
            "stderr.log",
            "metrics.json",
            "resource_usage.json",
            "timing.json",
            "environment.json",
            "artifact_manifest.json",
            "learning_curve.csv",
            "test_metrics.json",
        )
        missing = [name for name in required if not (log_dir / name).is_file()]
        checks: dict[str, bool] = {"required_files": not missing}
        errors: list[str] = [f"missing:{name}" for name in missing]
        if not missing:
            metrics = load_json(log_dir / "metrics.json")
            test = load_json(log_dir / "test_metrics.json")
            resources = load_json(log_dir / "resource_usage.json")
            timing = load_json(log_dir / "timing.json")
            environment = load_json(log_dir / "environment.json")
            resolved = yaml.safe_load((log_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
            args = yaml.safe_load((artifact_dir / "args.yaml").read_text(encoding="utf-8"))
            stdout = (log_dir / "stdout.log").read_text(encoding="utf-8", errors="replace")
            stderr = (log_dir / "stderr.log").read_text(encoding="utf-8", errors="replace")
            artifact_rows = read_csv(artifact_dir / "results.csv")
            copied_rows = read_csv(log_dir / "learning_curve.csv")
            manifest_ok, manifest_count, manifest_errors = verify_manifest(log_dir / "artifact_manifest.json")
            checks.update(
                {
                    "identity": metrics["run_id"] == spec["run_id"]
                    and metrics["seed"] == spec["seed"]
                    and metrics["method"] == spec["method"],
                    "pass_and_exit_zero": metrics["status"] == "PASS"
                    and metrics["exit_code"] == 0
                    and timing["training_exit_code"] == 0,
                    "test_json_exact": metrics["test"] == test,
                    "resource_json_exact": metrics["resources"] == resources,
                    "timing_json_exact": metrics["timing"] == timing,
                    "real_gpu_recorded": environment["cuda_available"] is True
                    and "RTX 4090" in environment["gpu_name"]
                    and str(resolved["device"]) == str(test["device"]),
                    "args_match_resolved_config": all(args.get(key) == resolved.get(key) for key in LOCKED_KEYS),
                    "learning_curve_exact": artifact_rows == copied_rows,
                    "one_hundred_finite_epochs": len(artifact_rows) == 100
                    and [int(row["epoch"]) for row in artifact_rows] == list(range(1, 101))
                    and all(
                        math.isfinite(float(value.strip()))
                        for row in artifact_rows
                        for value in row.values()
                        if value and value.strip()
                    ),
                    "full_log_completed": "100 epochs completed in " in stdout
                    and "===== LOCKED TEST EVALUATION =====" in stdout
                    and "Results saved to <repo>/" in stdout,
                    "no_failure_marker": re.search(
                        r"Traceback|RuntimeError|ValueError|FAILED|CUDA out of memory", stdout + stderr, re.IGNORECASE
                    )
                    is None,
                    "test_metrics_finite": all(
                        math.isfinite(float(metrics["test"][key]))
                        for key in ("map50_95", "map50", "precision", "recall")
                    ),
                    "artifact_manifest_verified": manifest_ok,
                    "gpu_hours_exact": close(timing["gpu_hours"], timing["training_seconds"] / 3600, 1e-6),
                }
            )
            errors.extend(manifest_errors)
            gpu_values = [float(value) for value in re.findall(r"\b(\d+(?:\.\d+)?)G\b", stdout)]
            checks["gpu_memory_matches_raw_log"] = (
                bool(gpu_values) and abs(max(gpu_values) * 1024 - float(resources["peak_gpu_memory_mib"])) < 20
            )
            checkpoint = REPO_ROOT / metrics["checkpoint"]["path"]
            checks["checkpoint_hash"] = (
                checkpoint.is_file()
                and checkpoint.stat().st_size == metrics["checkpoint"]["size_bytes"]
                and sha256(checkpoint) == metrics["checkpoint"]["sha256"]
            )
            try:
                checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = checkpoint_payload.get("model")
                train_args = checkpoint_payload.get("train_args", {})
                checkpoint_total = sum(parameter.numel() for parameter in model.parameters()) if model else -1
                checks["checkpoint_load_and_metadata"] = (
                    model is not None
                    and checkpoint_total == metrics["parameters"]["total_parameters"]
                    and int(train_args["seed"]) == spec["seed"]
                    and int(train_args["epochs"]) == 100
                )
                del checkpoint_payload, model
                gc.collect()
            except Exception as error:  # noqa: BLE001  # pragma: no cover - audit must retain any load failure
                checks["checkpoint_load_and_metadata"] = False
                checkpoint_failures.append(f"{spec['run_id']}:{type(error).__name__}")
            parameter_pairs[spec["tag"]].add(
                (metrics["parameters"]["trainable_parameters"], metrics["parameters"]["total_parameters"])
            )
            if spec["tag"] == "vpeft":
                adapter = metrics["adapter"]
                runtime = load_json(log_dir / "vpeft_runtime_metadata.json")
                placement = runtime["runtime_metadata"]["placement_plan"]
                target_audit = runtime["runtime_metadata"]["target_audit"]
                checks["planner_actually_applied"] = (
                    adapter["planner_status"] in {"ACCEPT", "ADAPT"}
                    and adapter["planner_backend"] == "vpeft"
                    and adapter["actual_backend"] == "peft"
                    and adapter["planned_targets"] == len(placement["targets"]) == 59
                    and adapter["applied_targets"] == target_audit["selected_count"] == 52
                    and placement["solver"] == "ao"
                    and placement["budget"]["max_adapter_params"] == 2_100_000
                )
                if placement["predicted_delta"] is None or placement["confidence"] is None:
                    historical_lovo_null_runs.append(spec["run_id"])
            else:
                checks["method_boundary"] = (
                    spec["tag"] == "full"
                    and int(resolved.get("freeze") or 0) == 0
                    and int(resolved.get("lora_r") or 0) == 0
                ) or (
                    spec["tag"] == "frozen"
                    and int(resolved.get("freeze") or 0) == 11
                    and int(resolved.get("lora_r") or 0) == 0
                )
            row = {
                "run_id": spec["run_id"],
                "dataset": spec["dataset_name"],
                "sample_size": spec["sample_size"],
                "method": spec["method_name"],
                "seed": spec["seed"],
                "source": spec["source"],
                "metrics_path": str((log_dir / "metrics.json").relative_to(REPO_ROOT)),
                "full_log_path": str((log_dir / "stdout.log").relative_to(REPO_ROOT)),
                "resolved_config_path": str((log_dir / "resolved_config.yaml").relative_to(REPO_ROOT)),
                "checkpoint_path": metrics["checkpoint"]["path"],
                "manifest_records": manifest_count,
                "checks": checks,
                "errors": errors,
                "status": "PASS" if all(checks.values()) and not errors else "FAIL",
                "metrics": metrics,
            }
        else:
            row = {
                "run_id": spec["run_id"],
                "dataset": spec["dataset_name"],
                "sample_size": spec["sample_size"],
                "method": spec["method_name"],
                "seed": spec["seed"],
                "source": spec["source"],
                "checks": checks,
                "errors": errors,
                "status": "FAIL",
            }
        rows.append(row)
    passed = sum(row["status"] == "PASS" for row in rows)
    result = {
        "expected_cells": 72,
        "passed_cells": passed,
        "failed_run_ids": [row["run_id"] for row in rows if row["status"] != "PASS"],
        "checkpoint_load_failures": checkpoint_failures,
        "parameter_pairs": {key: sorted(values) for key, values in parameter_pairs.items()},
        "vpeft_expected": 24,
        "vpeft_planner_applied": sum(row.get("checks", {}).get("planner_actually_applied", False) for row in rows),
        "historical_lovo_null_runs": historical_lovo_null_runs,
        "historical_lovo_interpretation": (
            "These 24 immutable P1/P2 runs predate LOVO evidence fields. Null is preserved as historical raw evidence "
            "and is not treated as calibrated LOVO."
        ),
        "status": "PASS" if passed == 72 and not checkpoint_failures else "FAIL",
    }
    return result, rows


def validate_p0_planner() -> dict[str, object]:
    audit = load_json(P0_ROOT / "evidence" / "solver_audit_20260831.json")
    audit_by_run = {row["run"]: row for row in audit["solver_runs"]}
    runs = (
        ("neu_det_vpeft_gpu_fp32_seed824", "ao", "ao"),
        ("deeppcb_vpeft_gpu_fp32_seed824", "ao", "ao"),
        ("neu_det_vpeft_dco_fixed_gpu_fp32_seed824", "dco", "dco"),
        ("deeppcb_vpeft_dco_fixed_gpu_fp32_seed824", "dco", "dco"),
        ("neu_det_vpeft_mip_fallback_gpu_fp32_seed824", "mip", "ao"),
    )
    run_rows = []
    for run_id, expected_requested, expected_effective in runs:
        log_dir = P0_ROOT / "logs" / run_id
        result = load_json(log_dir / "result.json")
        resolved = yaml.safe_load((log_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        runtime = load_json(log_dir / "vpeft_runtime_metadata.json")
        placement = runtime["runtime_metadata"]["placement_plan"]
        target_audit = runtime["runtime_metadata"]["target_audit"]
        metadata = placement["metadata"]
        requested = metadata.get("requested_solver", resolved["lora_planner_solver"])
        effective = metadata.get("effective_solver", placement["solver"])
        audit_row = audit_by_run[run_id]
        checks = {
            "completed": result["status"] == "completed" and result["exit_code"] == 0,
            "fixed_protocol": resolved["epochs"] == 1
            and resolved["batch"] == 1
            and resolved["imgsz"] == 320
            and resolved["seed"] == 824
            and resolved["amp"] is False,
            "strict_planner": resolved["lora_vpeft_strict"] is True and resolved["lora_planner_backend"] == "vpeft",
            "decision_budget_targets": placement["status"] in {"ACCEPT", "ADAPT", "REFUSE"}
            and placement["budget"]["max_adapter_params"] == 2_100_000
            and len(placement["targets"]) == 59
            and target_audit["selected_count"] == 52,
            "requested_effective_solver": requested == expected_requested and effective == expected_effective,
            "audit_matches_runtime": audit_row["requested_solver"] == requested
            and audit_row["effective_solver"] == effective
            and audit_row["planned_targets"] == len(placement["targets"])
            and audit_row["used_adapter_params"] == placement["budget"]["used_adapter_params"],
        }
        run_rows.append(
            {
                "run_id": run_id,
                "requested_solver": requested,
                "effective_solver": effective,
                "decision": placement["status"],
                "adapter_budget": placement["budget"]["max_adapter_params"],
                "planned_module_count": len(placement["targets"]),
                "applied_module_count": target_audit["selected_count"],
                "ranks": sorted({target["rank"] for target in placement["targets"]}),
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )
    mip_runtime = load_json(
        P0_ROOT / "logs" / "neu_det_vpeft_mip_fallback_gpu_fp32_seed824" / "vpeft_runtime_metadata.json"
    )
    mip_fallback = mip_runtime["runtime_metadata"]["placement_plan"]["metadata"]["solver_fallback"]
    failed_logs = [
        P0_ROOT / "logs" / "neu_det_vpeft_dco_gpu_fp32_seed824" / "train.log",
        P0_ROOT / "logs" / "deeppcb_vpeft_dco_gpu_fp32_seed824" / "train.log",
    ]
    failure_text = [path.read_text(encoding="utf-8", errors="replace") for path in failed_logs]
    fixed_runtime = load_json(
        P0_ROOT / "logs" / "neu_det_vpeft_dco_fixed_gpu_fp32_seed824" / "vpeft_runtime_metadata.json"
    )
    fixed_targets = fixed_runtime["runtime_metadata"]["placement_plan"]["targets"]
    prediction = audit["prediction_evidence"]
    checks = {
        "five_solver_runs_verified": len(run_rows) == 5 and all(row["status"] == "PASS" for row in run_rows),
        "ao_and_dco_both_executed": {row["effective_solver"] for row in run_rows} >= {"ao", "dco"},
        "mipr_fallback_audited": mip_fallback["requested_solver"] == "mip"
        and mip_fallback["effective_solver"] == "ao"
        and mip_fallback["exception_type"] == "ImportError"
        and "OR-Tools" in mip_fallback["reason"],
        "dco_failure_logs_preserved": len(failed_logs) == 2
        and all("PlacementPlan rank 64 for '1.conv' exceeds layer capacity 16" in text for text in failure_text),
        "dco_capacity_fix_observed": next(row["rank"] for row in fixed_targets if row["name"] == "1.conv") == 16,
        "lovo_is_cold_start_prior": close(prediction["predicted_delta"], 0.06602954545454547)
        and prediction["confidence_score"] == 0
        and prediction["state"] == "cold_start"
        and prediction["source"] == "default_prior"
        and prediction["observation_count"] == 0
        and prediction["uses_learned_evidence"] is False,
        "lovo_calibration_pending": audit["conclusion"]["lovo_status"].startswith("runtime null removed")
        and prediction["observation_count"] < 5,
    }
    return {
        "checks": checks,
        "solver_runs": run_rows,
        "failed_log_paths": [str(path.relative_to(REPO_ROOT)) for path in failed_logs],
        "regression_test_paths": ["tests/test_vpeft.py", "tests/test_vpeft_lora_e2e.py"],
        "lovo": prediction,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def validate_result_tables(run_rows: list[dict[str, object]]) -> dict[str, object]:
    raw_lookup = {(row["dataset"], row["sample_size"], row["method"], row["seed"]): row["metrics"] for row in run_rows}
    p1_all = read_csv(P1_ROOT / "results" / "p1_all_runs.csv")
    p2_all = read_csv(P2_ROOT / "results" / "p2_all_runs.csv")
    p1_raw_exact = all(
        close(row["mAP50-95"], raw_lookup[(row["dataset"], 100, row["method"], int(row["seed"]))]["test"]["map50_95"])
        and close(row["mAP50"], raw_lookup[(row["dataset"], 100, row["method"], int(row["seed"]))]["test"]["map50"])
        and int(row["trainable_params"])
        == raw_lookup[(row["dataset"], 100, row["method"], int(row["seed"]))]["parameters"]["trainable_parameters"]
        and int(row["total_params"])
        == raw_lookup[(row["dataset"], 100, row["method"], int(row["seed"]))]["parameters"]["total_parameters"]
        for row in p1_all
    )
    p2_raw_exact = all(
        close(
            row["mAP50-95"],
            raw_lookup[(row["dataset"], int(row["sample_size"]), row["method"], int(row["seed"]))]["test"]["map50_95"],
        )
        and close(
            row["mAP50"],
            raw_lookup[(row["dataset"], int(row["sample_size"]), row["method"], int(row["seed"]))]["test"]["map50"],
        )
        and close(
            row["peak_gpu_memory"],
            raw_lookup[(row["dataset"], int(row["sample_size"]), row["method"], int(row["seed"]))]["resources"][
                "peak_gpu_memory_mib"
            ],
        )
        and close(
            row["elapsed_time"],
            raw_lookup[(row["dataset"], int(row["sample_size"]), row["method"], int(row["seed"]))]["timing"][
                "training_seconds"
            ],
        )
        for row in p2_all
    )

    def validate_summary(all_rows: list[dict[str, str]], summary_path: Path, with_size: bool) -> bool:
        summary = read_csv(summary_path)
        group_keys = ("dataset", "sample_size", "method") if with_size else ("dataset", "method")
        groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in all_rows:
            groups[tuple(row[key] for key in group_keys)].append(row)
        for row in summary:
            group = groups[tuple(row[key] for key in group_keys)]
            if sorted(int(item["seed"]) for item in group) != list(SEEDS):
                return False
            for source, prefix in (
                ("mAP50-95", "map50_95"),
                ("mAP50", "map50"),
                ("precision", "precision"),
                ("recall", "recall"),
            ):
                mean, std, lower, upper = ci95([float(item[source]) for item in group])
                for expected, suffix in ((mean, "mean"), (std, "std"), (lower, "ci95_lower"), (upper, "ci95_upper")):
                    if not close(row[f"{prefix}_{suffix}"], expected):
                        return False
        return True

    retention = read_csv(P2_ROOT / "results" / "retention_multiseed.csv")
    p2_summary = read_csv(P2_ROOT / "results" / "p2_summary.csv")
    summary_lookup = {(row["dataset"], row["sample_size"], row["method"]): row for row in p2_summary}
    retention_exact = True
    for row in retention:
        full = summary_lookup[(row["dataset"], row["sample_size"], "Full-SFT")]
        vpeft = summary_lookup[(row["dataset"], row["sample_size"], "V-PEFT")]
        retention_exact &= close(
            row["accuracy_retention"], float(vpeft["map50_95_mean"]) / float(full["map50_95_mean"])
        )
        retention_exact &= close(
            row["trainable_parameter_reduction"],
            1 - float(vpeft["trainable_params"]) / float(full["trainable_params"]),
        )
    figure_paths = sorted((P2_ROOT / "visualizations" / "final").glob("*.png"))
    figure_manifest = [
        {"path": str(path.relative_to(REPO_ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in figure_paths
    ]
    checks = {
        "p1_rows_18": len(p1_all) == 18,
        "p1_rows_match_raw_metrics": p1_raw_exact,
        "p1_means_and_ci_recomputed": validate_summary(p1_all, P1_ROOT / "results" / "p1_summary.csv", False),
        "p2_rows_72": len(p2_all) == 72,
        "p2_rows_match_raw_metrics": p2_raw_exact,
        "p2_means_and_ci_recomputed": validate_summary(p2_all, P2_ROOT / "results" / "p2_summary.csv", True),
        "retention_recomputed": retention_exact,
        "figures_14_present": len(figure_paths) == 14 and all(path.stat().st_size > 0 for path in figure_paths),
    }
    return {
        "checks": checks,
        "provenance": {
            "p1_raw_table": "smoke/c3/p1/results/p1_all_runs.csv",
            "p1_raw_table_sha256": sha256(P1_ROOT / "results" / "p1_all_runs.csv"),
            "p2_raw_table": "smoke/c3/p2/results/p2_all_runs.csv",
            "p2_raw_table_sha256": sha256(P2_ROOT / "results" / "p2_all_runs.csv"),
            "p2_summary_sha256": sha256(P2_ROOT / "results" / "p2_summary.csv"),
            "plot_generator": "smoke/c3/p2/tools/summarize_multiseed.py",
            "plot_generator_sha256": sha256(P2_ROOT / "tools" / "summarize_multiseed.py"),
            "figures": figure_manifest,
        },
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def validate_serialization_and_docs() -> dict[str, object]:
    # Validate the publishable delivery, not ignored third-party node_modules or
    # local-only raw artifact directories.  Those artifacts are independently
    # checked above through their manifests, hashes, configs, and checkpoints.
    listed = (
        subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "smoke/c3"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
    )
    delivery_files = sorted(REPO_ROOT / value for value in listed if value and (REPO_ROOT / value).is_file())
    json_files = [path for path in delivery_files if path.suffix.lower() == ".json"]
    yaml_files = [path for path in delivery_files if path.suffix.lower() in {".yaml", ".yml"}]
    parse_errors = []
    for path in json_files:
        try:
            load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            parse_errors.append(f"{path.relative_to(REPO_ROOT)}:{type(error).__name__}")
    for path in yaml_files:
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            parse_errors.append(f"{path.relative_to(REPO_ROOT)}:{type(error).__name__}")

    link_pattern = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
    broken_links = []
    checked_links = 0
    for document in [path for path in delivery_files if path.suffix.lower() == ".md"]:
        for raw_target in link_pattern.findall(document.read_text(encoding="utf-8", errors="replace")):
            target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not target:
                continue
            checked_links += 1
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                broken_links.append(f"{document.relative_to(REPO_ROOT)} -> {raw_target}")

    findings: dict[str, list[str]] = {name: [] for name in SECRET_PATTERNS}
    privacy_files_scanned = 0
    for path in delivery_files:
        relative = path.relative_to(REPO_ROOT)
        if (
            path == Path(__file__).resolve()
            or "/artifacts/" in f"/{relative.as_posix()}"
            or path.suffix.lower() not in TEXT_SUFFIXES
        ):
            continue
        privacy_files_scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings[name].append(str(path.relative_to(REPO_ROOT)))
    findings = {name: paths for name, paths in findings.items() if paths}
    checks = {
        "all_json_yaml_parse": not parse_errors,
        "all_markdown_local_links_resolve": not broken_links,
        "no_secret_or_absolute_user_path": not findings,
    }
    return {
        "checks": checks,
        "json_files_parsed": len(json_files),
        "yaml_files_parsed": len(yaml_files),
        "markdown_local_links_checked": checked_links,
        "privacy_files_scanned": privacy_files_scanned,
        "privacy_scope": "Publishable C3 logs, docs, configs, evidence, results, and source; raw model artifacts are excluded.",
        "parse_errors": parse_errors,
        "broken_links": broken_links,
        "privacy_findings": findings,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def write_raw_command_manifest() -> dict[str, object]:
    """Consolidate the immutable per-run command files without rewriting them."""
    command_paths = []
    for spec in run_specs():
        path = spec["stage_root"] / "logs" / spec["run_id"] / "command.txt"
        command_paths.append((str(spec["run_id"]), str(spec["source"]), path))
    for run_id in (
        "neu_det_vpeft_gpu_fp32_seed824",
        "deeppcb_vpeft_gpu_fp32_seed824",
        "neu_det_vpeft_dco_gpu_fp32_seed824",
        "deeppcb_vpeft_dco_gpu_fp32_seed824",
        "neu_det_vpeft_dco_fixed_gpu_fp32_seed824",
        "deeppcb_vpeft_dco_fixed_gpu_fp32_seed824",
        "neu_det_vpeft_mip_fallback_gpu_fp32_seed824",
    ):
        command_paths.append((run_id, "p0_solver_audit", P0_ROOT / "logs" / run_id / "command.txt"))
    rows = [
        {
            "run_id": run_id,
            "source": source,
            "command_path": str(path.relative_to(REPO_ROOT)),
            "command_sha256": sha256(path),
            "command_text": path.read_text(encoding="utf-8").rstrip(),
        }
        for run_id, source, path in command_paths
    ]
    payload = {"schema_version": 1, "command_file_count": len(rows), "runs": rows}
    path = FINAL_ROOT / "evidence" / "raw_command_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path.relative_to(REPO_ROOT)), "command_file_count": len(rows), "sha256": sha256(path)}


def main() -> int:
    args = parse_args()
    output = (REPO_ROOT / args.output).resolve()
    output.relative_to(FINAL_ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)

    command_manifest = write_raw_command_manifest()
    matrix, run_rows = validate_training_matrix()
    planner = validate_p0_planner()
    tables = validate_result_tables(run_rows)
    hygiene = validate_serialization_and_docs()
    current_branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    sections = {
        "p0_planner": planner,
        "p1_p2_training_matrix": matrix,
        "tables_and_figures": tables,
        "hygiene": hygiene,
    }
    overall = "PASS" if all(section["status"] == "PASS" for section in sections.values()) else "FAIL"
    payload = {
        "schema_version": 1,
        "scope": "C3 P0/P1/P2 raw-evidence cross-check",
        "feature_branch": current_branch,
        "raw_command_manifest": command_manifest,
        "sections": sections,
        "status": overall,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"C3_RESEARCH_DELIVERY={overall}", flush=True)
    print(
        f"P0={planner['status']} MATRIX={matrix['status']} TABLES={tables['status']} HYGIENE={hygiene['status']}",
        flush=True,
    )
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
