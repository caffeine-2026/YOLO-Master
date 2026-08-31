#!/usr/bin/env python3
"""Validate P0 immutability and a six-run C3 P1 seed-824 evidence stage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
# P0 was moved from ``smoke/c3`` to ``smoke/c3/p0`` after P1 started.  Use the
# first commit after that lossless move when checking immutable run evidence;
# comparing the pre-move paths makes every current checkout look deleted.
P0_EVIDENCE_REF = "f4b1af0e5aa5669bd14dee2660cb0b8286bb293d"
DATASETS = ("neu_det", "deeppcb")
METHODS = ("full_sft", "frozen_backbone", "vpeft")
REQUIRED_FILES = (
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
FAIR_KEYS = (
    "task",
    "mode",
    "model",
    "pretrained",
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="smoke/c3/p1/evidence/pilot_validation.json")
    parser.add_argument("--epochs", type=int, choices=(30, 50, 75, 100), default=30)
    return parser.parse_args()


def run_id(dataset: str, method: str, epochs: int) -> str:
    if epochs == 30:
        return f"{dataset}_{method}_seed824"
    dataset_tag = {"neu_det": "neu", "deeppcb": "deeppcb"}[dataset]
    method_tag = {"full_sft": "full", "frozen_backbone": "frozen", "vpeft": "vpeft"}[method]
    return f"{dataset_tag}_{method_tag}_seed824_e{epochs}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_artifacts(log_dir: Path) -> tuple[bool, int, list[str]]:
    manifest_path = log_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        return False, 0, ["missing artifact_manifest.json"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    rows = payload.get("artifacts", [])
    for row in rows:
        path = REPO_ROOT / str(row.get("path", ""))
        if not path.is_file():
            errors.append(f"missing:{row.get('path')}")
        elif path.stat().st_size != row.get("size_bytes"):
            errors.append(f"size:{row.get('path')}")
        elif sha256(path) != row.get("sha256"):
            errors.append(f"sha256:{row.get('path')}")
    return bool(rows) and not errors, len(rows), errors


def finite_curve(path: Path) -> tuple[bool, int]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    finite = True
    for row in rows:
        for value in row.values():
            if value not in {None, ""}:
                try:
                    finite &= math.isfinite(float(value.strip()))
                except ValueError:
                    pass
    return finite, len(rows)


def p0_unchanged() -> tuple[bool, str]:
    paths = (
        "smoke/c3/p0/logs/neu_det_vpeft_gpu_fp32_seed824",
        "smoke/c3/p0/logs/deeppcb_vpeft_gpu_fp32_seed824",
        "smoke/c3/p0/evidence/deeppcb_manifest.json",
        "smoke/c3/p0/evidence/neu_det_fewshot_manifest.json",
    )
    completed = subprocess.run(
        ["git", "diff", "--exit-code", P0_EVIDENCE_REF, "--", *paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0, completed.stdout + completed.stderr


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if output.is_absolute():
        raise ValueError("--output must be repository-relative")
    output = (REPO_ROOT / output).resolve()
    output.relative_to(P1_ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)

    p0_summary = json.loads((REPO_ROOT / "smoke/c3/p0/evidence/c3_p0_summary.json").read_text(encoding="utf-8"))
    p0_diff_ok, p0_diff = p0_unchanged()
    p0_checks = {
        "recorded_p0_pass": p0_summary.get("official_p0", {}).get("status") == "PASS",
        "frozen_run_evidence_unchanged_since_p0_evidence_ref": p0_diff_ok,
    }

    split_checks: dict[str, object] = {}
    for dataset in DATASETS:
        manifest_path = P1_ROOT / "evidence" / f"{dataset}_split_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_list = P1_ROOT / "config" / dataset / "train_seed824.txt"
        split_checks[dataset] = {
            "manifest_status_pass": manifest.get("status") == "PASS",
            "sample_count_100": manifest.get("selected_train_images") == 100,
            "membership_sha_valid": sha256(train_list) == manifest.get("train_list_sha256"),
            "no_overlap": not any(manifest.get("split_overlap", {}).values()),
            "distribution_pass": manifest.get("class_distribution", {}).get("status") == "PASS",
        }

    run_checks: dict[str, object] = {}
    resolved_by_run: dict[str, dict[str, object]] = {}
    for dataset in DATASETS:
        for method in METHODS:
            identifier = run_id(dataset, method, args.epochs)
            log_dir = P1_ROOT / "logs" / identifier
            missing = [name for name in REQUIRED_FILES if not (log_dir / name).is_file()]
            if missing:
                run_checks[identifier] = {"status": "FAIL", "missing": missing}
                continue
            metrics = json.loads((log_dir / "metrics.json").read_text(encoding="utf-8"))
            resolved = load_yaml(log_dir / "resolved_config.yaml")
            resolved_by_run[identifier] = resolved
            artifact_ok, artifact_count, artifact_errors = validate_artifacts(log_dir)
            curve_finite, curve_rows = finite_curve(log_dir / "learning_curve.csv")
            test = metrics.get("test", {})
            checks = {
                "runner_status_pass": metrics.get("status") == "PASS" and metrics.get("exit_code") == 0,
                "required_files_present": not missing,
                "stdout_complete": (log_dir / "stdout.log").stat().st_size > 0,
                "stderr_captured": (log_dir / "stderr.log").is_file(),
                f"curve_{args.epochs}_epochs_finite": curve_rows == args.epochs and curve_finite,
                "test_metrics_finite": all(
                    key in test and math.isfinite(float(test[key]))
                    for key in ("map50_95", "map50", "precision", "recall")
                ),
                "parameters_recorded": metrics.get("parameters", {}).get("trainable_parameters", 0) > 0
                and metrics.get("parameters", {}).get("total_parameters", 0) > 0,
                "gpu_memory_recorded": metrics.get("resources", {}).get("peak_gpu_memory_mib") is not None,
                "timing_recorded": metrics.get("timing", {}).get("training_seconds", 0) > 0
                and metrics.get("timing", {}).get("gpu_hours", 0) > 0,
                "checkpoint_recorded": metrics.get("checkpoint", {}).get("size_bytes", 0) > 0,
                "artifact_hashes_valid": artifact_ok,
                "no_numerical_recovery": metrics.get("numerical_recovery", {}).get("detected") is False,
            }
            if method == "vpeft":
                adapter = metrics.get("adapter", {})
                checks["vpeft_strict_peft"] = all(
                    (
                        resolved.get("lora_vpeft_strict") is True,
                        adapter.get("planner_status") in {"ACCEPT", "ADAPT"},
                        adapter.get("planner_backend") == "vpeft",
                        adapter.get("actual_backend") == "peft",
                        int(adapter.get("planned_targets", 0)) > 0,
                        int(adapter.get("applied_targets", 0)) > 0,
                        adapter.get("size_bytes", 0) > 0,
                    )
                )
            elif method == "full_sft":
                checks["method_boundary"] = (
                    int(resolved.get("lora_r", 0) or 0) == 0 and int(resolved.get("freeze", 0) or 0) == 0
                )
            else:
                module_rows = metrics.get("parameters", {}).get("top_level_modules", [])
                backbone_rows = [
                    row for row in module_rows if int(str(row.get("module", "model.-1")).split(".")[-1]) < 11
                ]
                head_rows = [row for row in module_rows if int(str(row.get("module", "model.-1")).split(".")[-1]) >= 11]
                checks["method_boundary"] = (
                    int(resolved.get("lora_r", 0) or 0) == 0
                    and int(resolved.get("freeze", 0) or 0) == 11
                    and len(backbone_rows) == 11
                    and all(row.get("frozen") is True for row in backbone_rows)
                    and bool(head_rows)
                    and all(int(row.get("trainable_parameters", 0)) > 0 for row in head_rows)
                )
            run_checks[identifier] = {
                "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
                "artifact_count": artifact_count,
                "artifact_errors": artifact_errors,
            }

    fairness_checks: dict[str, object] = {}
    if len(resolved_by_run) == 6:
        reference = resolved_by_run[run_id("neu_det", "full_sft", args.epochs)]
        fairness_checks["common_training_protocol"] = all(
            all(resolved.get(key) == reference.get(key) for key in FAIR_KEYS) for resolved in resolved_by_run.values()
        )
        fairness_checks["same_data_within_dataset"] = all(
            len({resolved_by_run[run_id(dataset, method, args.epochs)].get("data") for method in METHODS}) == 1
            for dataset in DATASETS
        )
        fairness_checks["distinct_method_only_settings"] = all(
            all(
                (
                    int(resolved_by_run[run_id(dataset, "full_sft", args.epochs)].get("lora_r", 0) or 0) == 0,
                    int(resolved_by_run[run_id(dataset, "full_sft", args.epochs)].get("freeze", 0) or 0) == 0,
                    int(resolved_by_run[run_id(dataset, "frozen_backbone", args.epochs)].get("lora_r", 0) or 0) == 0,
                    int(resolved_by_run[run_id(dataset, "frozen_backbone", args.epochs)].get("freeze", 0) or 0) == 11,
                    int(resolved_by_run[run_id(dataset, "vpeft", args.epochs)].get("lora_r", 0) or 0) == 8,
                )
            )
            for dataset in DATASETS
        )
    else:
        fairness_checks = {
            "common_training_protocol": False,
            "same_data_within_dataset": False,
            "distinct_method_only_settings": False,
        }

    all_split_pass = all(all(value.values()) for value in split_checks.values())
    all_runs_pass = len(run_checks) == 6 and all(value.get("status") == "PASS" for value in run_checks.values())
    overall = (
        "PASS"
        if all(p0_checks.values()) and all_split_pass and all_runs_pass and all(fairness_checks.values())
        else "FAIL"
    )
    payload = {
        "schema_version": 1,
        "scope": f"C3 P1 seed824 {args.epochs}-epoch stage: two datasets x three methods",
        "p0_evidence_ref": P0_EVIDENCE_REF,
        "p0_checks": p0_checks,
        "p0_diff": p0_diff,
        "split_checks": split_checks,
        "run_checks": run_checks,
        "fairness_checks": fairness_checks,
        "overall_status": overall,
        "multi_seed_status": "NOT_RUN_BY_SCOPE",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall_status": overall, "runs": len(run_checks)}, ensure_ascii=False))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
