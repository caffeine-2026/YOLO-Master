#!/usr/bin/env python3
"""Independently validate the final 72-cell C3 P2 multi-seed delivery."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
P2_ROOT = REPO_ROOT / "smoke" / "c3" / "p2"
P1_IMMUTABLE_PATHS = (
    "smoke/c3/p1/logs",
    "smoke/c3/p1/artifacts",
    "smoke/c3/p1/evidence/deeppcb_split_manifest.json",
    "smoke/c3/p1/evidence/neu_det_split_manifest.json",
    "smoke/c3/p1/evidence/multiseed_manifest.json",
    "smoke/c3/p1/results/raw_metrics_multiseed.csv",
)
DATASETS = ("neu", "deeppcb")
SIZES = (10, 50, 100, 500)
SEEDS = (824, 825, 826)
METHODS = (("full", "full_sft"), ("frozen", "frozen_backbone"), ("vpeft", "vpeft"))
LOCKED_KEYS = (
    "model",
    "pretrained",
    "epochs",
    "batch",
    "imgsz",
    "workers",
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
    "lora_type",
    "lora_use_rslora",
    "lora_gradient_checkpointing",
    "lora_few_shot_mode",
    "lora_few_shot_adaptive_rank",
    "lora_lr_mult",
    "lora_save_adapters",
    "lora_planner_enabled",
    "lora_planner_backend",
    "lora_planner_solver",
    "lora_adapter_budget",
    "lora_vpeft_strict",
    "lora_exclude_modules",
)


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(path: Path) -> bool:
    """Verify all files recorded by an artifact manifest."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return all(
        (REPO_ROOT / row["path"]).is_file()
        and (REPO_ROOT / row["path"]).stat().st_size == row["size_bytes"]
        and sha256(REPO_ROOT / row["path"]) == row["sha256"]
        for row in manifest["artifacts"]
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run_id(dataset: str, size: int, method: str, seed: int) -> tuple[Path, str]:
    """Resolve a final matrix run directory."""
    if size == 100:
        name = f"{dataset}_{method}_seed{seed}_e100"
        return P1_ROOT / "logs" / name, name
    suffix = "" if seed == 824 else "_e100"
    name = f"{dataset}_{size}_{method}_seed{seed}{suffix}"
    return P2_ROOT / "logs" / name, name


def main() -> int:
    """Run the independent final gate and write structured evidence."""
    protocol = yaml.safe_load((P2_ROOT / "config" / "protocol.yaml").read_text(encoding="utf-8"))
    p1_validation = json.loads((P1_ROOT / "evidence" / "p1_final_validation.json").read_text())
    p1_seed824_audit = json.loads((P2_ROOT / "evidence" / "p1_100_reuse_audit.json").read_text())
    final_manifest = json.loads((P2_ROOT / "evidence" / "p2_multiseed_manifest.json").read_text())
    all_rows = read_csv(P2_ROOT / "results" / "p2_all_runs.csv")
    summary = read_csv(P2_ROOT / "results" / "p2_summary.csv")
    retention = read_csv(P2_ROOT / "results" / "retention_multiseed.csv")
    paired = read_csv(P2_ROOT / "results" / "paired_analysis.csv")
    characteristics = read_csv(P2_ROOT / "results" / "dataset_characteristics.csv")

    split_checks = {}
    for dataset in DATASETS:
        path = P2_ROOT / "evidence" / f"{dataset}_scaling_split_manifest.json"
        manifest = json.loads(path.read_text())
        hashes = {}
        for size in SIZES:
            split = manifest["splits"][str(size)]
            hashes[str(size)] = sha256(REPO_ROOT / split["train_list"]) == split["train_list_sha256"]
        split_checks[dataset] = {
            "manifest_pass": manifest["status"] == "PASS",
            "split_seed_824": manifest["split_seed"] == 824,
            "strict_nested": all(manifest["nested_verification"].values()),
            "all_train_hashes_match": all(hashes.values()),
            "fixed_val_test": bool(manifest["fixed_evaluation"]["val_membership_sha256"])
            and bool(manifest["fixed_evaluation"]["test_membership_sha256"]),
            "all_classes_covered": all(
                manifest["splits"][str(size)]["distribution"]["all_classes_covered"] for size in SIZES
            ),
        }

    run_checks = []
    vpeft_checks = []
    canonical_configs: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for dataset in DATASETS:
        for size in SIZES:
            for method, method_internal in METHODS:
                for seed in SEEDS:
                    log_dir, name = run_id(dataset, size, method, seed)
                    metrics = json.loads((log_dir / "metrics.json").read_text())
                    manifest_ok = verify_manifest(log_dir / "artifact_manifest.json")
                    checks = {
                        "status_pass": metrics.get("status") == "PASS" and metrics.get("exit_code") == 0,
                        "runner_checks_pass": all(metrics.get("checks", {}).values()),
                        "seed_matches": int(metrics["seed"]) == seed,
                        "artifact_manifest_verified": manifest_ok,
                        "metrics_finite": all(
                            math.isfinite(float(metrics["test"][key]))
                            for key in ("map50_95", "map50", "precision", "recall")
                        ),
                    }
                    run_checks.append(
                        {"run_id": name, "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}
                    )
                    if method == "vpeft":
                        adapter = metrics["adapter"]
                        strict = {
                            "planner_status": adapter.get("planner_status") in {"ACCEPT", "ADAPT"},
                            "planner_backend": adapter.get("planner_backend") == "vpeft",
                            "actual_backend": adapter.get("actual_backend") == "peft",
                            "planned_targets": int(adapter.get("planned_targets") or 0) > 0,
                            "applied_targets": int(adapter.get("applied_targets") or 0) > 0,
                            "adapter_export": int(adapter.get("size_bytes") or 0) > 0,
                            "no_silent_fallback": metrics["checks"].get("strict_vpeft") is True,
                        }
                        vpeft_checks.append(
                            {"run_id": name, "checks": strict, "status": "PASS" if all(strict.values()) else "FAIL"}
                        )
                    if size != 100:
                        resolved = yaml.safe_load((log_dir / "resolved_config.yaml").read_text())
                        canonical_configs.setdefault((dataset, size, method_internal), []).append(
                            {key: resolved.get(key) for key in LOCKED_KEYS}
                        )

    config_fairness = {
        "/".join(map(str, key)): len(configs) == 3 and configs[0] == configs[1] == configs[2]
        for key, configs in canonical_configs.items()
    }
    scheduler = [json.loads(line) for line in (P2_ROOT / "logs" / "multiseed_scheduler.jsonl").read_text().splitlines()]
    finish = next(row for row in scheduler if row["event"] == "suite_finish")
    expected_figures = [
        P2_ROOT / "visualizations" / "final" / f"{dataset}_{suffix}_multiseed.png"
        for dataset in DATASETS
        for suffix in ("map5095", "map50", "retention", "accuracy_vs_params", "memory", "training_time", "gpu_hours")
    ]
    expected_cells = {
        (dataset, str(size), method, str(seed))
        for _, dataset in (("neu", "NEU-DET"), ("deeppcb", "DeepPCB"))
        for size in SIZES
        for method in ("Full-SFT", "Frozen Backbone", "V-PEFT")
        for seed in SEEDS
    }
    found_cells = {(r["dataset"], r["sample_size"], r["method"], r["seed"]) for r in all_rows}
    source_counts = {
        source: sum(row["source"] == source for row in all_rows)
        for source in ("immutable_p2_seed824", "new_p2_multiseed", "reused_p1_100_multiseed")
    }
    checks = {
        "nested_split_integrity": all(all(row.values()) for row in split_checks.values()),
        "p1_100_reuse_integrity": p1_validation["P1 status"] == "PASS"
        and p1_validation["protocol_fairness"]["status"] == "PASS"
        and p1_validation["protocol_fairness"]["checks"]["only_seed_varies_within_dataset_method"] is True
        and p1_validation["protocol_fairness"]["checks"]["fixed_sample_ids"] is True
        and p1_seed824_audit["status"] == "PASS"
        and p1_seed824_audit["reused_cells"] == 6,
        "all_required_cells": len(all_rows) == 72 and found_cells == expected_cells,
        "source_accounting": source_counts
        == {"immutable_p2_seed824": 18, "new_p2_multiseed": 36, "reused_p1_100_multiseed": 18},
        "new_run_execution": finish["completed"] == 36 and finish["failures"] == [],
        "protocol_fairness": all(config_fairness.values())
        and protocol["training"]["final_seeds"] == [824, 825, 826]
        and protocol["training"]["new_seeds_this_stage"] == [825, 826],
        "multi_seed_completeness": len(run_checks) == 72 and all(row["status"] == "PASS" for row in run_checks),
        "vpeft_strict_24_of_24": len(vpeft_checks) == 24 and all(row["status"] == "PASS" for row in vpeft_checks),
        "statistics_completeness": len(summary) == 24
        and len(retention) == 8
        and len(paired) == 24
        and len(characteristics) == 8,
        "artifact_integrity": all(row["artifact_manifest_verified"] for row in final_manifest["runs"])
        and final_manifest["matrix_cells"] == 72
        and final_manifest["Overall C3 P2"] == "PASS"
        and final_manifest["final_protocol"]["sha256"] == sha256(P2_ROOT / "config" / "protocol.yaml"),
        "final_figures_14_of_14": all(path.is_file() and path.stat().st_size > 0 for path in expected_figures),
        "final_report": "`Overall C3 P2 = PASS`" in (P2_ROOT / "docs" / "C3_P2_REPORT.md").read_text(encoding="utf-8"),
        "p1_history_unmodified": subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *P1_IMMUTABLE_PATHS], cwd=REPO_ROOT, check=False
        ).returncode
        == 0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "schema_version": 1,
        "stage": "final_multiseed_scaling",
        "checks": checks,
        "split_checks": split_checks,
        "config_fairness": config_fairness,
        "source_counts": source_counts,
        "run_validation": {"expected": 72, "pass": sum(r["status"] == "PASS" for r in run_checks)},
        "vpeft_validation": {"expected": 24, "pass": sum(r["status"] == "PASS" for r in vpeft_checks)},
        "statistics": {"summary": len(summary), "retention": len(retention), "paired": len(paired)},
        "artifact_integrity": {
            "manifest_rows": len(final_manifest["runs"]),
            "status": "PASS" if checks["artifact_integrity"] else "FAIL",
        },
        "P2 status": status,
        "Overall C3 P2": status,
    }
    output = P2_ROOT / "evidence" / "p2_final_validation.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"P2_FINAL_VALIDATION={status}")
    print(f"OVERALL_C3_P2={status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
