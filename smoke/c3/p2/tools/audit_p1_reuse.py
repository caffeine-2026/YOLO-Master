#!/usr/bin/env python3
"""Prove that the P2 100-image seed824 cells may reuse immutable P1 runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
P2_ROOT = REPO_ROOT / "smoke" / "c3" / "p2"
DATASETS = (("neu", "neu_det", "NEU-DET"), ("deeppcb", "deeppcb", "DeepPCB"))
METHODS = (("full_sft", "full"), ("frozen_backbone", "frozen"), ("vpeft", "vpeft"))
LOCKED_KEYS = (
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
    "freeze",
    "lora_r",
)
VPEFT_KEYS = (
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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    dataset_audits = {}
    run_audits = []
    for tag, p1_tag, label in DATASETS:
        p1_list = P1_ROOT / "config" / p1_tag / "train_seed824.txt"
        p2_list = P2_ROOT / "config" / "splits" / f"{tag}_100_seed824.txt"
        p1_data = yaml.safe_load((P1_ROOT / "config" / p1_tag / "dataset.yaml").read_text(encoding="utf-8"))
        p2_data = yaml.safe_load((P2_ROOT / "config" / "splits" / f"{tag}_100.yaml").read_text(encoding="utf-8"))
        checks = {
            "train_list_byte_identical": p1_list.read_bytes() == p2_list.read_bytes(),
            "train_list_sha256_identical": sha256(p1_list) == sha256(p2_list),
            "train_membership_identical": set(p1_list.read_text().splitlines())
            == set(p2_list.read_text().splitlines()),
            "validation_identical": p1_data["val"] == p2_data["val"],
            "test_identical": p1_data["test"] == p2_data["test"],
            "class_mapping_identical": p1_data["names"] == p2_data["names"],
        }
        dataset_audits[label] = {
            "checks": checks,
            "p1_train_list_sha256": sha256(p1_list),
            "p2_train_list_sha256": sha256(p2_list),
            "status": "PASS" if all(checks.values()) else "FAIL",
        }
        for method, method_tag in METHODS:
            p2_config = yaml.safe_load(
                (P2_ROOT / "config" / "runs" / tag / f"{method}.yaml").read_text(encoding="utf-8")
            )
            p1_run = f"{tag}_{method_tag}_seed824_e100"
            p1_log = P1_ROOT / "logs" / p1_run
            p1_resolved = yaml.safe_load((p1_log / "resolved_config.yaml").read_text(encoding="utf-8"))
            p1_metrics = json.loads((p1_log / "metrics.json").read_text(encoding="utf-8"))
            comparison_keys = LOCKED_KEYS + VPEFT_KEYS if method == "vpeft" else LOCKED_KEYS
            key_checks = {key: p2_config.get(key) == p1_resolved.get(key) for key in comparison_keys}
            checks = {
                "p1_status_pass": p1_metrics.get("status") == "PASS" and p1_metrics.get("exit_code") == 0,
                "p1_all_checks_pass": all(p1_metrics.get("checks", {}).values()),
                "locked_protocol_identical": all(key_checks.values()),
                "split_identical": dataset_audits[label]["status"] == "PASS",
            }
            run_audits.append(
                {
                    "dataset": label,
                    "sample_size": 100,
                    "method": method,
                    "seed": 824,
                    "p1_run_id": p1_run,
                    "p1_metrics": (p1_log / "metrics.json").relative_to(REPO_ROOT).as_posix(),
                    "p1_metrics_sha256": sha256(p1_log / "metrics.json"),
                    "locked_key_checks": key_checks,
                    "checks": checks,
                    "status": "PASS" if all(checks.values()) else "FAIL",
                }
            )
    evaluator = P1_ROOT / "scripts" / "evaluate_p1.py"
    overall = all(row["status"] == "PASS" for row in dataset_audits.values()) and all(
        row["status"] == "PASS" for row in run_audits
    )
    payload = {
        "schema_version": 1,
        "decision": "REUSE_P1_100_SEED824" if overall else "RERUN_REQUIRED",
        "seed": 824,
        "reused_cells": 6 if overall else 0,
        "dataset_audits": dataset_audits,
        "run_audits": run_audits,
        "evaluation_code": evaluator.relative_to(REPO_ROOT).as_posix(),
        "evaluation_code_sha256": sha256(evaluator),
        "status": "PASS" if overall else "FAIL",
    }
    output = P2_ROOT / "evidence" / "p1_100_reuse_audit.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"P1_100_REUSE={payload['decision']}")
    print(f"REUSED_CELLS={payload['reused_cells']}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
