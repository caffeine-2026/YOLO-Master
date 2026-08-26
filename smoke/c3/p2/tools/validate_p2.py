#!/usr/bin/env python3
"""Independently validate the C3 P2 seed824 scaling gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
P2_ROOT = REPO_ROOT / "smoke" / "c3" / "p2"
SIZES = (10, 50, 100, 500)
DATASETS = ("neu", "deeppcb")
METHODS = ("full", "frozen", "vpeft")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_manifest(path: Path) -> tuple[bool, int]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for row in manifest["artifacts"]:
        artifact = REPO_ROOT / row["path"]
        if not artifact.is_file() or artifact.stat().st_size != row["size_bytes"] or sha256(artifact) != row["sha256"]:
            return False, manifest["artifact_count"]
    return True, manifest["artifact_count"]


def main() -> int:
    protocol = yaml.safe_load((P2_ROOT / "config" / "protocol.yaml").read_text(encoding="utf-8"))
    reuse = json.loads((P2_ROOT / "evidence" / "p1_100_reuse_audit.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((P2_ROOT / "evidence" / "seed824_run_manifest.json").read_text(encoding="utf-8"))
    scaling_rows = list(
        csv.DictReader((P2_ROOT / "results" / "scaling_seed824.csv").open(encoding="utf-8", newline=""))
    )
    tradeoff_rows = list(
        csv.DictReader((P2_ROOT / "results" / "scaling_tradeoff_seed824.csv").open(encoding="utf-8", newline=""))
    )

    split_checks = {}
    for dataset in DATASETS:
        manifest = json.loads(
            (P2_ROOT / "evidence" / f"{dataset}_scaling_split_manifest.json").read_text(encoding="utf-8")
        )
        nested = manifest["nested_verification"]
        split_checks[dataset] = {
            "manifest_status_pass": manifest["status"] == "PASS",
            "strict_nesting": all(nested.values()),
            "all_sizes_cover_all_classes": all(
                manifest["splits"][str(size)]["distribution"]["all_classes_covered"] for size in SIZES
            ),
            "p1_100_split_reuse_eligible": manifest["p1_100_reuse_audit"]["eligible_by_split"],
            "fixed_val_test": bool(manifest["fixed_evaluation"]["val_membership_sha256"])
            and bool(manifest["fixed_evaluation"]["test_membership_sha256"]),
        }

    new_runs = []
    vpeft_checks = []
    artifact_counts = 0
    for dataset in DATASETS:
        for size in (10, 50, 500):
            for method in METHODS:
                run_id = f"{dataset}_{size}_{method}_seed824"
                log_dir = P2_ROOT / "logs" / run_id
                metrics = json.loads((log_dir / "metrics.json").read_text(encoding="utf-8"))
                with (log_dir / "learning_curve.csv").open(encoding="utf-8", newline="") as stream:
                    curve = list(csv.DictReader(stream))
                manifest_ok, artifact_count = verify_artifact_manifest(log_dir / "artifact_manifest.json")
                artifact_counts += artifact_count
                checks = {
                    "status_pass": metrics.get("status") == "PASS" and metrics.get("exit_code") == 0,
                    "runner_checks_pass": all(metrics.get("checks", {}).values()),
                    "epochs_complete": len(curve) == 100,
                    "curve_finite": all(
                        math.isfinite(float(value))
                        for row in curve
                        for key, value in row.items()
                        if key != "epoch" and value not in (None, "")
                    ),
                    "artifact_manifest_verified": manifest_ok,
                    "sample_size_matches": metrics.get("sample_size") == size,
                    "seed824": metrics.get("seed") == 824,
                }
                new_runs.append(
                    {"run_id": run_id, "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}
                )
                if method == "vpeft":
                    adapter = metrics["adapter"]
                    strict = {
                        "planner_status": adapter.get("planner_status") in {"ACCEPT", "ADAPT"},
                        "planner_backend": adapter.get("planner_backend") == "vpeft",
                        "actual_backend": adapter.get("actual_backend") == "peft",
                        "planned_targets": int(adapter.get("planned_targets") or 0) > 0,
                        "applied_targets": int(adapter.get("applied_targets") or 0) > 0,
                        "adapter_params": int(metrics["parameters"].get("adapter_parameters") or 0) > 0,
                        "adapter_export": int(adapter.get("size_bytes") or 0) > 0,
                        "no_silent_fallback": metrics["checks"].get("strict_vpeft") is True,
                    }
                    vpeft_checks.append(
                        {"run_id": run_id, "checks": strict, "status": "PASS" if all(strict.values()) else "FAIL"}
                    )

    expected_plots = [
        P2_ROOT / "visualizations" / f"{dataset}_{suffix}.png"
        for dataset in DATASETS
        for suffix in (
            "scaling_map5095",
            "scaling_map50",
            "accuracy_retention",
            "accuracy_vs_params",
            "peak_gpu_memory",
            "training_time",
            "gpu_hours",
        )
    ]
    checks = {
        "protocol_seed824_only": protocol["training"]["seeds_this_stage"] == [824]
        and protocol["training"]["prohibited_seeds_this_stage"] == [825, 826],
        "p1_history_unchanged": subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "smoke/c3/p1"], cwd=REPO_ROOT, check=False
        ).returncode
        == 0,
        "split_manifests_pass": all(all(row.values()) for row in split_checks.values()),
        "p1_reuse_6_of_6": reuse["status"] == "PASS" and reuse["reused_cells"] == 6,
        "new_runs_18_of_18": len(new_runs) == 18 and all(row["status"] == "PASS" for row in new_runs),
        "vpeft_strict_6_of_6": len(vpeft_checks) == 6 and all(row["status"] == "PASS" for row in vpeft_checks),
        "scaling_cells_24": len(scaling_rows) == 24
        and {(row["dataset"], row["sample_size"], row["method"]) for row in scaling_rows}.__len__() == 24,
        "tradeoff_cells_24": len(tradeoff_rows) == 24,
        "plots_14_of_14": all(path.is_file() and path.stat().st_size > 0 for path in expected_plots),
        "analysis_present": (P2_ROOT / "docs" / "SCALING_ANALYSIS_SEED824.md").is_file(),
        "report_present": (P2_ROOT / "docs" / "C3_P2_REPORT.md").is_file(),
        "informative_curves": run_manifest.get("informative_nonconstant_curves") is True,
        "seed825_826_not_run": run_manifest.get("seed825_826_run_count") == 0
        and not any((P2_ROOT / "logs").glob("*_seed825"))
        and not any((P2_ROOT / "logs").glob("*_seed826")),
        "multiseed_ready": run_manifest.get("multiseed_ready") is True,
        "p2_pass_withheld": run_manifest.get("overall_c3_p2") == "IN_PROGRESS",
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "schema_version": 1,
        "stage": "seed824_scaling_gate",
        "checks": checks,
        "split_checks": split_checks,
        "new_runs": new_runs,
        "vpeft_checks": vpeft_checks,
        "verified_artifact_records": artifact_counts,
        "seed824_stage_status": status,
        "MULTISEED_READY": "YES" if status == "PASS" else "NO",
        "Overall C3 P2": "IN_PROGRESS",
    }
    output = P2_ROOT / "evidence" / "p2_seed824_validation.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"P2_SEED824_STAGE={status}")
    print(f"MULTISEED_READY={payload['MULTISEED_READY']}")
    print("OVERALL_C3_P2=IN_PROGRESS")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
