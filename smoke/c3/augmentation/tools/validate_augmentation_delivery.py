#!/usr/bin/env python3
"""Strict validator for the C3 augmentation ablation and locked-test evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "smoke" / "c3" / "augmentation"
BASE_COMMIT = "bf6c7c508635dec0be849aedaa3eac5d88ed220d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_contract() -> dict:
    json_paths = sorted(ROOT.rglob("*.json"))
    yaml_paths = sorted(ROOT.rglob("*.yaml"))
    for path in json_paths:
        json.loads(path.read_text(encoding="utf-8"))
    for path in yaml_paths:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    return {"json": len(json_paths), "yaml": len(yaml_paths), "status": "PASS"}


def immutable_history_contract() -> dict:
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE_COMMIT, "--"], cwd=REPO_ROOT, text=True
    ).splitlines()
    allowed = (
        "smoke/c3/augmentation/",
        "smoke/c3/final/evidence/augmentation_revalidation_20260831.json",
        "tests/test_c3_augmentation_policy.py",
        "ultralytics/",
    )
    unexpected = [path for path in changed if not path.startswith(allowed)]
    require(not unexpected, f"Existing experiment history changed: {unexpected}")
    forbidden = [
        path
        for path in changed
        if path.startswith(("smoke/c3/p0/", "smoke/c3/p1/", "smoke/c3/p2/", "smoke/c3/completion/"))
    ]
    require(not forbidden, f"Historical P0/P1/P2/completion artifacts changed: {forbidden}")
    return {"base_commit": BASE_COMMIT, "changed_files": changed, "forbidden_changes": forbidden, "status": "PASS"}


def training_contract() -> dict:
    frozen = json.loads((ROOT / "results" / "frozen_selection.json").read_text(encoding="utf-8"))
    expected = 16
    expected += sum(18 for details in frozen["datasets"].values() if details["scaling_trigger_passed"])
    metric_paths = sorted((ROOT / "logs").glob("*/*/metrics.json"))
    require(len(metric_paths) == expected, f"Expected {expected} training runs, found {len(metric_paths)}")
    checkpoint_rows = []
    manifest_artifacts_verified = 0
    for metrics_path in metric_paths:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_id = payload["run_id"]
        require(payload["status"] == "PASS" and payload["exit_code"] == 0, f"Failed run: {run_id}")
        require(payload["test"] is None, f"Training record contains test results: {run_id}")
        require(all(payload["checks"].values()), f"Run checks failed: {run_id}")
        require(payload["parameters"]["trainable_parameters"] == 613_602, f"Wrong trainable count: {run_id}")
        require(payload["parameters"]["total_parameters"] == 2_772_770, f"Wrong total count: {run_id}")
        resolved = yaml.safe_load(metrics_path.with_name("resolved_config.yaml").read_text(encoding="utf-8"))
        require(
            resolved["seed"] == payload["seed"]
            and resolved["epochs"] == 100
            and resolved["batch"] == 8
            and resolved["imgsz"] == 640
            and resolved["amp"] is False,
            f"Resolved training protocol mismatch: {run_id}",
        )
        require("test" not in yaml.safe_load((REPO_ROOT / payload["split"]["data"]).read_text()), f"Test key: {run_id}")
        require(
            len(list(csv.DictReader(metrics_path.with_name("learning_curve.csv").open()))) == 100, f"Curve: {run_id}"
        )
        manifest = json.loads(metrics_path.with_name("artifact_manifest.json").read_text(encoding="utf-8"))
        require(manifest["run_id"] == run_id, f"Artifact manifest run mismatch: {run_id}")
        for artifact in manifest["artifacts"]:
            artifact_path = REPO_ROOT / artifact["path"]
            require(
                artifact_path.is_file()
                and artifact_path.stat().st_size == artifact["size_bytes"]
                and sha256(artifact_path) == artifact["sha256"],
                f"Artifact drift: {artifact['path']}",
            )
            manifest_artifacts_verified += 1
        for kind, row in payload["checkpoints"].items():
            path = REPO_ROOT / row["path"]
            require(path.is_file() and sha256(path) == row["sha256"], f"Checkpoint hash mismatch: {run_id}/{kind}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = checkpoint.get("model")
            train_args = checkpoint.get("train_args", {})
            require(model is not None, f"Checkpoint load has no model: {run_id}/{kind}")
            named_parameters = list(model.named_parameters())
            total_parameters = sum(parameter.numel() for _, parameter in named_parameters)
            trainable_parameters = sum(
                parameter.numel() for _, parameter in named_parameters if parameter.requires_grad
            )
            adapter_parameters = sum(
                parameter.numel()
                for name, parameter in named_parameters
                if any(marker in name.lower() for marker in ("lora_", "dora_", "magnitude_vector"))
            )
            require(total_parameters == 2_772_770, f"Checkpoint parameter mismatch: {run_id}/{kind}")
            expected_serialized_trainable = 613_602 if kind == "last_healthy" else 0
            require(
                trainable_parameters == expected_serialized_trainable,
                f"Checkpoint serialized trainable flags mismatch: {run_id}/{kind}",
            )
            require(adapter_parameters == 181_760, f"Checkpoint adapter mismatch: {run_id}/{kind}")
            require(
                train_args.get("seed") == payload["seed"]
                and train_args.get("epochs") == 100
                and train_args.get("batch") == 8
                and train_args.get("imgsz") == 640
                and train_args.get("data") == payload["split"]["data"],
                f"Checkpoint seed/epoch mismatch: {run_id}/{kind}",
            )
            checkpoint_rows.append(
                {
                    "run_id": run_id,
                    "kind": kind,
                    "sha256": row["sha256"],
                    "seed": train_args["seed"],
                    "configured_epochs": train_args["epochs"],
                    "stored_epoch": checkpoint.get("epoch"),
                    "total_parameters": total_parameters,
                    "trainable_parameters": trainable_parameters,
                    "adapter_parameters": adapter_parameters,
                    "load": "PASS",
                }
            )
            del checkpoint, model
    output = ROOT / "evidence" / "checkpoint_load_validation.json"
    output.write_text(
        json.dumps({"status": "PASS", "loaded": len(checkpoint_rows), "checkpoints": checkpoint_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "runs": len(metric_paths),
        "expected": expected,
        "manifest_artifacts_verified": manifest_artifacts_verified,
        "checkpoints_loaded": len(checkpoint_rows),
        "status": "PASS",
    }


def selection_and_test_contract() -> dict:
    initial = json.loads((ROOT / "results" / "initial_selection.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "results" / "frozen_selection.json").read_text(encoding="utf-8"))
    require(initial["test_access_before_selection"] is False, "Initial selection used test")
    require(frozen["test_access_before_selection"] is False, "Frozen selection used test")
    require(frozen["test_metrics_used_for_selection"] is False, "Test metrics selected policy")
    frozen_at = datetime.fromisoformat(frozen["frozen_at"])
    evaluation_paths = sorted((ROOT / "evaluations").glob("*.json"))
    expected = 0
    for details in frozen["datasets"].values():
        policies = len({"baseline", details["frozen_policy"]})
        expected += policies * 3
        if details["scaling_trigger_passed"]:
            expected += policies * 3 * 3
    require(len(evaluation_paths) == expected, f"Expected {expected} test evaluations, found {len(evaluation_paths)}")
    for path in evaluation_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload["status"] == "PASS" and payload["split"] == "locked_test", f"Bad evaluation: {path.name}")
        require(payload["selection_frozen_before_test"] is True, f"Unfrozen evaluation: {path.name}")
        environment = json.loads(
            (ROOT / "logs" / "locked_test" / path.stem / "environment.json").read_text(encoding="utf-8")
        )
        require(datetime.fromisoformat(environment["captured_utc"]) > frozen_at, f"Pre-freeze test: {path.name}")
        checkpoint = REPO_ROOT / payload["source_checkpoint"]["path"]
        require(sha256(checkpoint) == payload["source_checkpoint"]["sha256"], f"Evaluation hash: {path.name}")
        require(len(payload["per_class"]) == 6, f"Incomplete per-class metrics: {path.name}")
        require(all(math.isfinite(float(value)) for value in payload["overall"].values()), f"Nonfinite: {path.name}")
    return {
        "initial": initial["status"],
        "frozen": frozen["status"],
        "evaluations": len(evaluation_paths),
        "status": "PASS",
    }


def result_and_figure_contract() -> dict:
    required = [
        "initial_search.csv",
        "confirmation_runs.csv",
        "confirmation_summary.csv",
        "paired_validation_statistics.csv",
        "locked_test_runs.csv",
        "locked_test_summary.csv",
        "paired_test_statistics.csv",
        "historical_baseline_paired_statistics.csv",
        "per_class_test_runs.csv",
        "per_class_test_summary.csv",
        "per_class_paired_statistics.csv",
        "reference_comparison.csv",
        "scaling_comparison.csv",
    ]
    for name in required:
        path = ROOT / "results" / name
        require(path.is_file() and list(csv.DictReader(path.open(encoding="utf-8"))), f"Missing/empty CSV: {name}")
    manifest = json.loads((ROOT / "figures" / "figure_manifest.json").read_text(encoding="utf-8"))
    for name, details in manifest["figures"].items():
        figure = ROOT / "figures" / name
        source = REPO_ROOT / details["source"]
        require(figure.is_file() and figure.stat().st_size > 0, f"Missing figure: {name}")
        require(sha256(source) == details["source_sha256"], f"Figure source drift: {name}")
        source_rows = list(csv.DictReader(source.open(encoding="utf-8")))
        if "filter" in details:
            source_rows = [
                row
                for row in source_rows
                if all(str(row[key]) == str(value) for key, value in details["filter"].items())
            ]
        require(source_rows == details["plotted_rows"], f"Plotted numbers differ from CSV: {name}")
    return {"csv_tables": len(required), "figures": len(manifest["figures"]), "status": "PASS"}


def hygiene_and_links_contract() -> dict:
    roots = [ROOT / name for name in ("config", "docs", "evidence", "evaluations", "failures", "logs", "results")]
    patterns = {
        "absolute_user_path": re.compile(r"/ho" r"me/[A-Za-z0-9._-]+/"),
        "openai_token": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        "github_token": re.compile(r"gh[opsu]_[A-Za-z0-9]{20,}"),
        "bearer_token": re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}", re.IGNORECASE),
    }
    findings = []
    scanned = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() in {".png", ".pt", ".bin", ".safetensors"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            scanned += 1
            findings.extend(
                {"path": path.relative_to(REPO_ROOT).as_posix(), "kind": kind}
                for kind, pattern in patterns.items()
                if pattern.search(text)
            )
    require(not findings, f"Privacy/token findings: {findings[:10]}")
    links = []
    missing = []
    for document in sorted((ROOT / "docs").glob("*.md")):
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            resolved = (document.parent / target.split("#", 1)[0]).resolve()
            links.append({"document": document.name, "target": target})
            if not resolved.exists():
                missing.append(links[-1])
    require(not missing, f"Broken documentation links: {missing}")
    return {"text_files_scanned": scanned, "privacy_findings": findings, "links_checked": len(links), "status": "PASS"}


def main() -> int:
    output = ROOT / "evidence" / "augmentation_delivery_validation.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite augmentation validation evidence")
    checks = {
        "parse": parse_contract(),
        "immutable_history": immutable_history_contract(),
        "training": training_contract(),
        "selection_and_locked_test": selection_and_test_contract(),
        "results_and_figures": result_and_figure_contract(),
        "hygiene_and_links": hygiene_and_links_contract(),
    }
    payload = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "checks": checks,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": list(checks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
