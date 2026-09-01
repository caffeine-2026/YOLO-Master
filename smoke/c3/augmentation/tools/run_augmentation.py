#!/usr/bin/env python3
"""Run one immutable validation-only V-PEFT augmentation experiment."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smoke.c3.p1.scripts import run_p1 as common

P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
P2_ROOT = REPO_ROOT / "smoke" / "c3" / "p2"
ROOT = REPO_ROOT / "smoke" / "c3" / "augmentation"
PROTOCOL_PATH = ROOT / "config" / "augmentation_protocol.yaml"
DATASETS = ("neu", "deeppcb")
POLICIES = ("baseline", "mild", "medium", "strong")
SEEDS = (824, 825, 826)
EXPECTED_TRAINABLE = 613_602
EXPECTED_TOTAL = 2_772_770
AUGMENTATION_KEYS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "industrial_contrast",
    "industrial_blur_prob",
    "industrial_blur_sigma",
    "industrial_noise_prob",
    "industrial_noise_std",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "close_mosaic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("search", "confirm", "scaling"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--sample-size", type=int, choices=(10, 50, 100, 500), default=100)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def protocol() -> dict:
    payload = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("base_commit") != "bf6c7c508635dec0be849aedaa3eac5d88ed220d":
        raise ValueError("Augmentation protocol base commit drifted")
    return payload


def assert_gpu_idle(device: str) -> dict:
    command = [
        "nvidia-smi",
        f"--id={device}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"GPU preflight failed for device {device}: {completed.stderr.strip()}")
    processes = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if processes:
        raise RuntimeError(f"GPU {device} is occupied; refusing to launch: {processes}")
    return {"device": device, "query": "nvidia-smi compute applications", "processes": processes, "idle": True}


def selected_policy(dataset: str, stage: str) -> str:
    filename = "initial_selection.json" if stage == "initial" else "frozen_selection.json"
    path = ROOT / "results" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing locked selection: {common.relative(path)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_access_before_selection") is not False:
        raise ValueError("Selection evidence does not prohibit test access")
    key = "top_augmentation" if stage == "initial" else "frozen_policy"
    return str(payload["datasets"][dataset][key])


def validate_phase(args: argparse.Namespace) -> None:
    if args.phase == "search" and (args.sample_size != 100 or args.seed != 824):
        raise ValueError("Initial search is locked to 100-shot seed824")
    if args.phase == "confirm":
        if args.sample_size != 100 or args.seed == 824:
            raise ValueError("Confirmation runs add only seeds 825 and 826 at 100-shot")
        allowed = {"baseline", selected_policy(args.dataset, "initial")}
        if args.policy not in allowed:
            raise ValueError(f"Confirmation policy must be one of {sorted(allowed)}")
    if args.phase == "scaling":
        frozen = selected_policy(args.dataset, "frozen")
        if args.sample_size == 100 or args.policy not in {"baseline", frozen}:
            raise ValueError("Scaling adds non-100-shot baseline and frozen-policy runs only")


def validation_data(dataset: str, sample_size: int) -> Path:
    path = ROOT / "config" / "data" / f"{dataset}_{sample_size}_validation_only.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation-only data config: {common.relative(path)}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "test" in data:
        raise ValueError("Validation-only data config must omit the test key")
    return path


def main() -> int:
    args = parse_args()
    validate_phase(args)
    locked = protocol()
    values = {**locked["common_disabled_augmentations"], **locked["policies"][args.dataset][args.policy]}
    strength = int(values.pop("strength"))
    missing = set(AUGMENTATION_KEYS) - set(values)
    if missing:
        raise ValueError(f"Policy is missing augmentation keys: {sorted(missing)}")

    run_id = f"{args.phase}_{args.dataset}_{args.sample_size}_{args.policy}_seed{args.seed}_e100"
    log_dir = ROOT / "logs" / args.phase / run_id
    artifact_dir = ROOT / "artifacts" / args.phase / run_id
    if log_dir.exists() or artifact_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable run: {run_id}")
    log_dir.mkdir(parents=True)
    gpu_preflight = assert_gpu_idle(args.device)

    source_config = P2_ROOT / "config" / "runs" / args.dataset / "vpeft.yaml"
    data = validation_data(args.dataset, args.sample_size)
    train_list = P2_ROOT / "config" / "splits" / f"{args.dataset}_{args.sample_size}_seed824.txt"
    for required in (source_config, data, train_list):
        if not required.is_file():
            raise FileNotFoundError(common.relative(required))
    source = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if int(source["epochs"]) != 100 or bool(source["amp"]):
        raise ValueError("Source protocol drifted from 100 epochs and amp=false")

    yolo = Path(sys.executable).with_name("yolo")
    overrides = [f"{key}={values[key]}" for key in AUGMENTATION_KEYS]
    train_command = [
        str(yolo),
        "train",
        f"cfg={common.relative(source_config)}",
        f"data={common.relative(data)}",
        f"device={args.device}",
        f"seed={args.seed}",
        f"name={run_id}",
        f"save_dir={common.relative(artifact_dir)}",
        "lora_head_train_policy=full",
        *overrides,
        "exist_ok=False",
    ]
    (log_dir / "command.txt").write_text(
        "# validation-only training; data config has no test key\n"
        + " ".join(["yolo", *train_command[1:]])
        + "\n\n# locked test evaluation prohibited until frozen_selection.json exists\n",
        encoding="utf-8",
    )
    environment = common.environment_evidence(args.device)
    environment["gpu_preflight"] = gpu_preflight
    environment["test_access"] = False
    common.json_write(log_dir / "environment.json", environment)

    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        exit_code, training_seconds = common.run_captured(
            train_command,
            log_dir / "stdout.log",
            log_dir / "stderr.log",
            args.device,
            log_dir / "resource_samples.csv",
        )
        timing = {
            "schema_version": 1,
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "training_seconds": round(training_seconds, 3),
            "gpu_hours": round(training_seconds / 3600, 8),
            "training_exit_code": exit_code,
        }
        common.json_write(log_dir / "timing.json", timing)
        resources = common.summarize_samples(log_dir / "resource_samples.csv")
        stdout_text = (log_dir / "stdout.log").read_text(encoding="utf-8")
        resources["epoch_reported_peak_gpu_memory_mib"] = common.epoch_gpu_peak_mib(stdout_text)
        resources["peak_gpu_memory_mib"] = (
            resources["epoch_reported_peak_gpu_memory_mib"] or resources["peak_process_tree_gpu_memory_mib"]
        )
        common.json_write(log_dir / "resource_usage.json", resources)
        resolved_source = artifact_dir / "args.yaml"
        if resolved_source.is_file():
            (log_dir / "resolved_config.yaml").write_text(
                common.clean_text(resolved_source.read_text(encoding="utf-8")), encoding="utf-8"
            )
        if exit_code != 0:
            common.json_write(
                log_dir / "metrics.json",
                {"schema_version": 1, "run_id": run_id, "status": "FAILED", "exit_code": exit_code},
            )
            return exit_code

        curve_source = artifact_dir / "results.csv"
        if not curve_source.is_file():
            raise FileNotFoundError("Training completed without results.csv")
        shutil.copy2(curve_source, log_dir / "learning_curve.csv")
        curve_rows, curve_finite = common.read_learning_curve(log_dir / "learning_curve.csv")
        map5095 = [float(row["metrics/mAP50-95(B)"]) for row in curve_rows]
        map50 = [float(row["metrics/mAP50(B)"]) for row in curve_rows]
        precision = [float(row["metrics/precision(B)"]) for row in curve_rows]
        recall = [float(row["metrics/recall(B)"]) for row in curve_rows]
        best_index = max(range(len(map5095)), key=map5095.__getitem__)
        checkpoints = {name: artifact_dir / "weights" / f"{name}.pt" for name in ("best", "last", "last_healthy")}
        for checkpoint in checkpoints.values():
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {common.relative(checkpoint)}")
        parameters, model = common.checkpoint_parameters(checkpoints["last_healthy"])
        best_parameters, _ = common.checkpoint_parameters(checkpoints["best"])
        adapter = common.adapter_evidence(model)
        adapter_dir = artifact_dir / "lora_adapter"
        common.export_adapter(checkpoints["last_healthy"], adapter_dir)
        adapter_size = sum(path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file())
        common.json_write(
            log_dir / "vpeft_runtime_metadata.json",
            {
                "backend": adapter["actual_backend"],
                "runtime_metadata": getattr(model, "lora_runtime_metadata", {}) or {},
            },
        )
        resolved = yaml.safe_load((log_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        stderr_text = (log_dir / "stderr.log").read_text(encoding="utf-8")
        recovery = re.findall(
            r"NaN recovery model|Loss NaN/Inf|Fitness NaN/Inf|Gradient NaN/Inf|EMA NaN/Inf|automatic numerical recovery",
            stdout_text + "\n" + stderr_text,
            flags=re.IGNORECASE,
        )
        checks = {
            "exit_code_zero": exit_code == 0,
            "requested_cuda_gpu_used": f"CUDA:{args.device} (NVIDIA GeForce RTX 4090" in stdout_text,
            "complete_finite_learning_curve": len(curve_rows) == 100 and curve_finite,
            "no_numerical_recovery": not recovery,
            "checkpoint_load_succeeded": True,
            "checkpoint_files_present": all(path.is_file() for path in checkpoints.values()),
            "accuracy_first_parameter_count": parameters["trainable_parameters"] == EXPECTED_TRAINABLE
            and parameters["total_parameters"] == EXPECTED_TOTAL,
            "protocol_locked": all(
                (
                    resolved.get("epochs") == 100,
                    resolved.get("batch") == 8,
                    resolved.get("imgsz") == 640,
                    resolved.get("seed") == args.seed,
                    resolved.get("optimizer") == "AdamW",
                    resolved.get("lr0") == 0.001,
                    resolved.get("weight_decay") == 0.0005,
                    resolved.get("amp") is False,
                    resolved.get("lora_adapter_budget") == 2_100_000,
                    resolved.get("lora_head_train_policy") == "full",
                    all(resolved.get(key) == value for key, value in values.items()),
                )
            ),
            "strict_vpeft": all(
                (
                    adapter["planner_status"] in {"ACCEPT", "ADAPT"},
                    adapter["planner_backend"] == "vpeft",
                    adapter["actual_backend"] == "peft",
                    int(adapter["planned_targets"] or 0) > 0,
                    int(adapter["applied_targets"] or 0) > 0,
                )
            ),
            "validation_data_omits_test": "test" not in yaml.safe_load(data.read_text(encoding="utf-8")),
            "test_not_accessed": not (log_dir / "test_metrics.json").exists()
            and not (artifact_dir / "test_eval").exists(),
        }
        status = "PASS" if all(checks.values()) else "FAILED"
        metrics = {
            "schema_version": 1,
            "stage": f"c3_augmentation_{args.phase}",
            "run_id": run_id,
            "dataset": args.dataset,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "policy": {"id": args.policy, "strength": strength, **values},
            "status": status,
            "exit_code": exit_code,
            "validation": {
                "best_epoch": best_index + 1,
                "best_map50_95": map5095[best_index],
                "best_map50": map50[best_index],
                "best_precision": precision[best_index],
                "best_recall": recall[best_index],
                "last_map50_95": map5095[-1],
                "last_map50": map50[-1],
            },
            "test": None,
            "parameters": parameters,
            "resources": resources,
            "timing": timing,
            "checkpoints": {
                name: {"path": common.relative(path), "size_bytes": path.stat().st_size, "sha256": common.sha256(path)}
                for name, path in checkpoints.items()
            },
            "best_checkpoint_load": {
                "succeeded": True,
                "total_parameters": best_parameters["total_parameters"],
                "adapter_parameters": best_parameters["adapter_parameters"],
            },
            "adapter": {"path": common.relative(adapter_dir), "size_bytes": adapter_size, **adapter},
            "split": {
                "data": common.relative(data),
                "data_sha256": common.sha256(data),
                "train_list": common.relative(train_list),
                "train_list_sha256": common.sha256(train_list),
                "test_key_present": False,
            },
            "checks": checks,
        }
        common.json_write(log_dir / "metrics.json", metrics)
        manifest_paths = [
            artifact_dir,
            *[
                log_dir / name
                for name in (
                    "command.txt",
                    "resolved_config.yaml",
                    "stdout.log",
                    "stderr.log",
                    "metrics.json",
                    "resource_usage.json",
                    "timing.json",
                    "environment.json",
                    "learning_curve.csv",
                    "vpeft_runtime_metadata.json",
                )
            ],
        ]
        artifacts = common.artifact_rows(manifest_paths)
        common.json_write(
            log_dir / "artifact_manifest.json",
            {"schema_version": 1, "run_id": run_id, "artifacts": artifacts, "artifact_count": len(artifacts)},
        )
        print(json.dumps({"run_id": run_id, "status": status, "phase": args.phase}))
        return 0 if status == "PASS" else 1
    except Exception:  # noqa: BLE001
        with (log_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            stream.write("\n===== AUGMENTATION RUNNER ERROR =====\n")
            stream.write(common.clean_text(traceback.format_exc()))
        if not (log_dir / "metrics.json").exists():
            common.json_write(
                log_dir / "metrics.json",
                {"schema_version": 1, "run_id": run_id, "status": "FAILED", "reason": "runner error"},
            )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
