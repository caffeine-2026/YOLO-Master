#!/usr/bin/env python3
"""Run one immutable C3 P2 multi-seed scaling experiment with full evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from smoke.c3.p1.scripts import run_p1 as common

P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
P2_ROOT = REPO_ROOT / "smoke" / "c3" / "p2"
DATASETS = ("neu", "deeppcb")
METHODS = ("full_sft", "frozen_backbone", "vpeft")
SAMPLE_SIZES = (10, 50, 500)
METHOD_TAGS = {"full_sft": "full", "frozen_backbone": "frozen", "vpeft": "vpeft"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--sample-size", type=int, choices=SAMPLE_SIZES, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=(825, 826), required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = f"{args.dataset}_{args.sample_size}_{METHOD_TAGS[args.method]}_seed{args.seed}_e100"
    log_dir = P2_ROOT / "logs" / run_id
    artifact_dir = P2_ROOT / "artifacts" / run_id
    if log_dir.exists() or artifact_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing P2 run: {run_id}")
    log_dir.mkdir(parents=True)

    config = P2_ROOT / "config" / "runs" / args.dataset / f"{args.method}.yaml"
    data = P2_ROOT / "config" / "splits" / f"{args.dataset}_{args.sample_size}.yaml"
    train_list = P2_ROOT / "config" / "splits" / f"{args.dataset}_{args.sample_size}_seed824.txt"
    for required in (config, data, train_list):
        if not required.is_file():
            raise FileNotFoundError(common.relative(required))
    config_values = yaml.safe_load(config.read_text(encoding="utf-8"))
    if int(config_values.get("epochs", 0)) != 100 or int(config_values.get("seed", 0)) != 824:
        raise ValueError("P2 source config must remain the locked 100-epoch seed824 template")

    yolo = Path(sys.executable).with_name("yolo")
    train_command = [
        str(yolo),
        "train",
        f"cfg={common.relative(config)}",
        f"data={common.relative(data)}",
        f"device={args.device}",
        f"seed={args.seed}",
        f"name={run_id}",
        f"save_dir={common.relative(artifact_dir)}",
        "exist_ok=False",
    ]
    evaluation_output = log_dir / "test_metrics.json"
    evaluation_command = [
        sys.executable,
        common.relative(P1_ROOT / "scripts" / "evaluate_p1.py"),
        "--model",
        common.relative(artifact_dir / "weights" / "best.pt"),
        "--data",
        common.relative(data),
        "--save-dir",
        common.relative(artifact_dir / "test_eval"),
        "--output",
        common.relative(evaluation_output),
        "--device",
        args.device,
        "--batch",
        "8",
        "--imgsz",
        "640",
    ]
    public_train = ["yolo", *train_command[1:]]
    public_eval = ["python", *evaluation_command[1:]]
    (log_dir / "command.txt").write_text(
        "# training\n" + " ".join(public_train) + "\n\n# locked test evaluation\n" + " ".join(public_eval) + "\n",
        encoding="utf-8",
    )
    common.json_write(log_dir / "environment.json", common.environment_evidence(args.device))

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
        resources["measurement_note"] = (
            "Primary peak is trainer-reported GPU_mem; one-second device and process-tree samples are retained."
        )
        common.json_write(log_dir / "resource_usage.json", resources)

        resolved_source = artifact_dir / "args.yaml"
        if resolved_source.is_file():
            (log_dir / "resolved_config.yaml").write_text(
                common.clean_text(resolved_source.read_text(encoding="utf-8")), encoding="utf-8"
            )
        else:
            (log_dir / "resolved_config.yaml").write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
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
        map_values = [float(row["metrics/mAP50-95(B)"]) for row in curve_rows]
        best_index = max(range(len(map_values)), key=map_values.__getitem__)

        checkpoint = artifact_dir / "weights" / "last_healthy.pt"
        best = artifact_dir / "weights" / "best.pt"
        last = artifact_dir / "weights" / "last.pt"
        for required in (checkpoint, best, last):
            if not required.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {common.relative(required)}")
        parameters, model = common.checkpoint_parameters(checkpoint)
        adapter = common.adapter_evidence(model)
        adapter_dir = artifact_dir / "lora_adapter"
        if args.method == "vpeft":
            common.export_adapter(checkpoint, adapter_dir)
            common.json_write(
                log_dir / "vpeft_runtime_metadata.json",
                {
                    "backend": adapter["actual_backend"],
                    "runtime_metadata": getattr(model, "lora_runtime_metadata", {}) or {},
                },
            )
        adapter_size = (
            sum(path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file()) if adapter_dir.exists() else 0
        )

        eval_exit = common.append_evaluation_logs(log_dir / "stdout.log", log_dir / "stderr.log", evaluation_command)
        if eval_exit != 0 or not evaluation_output.is_file():
            raise RuntimeError(f"Locked test evaluation failed with exit code {eval_exit}")
        test_metrics = json.loads(evaluation_output.read_text(encoding="utf-8"))
        finite_metrics = all(
            math.isfinite(float(test_metrics[key])) for key in ("precision", "recall", "map50", "map50_95")
        )

        stdout_text = (log_dir / "stdout.log").read_text(encoding="utf-8")
        stderr_text = (log_dir / "stderr.log").read_text(encoding="utf-8")
        recovery = re.findall(
            r"NaN recovery model|Loss NaN/Inf|Fitness NaN/Inf|Gradient NaN/Inf|EMA NaN/Inf|automatic numerical recovery",
            stdout_text + "\n" + stderr_text,
            flags=re.IGNORECASE,
        )
        resolved = yaml.safe_load((log_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        expected_train = common.relative(train_list)
        data_values = yaml.safe_load(data.read_text(encoding="utf-8"))
        checks = {
            "exit_code_zero": exit_code == 0,
            "requested_cuda_gpu_used": f"CUDA:{args.device} (NVIDIA GeForce RTX 4090" in stdout_text,
            "fixed_protocol": all(
                (
                    resolved.get("epochs") == 100,
                    resolved.get("batch") == 8,
                    resolved.get("imgsz") == 640,
                    resolved.get("workers") == 0,
                    resolved.get("seed") == args.seed,
                    resolved.get("optimizer") == "AdamW",
                    resolved.get("lr0") == 0.001,
                    resolved.get("weight_decay") == 0.0005,
                    resolved.get("cos_lr") is True,
                    resolved.get("amp") is False,
                    data_values.get("train") == expected_train,
                )
            ),
            "complete_finite_learning_curve": len(curve_rows) == 100 and curve_finite,
            "no_numerical_recovery": not recovery,
            "test_metrics_finite": finite_metrics,
            "checkpoints_present": all(path.is_file() for path in (checkpoint, best, last)),
            "resolved_config_present": (log_dir / "resolved_config.yaml").is_file(),
            "resources_present": resources.get("peak_gpu_memory_mib") is not None,
        }
        if args.method == "vpeft":
            checks["strict_vpeft"] = all(
                (
                    resolved.get("lora_vpeft_strict") is True,
                    resolved.get("lora_planner_enabled") is True,
                    resolved.get("lora_planner_backend") == "vpeft",
                    resolved.get("lora_planner_solver") == "ao",
                    adapter["planner_status"] in {"ACCEPT", "ADAPT"},
                    adapter["planner_backend"] == "vpeft",
                    adapter["actual_backend"] == "peft",
                    int(adapter["planned_targets"] or 0) > 0,
                    int(adapter["applied_targets"] or 0) > 0,
                    int(parameters["adapter_parameters"] or 0) > 0,
                    adapter_size > 0,
                )
            )
        else:
            checks["lora_disabled"] = int(resolved.get("lora_r", 0) or 0) == 0 and not adapter["enabled"]
        status = "PASS" if all(checks.values()) else "FAILED"
        metrics = {
            "schema_version": 1,
            "stage": "c3_p2_multiseed_scaling",
            "run_id": run_id,
            "dataset": args.dataset,
            "sample_size": args.sample_size,
            "method": args.method,
            "seed": args.seed,
            "status": status,
            "exit_code": exit_code,
            "test": test_metrics,
            "parameters": parameters,
            "resources": resources,
            "timing": timing,
            "epoch_summary": {
                "best_epoch": best_index + 1,
                "best_map50_95": map_values[best_index],
                "last_epoch_map50_95": map_values[-1],
            },
            "checkpoint": {
                "path": common.relative(best),
                "size_bytes": best.stat().st_size,
                "sha256": common.sha256(best),
            },
            "adapter": {
                "path": common.relative(adapter_dir) if adapter_dir.exists() else None,
                "size_bytes": adapter_size,
                **adapter,
            },
            "split": {"train_list": expected_train, "train_list_sha256": common.sha256(train_list)},
            "numerical_recovery": {"detected": bool(recovery), "markers": recovery},
            "checks": checks,
        }
        common.json_write(log_dir / "metrics.json", metrics)
        manifest_paths = [
            artifact_dir,
            log_dir / "command.txt",
            log_dir / "resolved_config.yaml",
            log_dir / "stdout.log",
            log_dir / "stderr.log",
            log_dir / "metrics.json",
            log_dir / "resource_usage.json",
            log_dir / "timing.json",
            log_dir / "environment.json",
            log_dir / "learning_curve.csv",
            log_dir / "test_metrics.json",
        ]
        artifacts = common.artifact_rows(manifest_paths)
        common.json_write(
            log_dir / "artifact_manifest.json",
            {"schema_version": 1, "run_id": run_id, "artifacts": artifacts, "artifact_count": len(artifacts)},
        )
        print(json.dumps({"run_id": run_id, "status": status, "exit_code": exit_code}))
        return 0 if status == "PASS" else 1
    except Exception:  # noqa: BLE001
        with (log_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            stream.write("\n===== P2 RUNNER ERROR =====\n")
            stream.write(common.clean_text(traceback.format_exc()))
        if not (log_dir / "metrics.json").exists():
            common.json_write(
                log_dir / "metrics.json",
                {"schema_version": 1, "run_id": run_id, "status": "FAILED", "exit_code": 1, "reason": "runner error"},
            )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
