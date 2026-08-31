#!/usr/bin/env python3
"""Run one immutable validation-search or final C3 efficient V-PEFT experiment."""

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
ROOT = REPO_ROOT / "smoke" / "c3" / "completion"
DATASETS = ("neu", "deeppcb")
SIZES = (10, 50, 100, 500)
SEEDS = (824, 825, 826)
CANDIDATES = {
    "b100k_predictors": {"adapter_budget": 100_000, "head_train_policy": "predictors"},
    "b150k_predictors": {"adapter_budget": 150_000, "head_train_policy": "predictors"},
    "b250k_predictors": {"adapter_budget": 250_000, "head_train_policy": "predictors"},
    "b150k_frozen": {"adapter_budget": 150_000, "head_train_policy": "frozen"},
}
TRAINABLE_LIMIT = 259_099


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("search", "final"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--sample-size", type=int, choices=SIZES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES))
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def assert_gpu_idle(device: str) -> dict:
    """Refuse to launch when another process already owns the requested GPU."""
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


def selected_candidate() -> str:
    path = ROOT / "results" / "efficiency_selection.json"
    if not path.is_file():
        raise FileNotFoundError("Final runs require the locked validation-only selection JSON")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "SELECTED" or not payload.get("test_access_before_selection") is False:
        raise ValueError("Efficiency selection is not locked or declares test access")
    return str(payload["selected_candidate"])


def paths_and_candidate(args: argparse.Namespace) -> tuple[str, dict, Path, Path]:
    if args.phase == "search":
        if args.sample_size != 100 or args.seed != 824 or not args.candidate:
            raise ValueError("Search is locked to 100-shot seed824 and requires an explicit candidate")
        candidate_id = args.candidate
        run_id = f"search_{args.dataset}_100_{candidate_id}_seed824"
    else:
        candidate_id = selected_candidate()
        if args.candidate and args.candidate != candidate_id:
            raise ValueError("Final candidate differs from the locked validation selection")
        run_id = f"final_{args.dataset}_{args.sample_size}_vpeft_{candidate_id}_seed{args.seed}_e100"
    config = CANDIDATES[candidate_id]
    return (
        run_id,
        config,
        ROOT / "logs" / args.phase / run_id,
        ROOT / "artifacts" / args.phase / run_id,
    )


def main() -> int:
    args = parse_args()
    run_id, candidate, log_dir, artifact_dir = paths_and_candidate(args)
    if log_dir.exists() or artifact_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable run: {run_id}")
    log_dir.mkdir(parents=True)
    gpu_preflight = assert_gpu_idle(args.device)

    source_config = P2_ROOT / "config" / "runs" / args.dataset / "vpeft.yaml"
    data = P2_ROOT / "config" / "splits" / f"{args.dataset}_{args.sample_size}.yaml"
    train_list = P2_ROOT / "config" / "splits" / f"{args.dataset}_{args.sample_size}_seed824.txt"
    for required in (source_config, data, train_list):
        if not required.is_file():
            raise FileNotFoundError(common.relative(required))
    source = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if int(source["epochs"]) != 100 or bool(source["amp"]):
        raise ValueError("Source protocol drifted from the locked 100-epoch amp=false protocol")

    yolo = Path(sys.executable).with_name("yolo")
    train_command = [
        str(yolo),
        "train",
        f"cfg={common.relative(source_config)}",
        f"data={common.relative(data)}",
        f"device={args.device}",
        f"seed={args.seed}",
        f"name={run_id}",
        f"save_dir={common.relative(artifact_dir)}",
        f"lora_adapter_budget={candidate['adapter_budget']}",
        f"lora_head_train_policy={candidate['head_train_policy']}",
        "exist_ok=False",
    ]
    command_lines = ["# training (validation is enabled by the locked config)", " ".join(["yolo", *train_command[1:]])]
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
    if args.phase == "final":
        command_lines.extend(
            [
                "",
                "# locked test evaluation after configuration selection",
                " ".join(["python", *evaluation_command[1:]]),
            ]
        )
    else:
        command_lines.extend(["", "# test evaluation intentionally prohibited during search"])
    (log_dir / "command.txt").write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    environment = common.environment_evidence(args.device)
    environment["gpu_preflight"] = gpu_preflight
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
        else:
            (log_dir / "resolved_config.yaml").write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
        if exit_code != 0:
            common.json_write(
                log_dir / "metrics.json",
                {"schema_version": 1, "run_id": run_id, "status": "FAILED", "exit_code": exit_code},
            )
            return exit_code

        curve_source = artifact_dir / "results.csv"
        shutil.copy2(curve_source, log_dir / "learning_curve.csv")
        curve_rows, curve_finite = common.read_learning_curve(log_dir / "learning_curve.csv")
        map50 = [float(row["metrics/mAP50(B)"]) for row in curve_rows]
        map5095 = [float(row["metrics/mAP50-95(B)"]) for row in curve_rows]
        best_index = max(range(len(map5095)), key=map5095.__getitem__)
        checkpoints = {name: artifact_dir / "weights" / f"{name}.pt" for name in ("best", "last", "last_healthy")}
        for checkpoint in checkpoints.values():
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {common.relative(checkpoint)}")
        # ``best.pt`` stores the EMA copy with all ``requires_grad`` flags cleared;
        # use the saved online model for the actual trainable-parameter audit and
        # independently load best.pt to prove the evaluation checkpoint is valid.
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
        test_metrics = None
        if args.phase == "final":
            eval_exit = common.append_evaluation_logs(
                log_dir / "stdout.log", log_dir / "stderr.log", evaluation_command
            )
            if eval_exit != 0 or not evaluation_output.is_file():
                raise RuntimeError(f"Locked test evaluation failed with exit code {eval_exit}")
            test_metrics = json.loads(evaluation_output.read_text(encoding="utf-8"))
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
                    resolved.get("lora_adapter_budget") == candidate["adapter_budget"],
                    resolved.get("lora_head_train_policy") == candidate["head_train_policy"],
                )
            ),
            "strict_vpeft": all(
                (
                    resolved.get("lora_planner_backend") == "vpeft",
                    resolved.get("lora_planner_solver") == "ao",
                    resolved.get("lora_vpeft_strict") is True,
                    adapter["planner_status"] in {"ACCEPT", "ADAPT"},
                    adapter["planner_backend"] == "vpeft",
                    adapter["actual_backend"] == "peft",
                    int(adapter["planned_targets"] or 0) > 0,
                    int(adapter["applied_targets"] or 0) > 0,
                )
            ),
            "trainable_at_most_ten_percent_full": parameters["trainable_parameters"] <= TRAINABLE_LIMIT,
            "test_not_accessed_during_search": args.phase == "final" or not evaluation_output.exists(),
            "test_metrics_present_for_final": args.phase == "search" or test_metrics is not None,
        }
        status = "PASS" if all(checks.values()) else "FAILED"
        metrics = {
            "schema_version": 1,
            "stage": f"c3_efficiency_{args.phase}",
            "run_id": run_id,
            "dataset": args.dataset,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "candidate": {"id": args.candidate if args.phase == "search" else selected_candidate(), **candidate},
            "status": status,
            "exit_code": exit_code,
            "validation": {
                "best_epoch": best_index + 1,
                "best_map50": map50[best_index],
                "best_map50_95": map5095[best_index],
                "last_map50": map50[-1],
                "last_map50_95": map5095[-1],
            },
            "test": test_metrics,
            "parameters": parameters,
            "resources": resources,
            "timing": timing,
            "checkpoint": {
                name: {"path": common.relative(path), "size_bytes": path.stat().st_size, "sha256": common.sha256(path)}
                for name, path in checkpoints.items()
            },
            "best_checkpoint_load": {
                "succeeded": True,
                "total_parameters": best_parameters["total_parameters"],
                "adapter_parameters": best_parameters["adapter_parameters"],
                "requires_grad_note": "EMA best checkpoint intentionally stores requires_grad=false",
            },
            "adapter": {"path": common.relative(adapter_dir), "size_bytes": adapter_size, **adapter},
            "split": {"train_list": common.relative(train_list), "train_list_sha256": common.sha256(train_list)},
            "numerical_recovery": {"detected": bool(recovery), "markers": recovery},
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
        if args.phase == "final":
            manifest_paths.append(evaluation_output)
        artifacts = common.artifact_rows(manifest_paths)
        common.json_write(
            log_dir / "artifact_manifest.json",
            {"schema_version": 1, "run_id": run_id, "artifacts": artifacts, "artifact_count": len(artifacts)},
        )
        print(json.dumps({"run_id": run_id, "status": status, "phase": args.phase}))
        return 0 if status == "PASS" else 1
    except Exception:  # noqa: BLE001
        with (log_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            stream.write("\n===== EFFICIENCY RUNNER ERROR =====\n")
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
