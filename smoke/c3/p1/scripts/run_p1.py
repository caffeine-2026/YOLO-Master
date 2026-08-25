#!/usr/bin/env python3
"""Run one immutable C3 P1 pilot and emit complete reproducibility evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
DATASETS = ("neu_det", "deeppcb")
METHODS = ("full_sft", "frozen_backbone", "vpeft")
ADAPTER_MARKERS = ("lora_", "hada_", "lokr_", "oft_", "boft_", "ia3_", "hra_")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=824)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def clean_text(value: str) -> str:
    clean = ANSI_RE.sub("", value)
    replacements = sorted(
        ((str(REPO_ROOT), "<repo>"), (str(Path.home()), "<user-home>")),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for source, target in replacements:
        clean = clean.replace(source, target)
    return clean


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def environment_evidence(device: str) -> dict[str, object]:
    import torch

    gpu_name = None
    if torch.cuda.is_available() and device.lower() != "cpu":
        gpu_name = torch.cuda.get_device_name(int(device.split(",")[0]))
    return {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device_argument": device,
        "gpu_name": gpu_name,
        "git_ref": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    }


def process_tree_pids(process: subprocess.Popen[str]) -> set[int]:
    pids = {process.pid}
    try:
        import psutil
    except ModuleNotFoundError:
        return pids
    try:
        pids.update(child.pid for child in psutil.Process(process.pid).children(recursive=True))
    except psutil.Error:
        return pids
    return pids


def sample_resources(process: subprocess.Popen[str], path: Path, stop: threading.Event, device: str) -> None:
    try:
        import psutil
    except ModuleNotFoundError:
        psutil = None
    tracked = psutil.Process(process.pid) if psutil else None
    if tracked:
        tracked.cpu_percent(interval=None)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "timestamp_utc",
                "cpu_percent",
                "rss_mib",
                "gpu_index",
                "device_memory_used_mib",
                "process_tree_gpu_memory_mib",
            )
        )
        while not stop.wait(1.0):
            cpu = rss = ""
            if tracked:
                try:
                    children = tracked.children(recursive=True)
                    cpu = tracked.cpu_percent(interval=None) + sum(child.cpu_percent(interval=None) for child in children)
                    rss = (tracked.memory_info().rss + sum(child.memory_info().rss for child in children)) / 1024**2
                except psutil.Error:
                    pass
            gpu_index = device_memory = process_memory = ""
            if device.lower() != "cpu" and shutil.which("nvidia-smi"):
                gpu_query = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={device.split(',')[0]}",
                        "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if gpu_query.returncode == 0 and gpu_query.stdout.strip():
                    gpu_index, device_memory = [value.strip() for value in gpu_query.stdout.splitlines()[0].split(",", 1)]
                app_query = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if app_query.returncode == 0:
                    wanted = process_tree_pids(process)
                    used = 0.0
                    found = False
                    for line in app_query.stdout.splitlines():
                        fields = [value.strip() for value in line.split(",", 1)]
                        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) in wanted:
                            used += float(fields[1])
                            found = True
                    process_memory = f"{used:.2f}" if found else ""
            writer.writerow(
                (
                    datetime.now(timezone.utc).isoformat(),
                    f"{cpu:.2f}" if isinstance(cpu, float) else cpu,
                    f"{rss:.2f}" if isinstance(rss, float) else rss,
                    gpu_index,
                    device_memory,
                    process_memory,
                )
            )
            stream.flush()


def summarize_samples(path: Path) -> dict[str, float | None]:
    columns = {"cpu_percent": [], "rss_mib": [], "device_memory_used_mib": [], "process_tree_gpu_memory_mib": []}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            for key, values in columns.items():
                if row.get(key):
                    values.append(float(row[key]))
    return {
        "peak_cpu_percent": max(columns["cpu_percent"], default=None),
        "peak_rss_mib": max(columns["rss_mib"], default=None),
        "peak_device_memory_mib": max(columns["device_memory_used_mib"], default=None),
        "peak_process_tree_gpu_memory_mib": max(columns["process_tree_gpu_memory_mib"], default=None),
    }


def tee_stream(source: IO[str], destination: Path, terminal: IO[str]) -> None:
    with destination.open("w", encoding="utf-8") as stream:
        for line in iter(source.readline, ""):
            line = clean_text(line)
            stream.write(line)
            stream.flush()
            terminal.write(line)
            terminal.flush()
    source.close()


def run_captured(command: list[str], stdout_path: Path, stderr_path: Path, device: str, resource_path: Path) -> tuple[int, float]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None and process.stderr is not None
    stop = threading.Event()
    sampler = threading.Thread(target=sample_resources, args=(process, resource_path, stop, device), daemon=True)
    out_thread = threading.Thread(target=tee_stream, args=(process.stdout, stdout_path, sys.stdout), daemon=True)
    err_thread = threading.Thread(target=tee_stream, args=(process.stderr, stderr_path, sys.stderr), daemon=True)
    sampler.start()
    out_thread.start()
    err_thread.start()
    exit_code = process.wait()
    out_thread.join()
    err_thread.join()
    stop.set()
    sampler.join(timeout=5)
    return exit_code, time.monotonic() - started


def append_evaluation_logs(stdout_path: Path, stderr_path: Path, command: list[str]) -> int:
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    with stdout_path.open("a", encoding="utf-8") as stream:
        stream.write("\n===== LOCKED TEST EVALUATION =====\n")
        stream.write(clean_text(completed.stdout))
    with stderr_path.open("a", encoding="utf-8") as stream:
        stream.write("\n===== LOCKED TEST EVALUATION =====\n")
        stream.write(clean_text(completed.stderr))
    print(clean_text(completed.stdout), end="")
    if completed.stderr:
        print(clean_text(completed.stderr), end="", file=sys.stderr)
    return completed.returncode


def epoch_gpu_peak_mib(text: str) -> float | None:
    values = []
    for amount, unit in re.findall(r"^\s*\d+/\d+\s+([0-9.]+)([GM])\s+", text, flags=re.MULTILINE):
        value = float(amount) * (1024 if unit == "G" else 1)
        values.append(value)
    return max(values, default=None)


def read_learning_curve(path: Path) -> tuple[list[dict[str, str]], bool]:
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
    return rows, finite


def checkpoint_parameters(checkpoint: Path) -> tuple[dict[str, object], object]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = payload.get("model")
    if model is None:
        raise ValueError(f"Checkpoint has no online model: {relative(checkpoint)}")
    named = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in named)
    trainable = sum(parameter.numel() for _, parameter in named if parameter.requires_grad)
    adapter = sum(
        parameter.numel() for name, parameter in named if any(marker in name.lower() for marker in ADAPTER_MARKERS)
    )
    top_level: dict[int, dict[str, int]] = {}
    for name, parameter in named:
        match = re.match(r"^model\.(\d+)\.", name)
        if not match:
            continue
        index = int(match.group(1))
        entry = top_level.setdefault(index, {"total": 0, "trainable": 0})
        entry["total"] += parameter.numel()
        if parameter.requires_grad:
            entry["trainable"] += parameter.numel()
    module_rows = [
        {
            "module": f"model.{index}",
            "total_parameters": values["total"],
            "trainable_parameters": values["trainable"],
            "frozen": values["trainable"] == 0,
        }
        for index, values in sorted(top_level.items())
    ]
    return (
        {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_parameter_ratio": trainable / total,
            "adapter_parameters": adapter,
            "frozen_modules": [row["module"] for row in module_rows if row["frozen"]],
            "trainable_modules": [row["module"] for row in module_rows if not row["frozen"]],
            "top_level_modules": module_rows,
        },
        model,
    )


def adapter_evidence(model: object) -> dict[str, object]:
    metadata = dict(getattr(model, "lora_runtime_metadata", {}) or {})
    plan = dict(metadata.get("placement_plan") or {})
    targets = list(metadata.get("target_modules") or [])
    return {
        "enabled": bool(getattr(model, "lora_enabled", False)),
        "planner_status": plan.get("status"),
        "planner_backend": plan.get("planner_backend"),
        "actual_backend": metadata.get("effective_backend"),
        "planned_targets": len(plan.get("targets") or []),
        "applied_targets": len(targets),
    }


def export_adapter(checkpoint: Path, adapter_dir: Path) -> None:
    from ultralytics import YOLO

    if adapter_dir.exists():
        raise FileExistsError(f"Refusing to overwrite adapter directory: {relative(adapter_dir)}")
    model = YOLO(checkpoint)
    if not model.save_adapters(adapter_dir):
        raise RuntimeError("Independent V-PEFT adapter export failed")


def artifact_rows(paths: list[Path]) -> list[dict[str, object]]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(item for item in path.rglob("*") if item.is_file())
    return [
        {"path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(files)
    ]


def main() -> int:
    args = parse_args()
    if args.seed != 824:
        raise ValueError("This pilot stage is locked to seed=824; seeds 825/826 are outside the current scope")
    run_id = f"{args.dataset}_{args.method}_seed{args.seed}"
    log_dir = P1_ROOT / "logs" / run_id
    artifact_dir = P1_ROOT / "artifacts" / run_id
    if log_dir.exists() or artifact_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing P1 run: {run_id}")
    log_dir.mkdir(parents=True)

    config = P1_ROOT / "config" / args.dataset / f"{args.method}.yaml"
    data = P1_ROOT / "config" / args.dataset / "dataset.yaml"
    train_list = P1_ROOT / "config" / args.dataset / f"train_seed{args.seed}.txt"
    for required in (config, data, train_list):
        if not required.is_file():
            raise FileNotFoundError(relative(required))
    config_values = yaml.safe_load(config.read_text(encoding="utf-8"))
    if int(config_values.get("epochs", 0)) != 30 or bool(config_values.get("amp", True)):
        raise ValueError("Pilot config must use 30 epochs and amp=false")

    yolo = Path(sys.executable).with_name("yolo")
    if not yolo.is_file():
        raise FileNotFoundError("The active virtual environment has no yolo CLI")
    train_command = [
        str(yolo),
        "train",
        f"cfg={relative(config)}",
        f"data={relative(data)}",
        f"device={args.device}",
        f"seed={args.seed}",
        f"name={run_id}",
        f"save_dir={relative(artifact_dir)}",
        "exist_ok=False",
    ]
    public_train_command = ["yolo", *train_command[1:]]
    evaluation_output = log_dir / "test_metrics.json"
    evaluation_command = [
        sys.executable,
        relative(P1_ROOT / "scripts" / "evaluate_p1.py"),
        "--model",
        relative(artifact_dir / "weights" / "best.pt"),
        "--data",
        relative(data),
        "--save-dir",
        relative(artifact_dir / "test_eval"),
        "--output",
        relative(evaluation_output),
        "--device",
        args.device,
        "--batch",
        str(config_values["batch"]),
        "--imgsz",
        str(config_values["imgsz"]),
    ]
    public_eval_command = ["python", *evaluation_command[1:]]
    (log_dir / "command.txt").write_text(
        "# training\n" + " ".join(public_train_command) + "\n\n# locked test evaluation\n" + " ".join(public_eval_command) + "\n",
        encoding="utf-8",
    )
    json_write(log_dir / "environment.json", environment_evidence(args.device))

    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        exit_code, training_seconds = run_captured(
            train_command,
            log_dir / "stdout.log",
            log_dir / "stderr.log",
            args.device,
            log_dir / "resource_samples.csv",
        )
        finished_utc = datetime.now(timezone.utc).isoformat()
        timing = {
            "schema_version": 1,
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "training_seconds": round(training_seconds, 3),
            "gpu_hours": round(training_seconds / 3600, 8),
            "training_exit_code": exit_code,
        }
        json_write(log_dir / "timing.json", timing)
        samples = summarize_samples(log_dir / "resource_samples.csv")
        stdout_text = (log_dir / "stdout.log").read_text(encoding="utf-8")
        samples["epoch_reported_peak_gpu_memory_mib"] = epoch_gpu_peak_mib(stdout_text)
        samples["peak_gpu_memory_mib"] = samples["epoch_reported_peak_gpu_memory_mib"] or samples[
            "peak_process_tree_gpu_memory_mib"
        ]
        samples["measurement_note"] = (
            "Primary peak is the maximum GPU_mem reported by the trainer; device and process-tree nvidia-smi samples "
            "are retained separately at one-second resolution."
        )
        json_write(log_dir / "resource_usage.json", samples)

        resolved_source = artifact_dir / "args.yaml"
        if resolved_source.is_file():
            (log_dir / "resolved_config.yaml").write_text(
                clean_text(resolved_source.read_text(encoding="utf-8")), encoding="utf-8"
            )
        else:
            (log_dir / "resolved_config.yaml").write_text(config.read_text(encoding="utf-8"), encoding="utf-8")

        if exit_code != 0:
            failure = {"schema_version": 1, "run_id": run_id, "status": "FAIL", "exit_code": exit_code}
            json_write(log_dir / "metrics.json", failure)
            return exit_code

        curve_source = artifact_dir / "results.csv"
        if not curve_source.is_file():
            raise FileNotFoundError("Training completed without results.csv")
        shutil.copy2(curve_source, log_dir / "learning_curve.csv")
        curve_rows, curve_finite = read_learning_curve(log_dir / "learning_curve.csv")

        checkpoint = artifact_dir / "weights" / "last_healthy.pt"
        best = artifact_dir / "weights" / "best.pt"
        last = artifact_dir / "weights" / "last.pt"
        for required in (checkpoint, best, last):
            if not required.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {relative(required)}")
        parameters, model = checkpoint_parameters(checkpoint)
        vpeft = adapter_evidence(model)
        adapter_dir = artifact_dir / "lora_adapter"
        if args.method == "vpeft":
            export_adapter(checkpoint, adapter_dir)
            runtime_payload = {
                "backend": vpeft["actual_backend"],
                "runtime_metadata": getattr(model, "lora_runtime_metadata", {}) or {},
            }
            json_write(log_dir / "vpeft_runtime_metadata.json", runtime_payload)

        eval_exit = append_evaluation_logs(log_dir / "stdout.log", log_dir / "stderr.log", evaluation_command)
        if eval_exit != 0 or not evaluation_output.is_file():
            raise RuntimeError(f"Locked test evaluation failed with exit code {eval_exit}")
        test_metrics = json.loads(evaluation_output.read_text(encoding="utf-8"))
        finite_metrics = all(
            math.isfinite(float(test_metrics[key])) for key in ("precision", "recall", "map50", "map50_95")
        )

        stdout_text = (log_dir / "stdout.log").read_text(encoding="utf-8")
        stderr_text = (log_dir / "stderr.log").read_text(encoding="utf-8")
        recovery_markers = re.findall(
            r"NaN recovery model|Loss NaN/Inf|Fitness NaN/Inf|Gradient NaN/Inf|EMA NaN/Inf|automatic numerical recovery",
            stdout_text + "\n" + stderr_text,
            flags=re.IGNORECASE,
        )
        adapter_size = sum(path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file()) if adapter_dir.exists() else 0
        resolved = yaml.safe_load((log_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        checks = {
            "exit_code_zero": exit_code == 0,
            "cuda_gpu0_used": args.device == "0" and "CUDA:0 (NVIDIA GeForce RTX 4090" in stdout_text,
            "fixed_protocol": all(
                (
                    resolved.get("epochs") == 30,
                    resolved.get("batch") == 8,
                    resolved.get("imgsz") == 640,
                    resolved.get("workers") == 0,
                    resolved.get("seed") == 824,
                    resolved.get("amp") is False,
                )
            ),
            "complete_finite_learning_curve": len(curve_rows) == 30 and curve_finite,
            "no_numerical_recovery": not recovery_markers,
            "test_metrics_finite": finite_metrics,
            "checkpoints_present": all(path.is_file() for path in (checkpoint, best, last)),
            "resolved_config_present": (log_dir / "resolved_config.yaml").is_file(),
            "resources_present": samples.get("peak_gpu_memory_mib") is not None,
        }
        if args.method == "vpeft":
            checks["strict_vpeft"] = all(
                (
                    resolved.get("lora_planner_enabled") is True,
                    resolved.get("lora_planner_backend") == "vpeft",
                    resolved.get("lora_planner_solver") == "ao",
                    resolved.get("lora_vpeft_strict") is True,
                    vpeft["planner_status"] in {"ACCEPT", "ADAPT"},
                    vpeft["planner_backend"] == "vpeft",
                    vpeft["actual_backend"] == "peft",
                    int(vpeft["planned_targets"] or 0) > 0,
                    int(vpeft["applied_targets"] or 0) > 0,
                    adapter_dir.is_dir(),
                    adapter_size > 0,
                )
            )
        else:
            checks["lora_disabled"] = int(resolved.get("lora_r", 0) or 0) == 0 and not vpeft["enabled"]
        status = "PASS" if all(checks.values()) else "FAIL"
        metrics = {
            "schema_version": 1,
            "run_id": run_id,
            "dataset": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "status": status,
            "exit_code": exit_code,
            "protocol": {
                "model": resolved.get("model"),
                "epochs": resolved.get("epochs"),
                "batch": resolved.get("batch"),
                "imgsz": resolved.get("imgsz"),
                "optimizer": resolved.get("optimizer"),
                "lr0": resolved.get("lr0"),
                "weight_decay": resolved.get("weight_decay"),
                "amp": resolved.get("amp"),
                "device": resolved.get("device"),
            },
            "test": test_metrics,
            "parameters": parameters,
            "resources": samples,
            "timing": timing,
            "checkpoint": {
                "path": relative(best),
                "size_bytes": best.stat().st_size,
                "sha256": sha256(best),
            },
            "adapter": {
                "path": relative(adapter_dir) if adapter_dir.exists() else None,
                "size_bytes": adapter_size,
                **vpeft,
            },
            "numerical_recovery": {"detected": bool(recovery_markers), "markers": recovery_markers},
            "checks": checks,
        }
        json_write(log_dir / "metrics.json", metrics)
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
        artifacts = artifact_rows(manifest_paths)
        json_write(
            log_dir / "artifact_manifest.json",
            {"schema_version": 1, "run_id": run_id, "artifacts": artifacts, "artifact_count": len(artifacts)},
        )
        print(json.dumps({"run_id": run_id, "status": status, "exit_code": exit_code}, ensure_ascii=False))
        return 0 if status == "PASS" else 1
    except Exception:  # noqa: BLE001 - the evidence boundary must preserve every post-training failure
        with (log_dir / "stderr.log").open("a", encoding="utf-8") as stream:
            stream.write("\n===== P1 RUNNER ERROR =====\n")
            stream.write(clean_text(traceback.format_exc()))
        failure_path = log_dir / "metrics.json"
        if not failure_path.exists():
            json_write(
                failure_path,
                {"schema_version": 1, "run_id": run_id, "status": "FAIL", "exit_code": 1, "reason": "runner error; see stderr.log"},
            )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
