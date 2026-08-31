#!/usr/bin/env python3
"""运行一个 C3 训练任务，并生成可提交的路径脱敏日志与证据索引。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
SMOKE_ROOT = REPO_ROOT / "smoke" / "c3" / "p0"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="smoke/c3/p0/config/vpeft_smoke.yaml")
    parser.add_argument("--data", default="smoke/c3/p0/config/datasets/neu_det_fewshot.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", required=True, help="同时作为训练目录名和证据运行 ID。")
    parser.add_argument("--amp", choices=("true", "false"), default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--solver", choices=("ao", "dco", "mip"), default=None)
    parser.add_argument("--refresh-existing", action="store_true", help="重新索引已完成的同名运行，不再次训练。")
    return parser.parse_args()


def relative_path(value: str, *, must_exist: bool = True) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"仅接受仓库相对路径：{value}")
    resolved = (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"路径超出仓库范围：{value}") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(value)
    return resolved


def clean_text(value: str) -> str:
    """移除终端控制符并替换本机路径前缀，不删除任何日志行。"""
    clean = ANSI_RE.sub("", value)
    replacements = sorted(
        ((str(REPO_ROOT), "<repo>"), (str(Path.home()), "<user-home>")), key=lambda item: len(item[0]), reverse=True
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


def sample_resources(process: subprocess.Popen[str], output: Path, stop: threading.Event, device: str) -> None:
    try:
        import psutil
    except ModuleNotFoundError:
        psutil = None

    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("timestamp_utc", "cpu_percent", "rss_mib", "gpu_index", "gpu_memory_used_mib"))
        tracked = psutil.Process(process.pid) if psutil else None
        if tracked:
            tracked.cpu_percent(interval=None)
        while not stop.wait(1.0):
            cpu_percent = ""
            rss_mib = ""
            if tracked:
                try:
                    children = tracked.children(recursive=True)
                    cpu_percent = tracked.cpu_percent(interval=None) + sum(
                        child.cpu_percent(interval=None) for child in children
                    )
                    rss_mib = (tracked.memory_info().rss + sum(child.memory_info().rss for child in children)) / (
                        1024**2
                    )
                except psutil.Error:
                    pass

            gpu_index = ""
            gpu_memory = ""
            if device.lower() != "cpu" and shutil.which("nvidia-smi"):
                query = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={device.split(',')[0]}",
                        "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    check=False,
                    text=True,
                )
                if query.returncode == 0 and query.stdout.strip():
                    gpu_index, gpu_memory = (item.strip() for item in query.stdout.splitlines()[0].split(",", 1))
            writer.writerow(
                (
                    datetime.now(timezone.utc).isoformat(),
                    f"{cpu_percent:.2f}" if isinstance(cpu_percent, float) else cpu_percent,
                    f"{rss_mib:.2f}" if isinstance(rss_mib, float) else rss_mib,
                    gpu_index,
                    gpu_memory,
                )
            )
            stream.flush()


def summarize_resources(path: Path) -> dict[str, float | None]:
    cpu_values: list[float] = []
    rss_values: list[float] = []
    gpu_values: list[float] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["cpu_percent"]:
                cpu_values.append(float(row["cpu_percent"]))
            if row["rss_mib"]:
                rss_values.append(float(row["rss_mib"]))
            if row["gpu_memory_used_mib"]:
                gpu_values.append(float(row["gpu_memory_used_mib"]))
    return {
        "peak_cpu_percent": max(cpu_values, default=None),
        "peak_rss_mib": max(rss_values, default=None),
        "peak_gpu_memory_mib": max(gpu_values, default=None),
    }


def analyze_numerical_recovery(log_dir: Path) -> dict[str, object]:
    """从完整日志识别训练器是否因非有限值重试了 epoch。"""
    train_log = (log_dir / "train.log").read_text(encoding="utf-8")
    resolved_config = log_dir / "resolved_config.yaml"
    settings = (yaml.safe_load(resolved_config.read_text(encoding="utf-8")) or {}) if resolved_config.is_file() else {}
    requested_epochs = int(settings.get("epochs", 0) or 0)
    epoch_attempts = len(re.findall(r"^\s*Epoch\s+GPU_mem\b", train_log, flags=re.MULTILINE))
    recovery_attempts = max(0, epoch_attempts - requested_epochs)
    amp_requested = bool(settings.get("amp", False))
    return {
        "detected": recovery_attempts > 0,
        "requested_epochs": requested_epochs,
        "observed_epoch_attempts": epoch_attempts,
        "recovery_attempts": recovery_attempts,
        "amp_requested": amp_requested,
        "amp_disabled_by_recovery": amp_requested and recovery_attempts > 0,
        "explicit_lora_restore_messages": train_log.count("NaN recovery model"),
        "classification": (
            "AMP 非有限值恢复后完成，属于降级运行"
            if amp_requested and recovery_attempts > 0
            else "未检测到自动数值恢复"
        ),
    }


def artifact_rows(run_dir: Path) -> list[dict[str, str | int]]:
    rows = []
    if not run_dir.is_dir():
        return rows
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def export_adapters_if_requested(config: Path, log_dir: Path, run_dir: Path) -> None:
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not settings.get("lora_save_adapters") or int(settings.get("lora_r", 0) or 0) <= 0:
        return
    adapter_name = str(settings.get("lora_adapter_dir") or "lora_adapter")
    adapter_dir = run_dir / adapter_name
    if adapter_dir.is_dir():
        return
    checkpoint = run_dir / "weights" / "last_healthy.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("LoRA 运行缺少可用于导出适配器的检查点")
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    if not model.save_adapters(adapter_dir):
        raise RuntimeError("独立 LoRA 适配器导出失败")
    for path in adapter_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".txt", ".yaml", ".yml"}:
            path.write_text(clean_text(path.read_text(encoding="utf-8")), encoding="utf-8")
    (log_dir / "postprocess.txt").write_text(
        f"adapter_export=completed\nadapter_dir={adapter_dir.relative_to(REPO_ROOT).as_posix()}\n",
        encoding="utf-8",
    )


def write_artifact_evidence(
    config: Path,
    log_dir: Path,
    run_dir: Path,
    *,
    export_adapters: bool = True,
) -> list[dict[str, str | int]]:
    if export_adapters:
        export_adapters_if_requested(config, log_dir, run_dir)
    adapter_dir = run_dir / "lora_adapter"
    for source_name, target_name in (
        ("adapter_config.json", "adapter_config.json"),
        ("runtime_metadata.json", "vpeft_runtime_metadata.json"),
    ):
        source = adapter_dir / source_name
        if source.is_file():
            (log_dir / target_name).write_text(clean_text(source.read_text(encoding="utf-8")), encoding="utf-8")
    if (run_dir / "args.yaml").is_file():
        resolved = (run_dir / "args.yaml").read_text(encoding="utf-8")
        (log_dir / "resolved_config.yaml").write_text(clean_text(resolved), encoding="utf-8")
    if (run_dir / "results.csv").is_file():
        shutil.copy2(run_dir / "results.csv", log_dir / "metrics.csv")
    artifacts = artifact_rows(run_dir)
    with (log_dir / "artifact_index.txt").open("w", encoding="utf-8") as stream:
        stream.write("sha256  size_bytes  path\n")
        for item in artifacts:
            stream.write(f"{item['sha256']}  {item['size_bytes']}  {item['path']}\n")
    return artifacts


def main() -> int:
    args = parse_args()
    config = relative_path(args.config)
    data = relative_path(args.data)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        raise ValueError("--name 仅允许字母、数字、点、下划线和连字符")

    log_dir = SMOKE_ROOT / "logs" / args.name
    # Ultralytics 将相对 project 放在 runs/<task>/ 下；当前命令的 project 值为 runs/vpeft_smoke。
    run_dir = REPO_ROOT / "runs" / "detect" / "runs" / "vpeft_smoke" / args.name
    if args.refresh_existing:
        if not log_dir.is_dir() or not run_dir.is_dir():
            raise FileNotFoundError("刷新要求同名日志目录和训练目录均已存在")
        result_path = log_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        artifacts = write_artifact_evidence(
            config,
            log_dir,
            run_dir,
            export_adapters=result.get("status") == "completed",
        )
        result.update(
            {
                "schema_version": 2,
                "run_dir": run_dir.relative_to(REPO_ROOT).as_posix(),
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
                "numerical_recovery": analyze_numerical_recovery(log_dir),
            }
        )
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "refreshed", "artifact_count": len(artifacts)}, ensure_ascii=False))
        return 0
    if log_dir.exists():
        raise FileExistsError(f"证据目录已存在：smoke/c3/p0/logs/{args.name}")
    log_dir.mkdir(parents=True)

    if run_dir.exists():
        raise FileExistsError(f"训练目录已存在：{run_dir.relative_to(REPO_ROOT)}")

    yolo = Path(sys.executable).with_name("yolo")
    if not yolo.is_file():
        raise FileNotFoundError("当前 Python 环境中没有 yolo CLI")
    command = [
        str(yolo),
        "train",
        f"cfg={config.relative_to(REPO_ROOT).as_posix()}",
        f"data={data.relative_to(REPO_ROOT).as_posix()}",
        f"device={args.device}",
        f"name={args.name}",
        "project=runs/vpeft_smoke",
        "exist_ok=False",
    ]
    if args.amp is not None:
        command.append(f"amp={args.amp}")
    if args.imgsz is not None:
        command.append(f"imgsz={args.imgsz}")
    if args.epochs is not None:
        command.append(f"epochs={args.epochs}")
    if args.solver is not None:
        command.append(f"lora_planner_solver={args.solver}")
    publishable_command = ["yolo", *command[1:]]
    (log_dir / "command.txt").write_text(" ".join(publishable_command) + "\n", encoding="utf-8")

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stop = threading.Event()
    sampler = threading.Thread(
        target=sample_resources,
        args=(process, log_dir / "resources.csv", stop, args.device),
        daemon=True,
    )
    sampler.start()
    with (log_dir / "train.log").open("w", encoding="utf-8") as log:
        assert process.stdout is not None
        for line in process.stdout:
            normalized = clean_text(line)
            line_ending = "\n" if normalized.endswith(("\n", "\r")) else ""
            normalized = normalized.rstrip(" \t\r\n") + line_ending
            log.write(normalized)
            log.flush()
            print(normalized, end="", flush=True)
    exit_code = process.wait()
    stop.set()
    sampler.join(timeout=5)
    wall_seconds = time.monotonic() - started
    finished_utc = datetime.now(timezone.utc).isoformat()

    artifacts = write_artifact_evidence(config, log_dir, run_dir, export_adapters=exit_code == 0)

    resource_summary = summarize_resources(log_dir / "resources.csv")
    result = {
        "schema_version": 2,
        "status": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "wall_seconds": round(wall_seconds, 3),
        "device": args.device,
        "command": publishable_command,
        "config": config.relative_to(REPO_ROOT).as_posix(),
        "data": data.relative_to(REPO_ROOT).as_posix(),
        "run_dir": run_dir.relative_to(REPO_ROOT).as_posix(),
        "resource_summary": resource_summary,
        "numerical_recovery": analyze_numerical_recovery(log_dir),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "log_path_policy": "完整保留日志行，仅将仓库根目录和用户主目录前缀替换为占位符。",
    }
    (log_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (log_dir / "time.txt").write_text(
        f"started_utc={started_utc}\nfinished_utc={finished_utc}\nwall_seconds={wall_seconds:.3f}\nexit_code={exit_code}\n",
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
