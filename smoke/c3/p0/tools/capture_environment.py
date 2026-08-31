#!/usr/bin/env python3
"""采集 C3 可复现环境与 GPU 入口证据，并移除本机用户路径。"""

from __future__ import annotations

import glob
import hashlib
import json
import platform
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import peft
import torch
import torchvision
import yaml

import ultralytics

REPO_ROOT = Path(__file__).resolve().parents[4]
SMOKE_ROOT = REPO_ROOT / "smoke" / "c3" / "p0"


def clean_text(value: str) -> str:
    for source, target in sorted(
        ((str(REPO_ROOT), "<repo>"), (str(Path.home()), "<user-home>")), key=lambda item: len(item[0]), reverse=True
    ):
        value = value.replace(source, target)
    return value


def command(args: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, check=False, text=True)
    except FileNotFoundError as exc:
        return {
            "command": args,
            "exit_code": 127,
            "stdout": "",
            "stderr": clean_text(str(exc)),
        }
    return {
        "command": args,
        "exit_code": completed.returncode,
        "stdout": clean_text(completed.stdout),
        "stderr": clean_text(completed.stderr),
    }


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    yolo = str(Path(sys.executable).with_name("yolo"))
    nvidia_smi = command(["nvidia-smi"])
    nvidia_inventory = command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    nvcc = command(["nvcc", "--version"])
    anchor = command([sys.executable, "scripts/reproduce_yolo_peft_paper.py", "--check-anchor"])
    yolo_checks = command([yolo, "checks"])
    commit = command(["git", "rev-parse", "HEAD"])
    pci = command(["lspci", "-nn"])
    modules = command(["lsmod"])
    gpu_paths = sorted(glob.glob("/dev/nvidia*"))
    gpu_nodes = [path for path in gpu_paths if stat.S_ISCHR(Path(path).stat().st_mode)]
    pci_nvidia_gpus = [
        line
        for line in str(pci["stdout"]).splitlines()
        if "NVIDIA" in line and ("VGA compatible controller" in line or "3D controller" in line)
    ]
    nvidia_modules = [line for line in str(modules["stdout"]).splitlines() if line.startswith("nvidia")]
    inventory = []
    for line in str(nvidia_inventory["stdout"]).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            inventory.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "driver_version": fields[3],
                }
            )
    cuda_compatibility_match = re.search(r"CUDA Version:\s*([0-9.]+)", str(nvidia_smi["stdout"]))
    required_device_nodes = ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm")
    required_nodes_present = all(path in gpu_nodes for path in required_device_nodes)
    allocation_test = {"attempted": False, "passed": False, "error": None}
    if torch.cuda.is_available():
        allocation_test["attempted"] = True
        try:
            torch.empty(1, device="cuda")
            allocation_test["passed"] = True
        except Exception as exc:  # noqa: BLE001 - 需要把 CUDA 运行时错误写入证据
            allocation_test["error"] = clean_text(f"{type(exc).__name__}: {exc}")
    gpu_ready = all(
        (
            required_nodes_present,
            nvidia_smi["exit_code"] == 0,
            torch.version.cuda is not None,
            torch.cuda.is_available(),
            torch.cuda.device_count() > 0,
            allocation_test["passed"],
        )
    )
    report = {
        "schema_version": 2,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": str(commit["stdout"]).strip(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "peft": peft.__version__,
            "opencv": cv2.__version__,
            "pyyaml": yaml.__version__,
        },
        "cuda": {
            "gpu_execution_ready": gpu_ready,
            "readiness_policy": "必须同时通过字符设备、nvidia-smi、CUDA 版 PyTorch、设备枚举和张量分配检查",
            "pci_nvidia_gpu_count": len(pci_nvidia_gpus),
            "pci_nvidia_gpus": pci_nvidia_gpus,
            "nvidia_smi_inventory": inventory,
            "driver_version": inventory[0]["driver_version"] if inventory else None,
            "nvidia_smi_cuda_compatibility": (cuda_compatibility_match.group(1) if cuda_compatibility_match else None),
            "loaded_nvidia_kernel_modules": nvidia_modules,
            "torch_cuda_build": torch.version.cuda,
            "torch_cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "required_device_nodes": list(required_device_nodes),
            "required_device_nodes_present": required_nodes_present,
            "device_nodes": gpu_nodes,
            "matched_device_paths": gpu_paths,
            "nvidia_smi_exit_code": nvidia_smi["exit_code"],
            "nvcc_exit_code": nvcc["exit_code"],
            "allocation_test": allocation_test,
            "current_classification": "GPU 不可执行；仅允许 CPU 辅助功能核验" if not gpu_ready else "GPU 可执行",
        },
        "assets": {
            "yolo11n_pt_sha256": sha256(REPO_ROOT / "yolo11n.pt"),
            "neu_det_archive_sha256": sha256(REPO_ROOT / "datasets" / "raw" / "NEU-DET.zip"),
            "neu_det_fewshot_manifest_sha256": sha256(
                REPO_ROOT / "datasets" / "neu_det_fewshot_yolo" / "split_manifest.json"
            ),
        },
        "checks": {
            "paper_anchor_exit_code": anchor["exit_code"],
            "yolo_checks_exit_code": yolo_checks["exit_code"],
        },
    }
    evidence = SMOKE_ROOT / "evidence" / "environment.json"
    evidence.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log_dir = SMOKE_ROOT / "logs" / "preflight"
    log_dir.mkdir(parents=True, exist_ok=True)
    sections = [
        ("环境摘要", json.dumps(report, ensure_ascii=False, indent=2)),
        ("nvidia-smi", str(nvidia_smi["stdout"]) + str(nvidia_smi["stderr"])),
        (
            "nvidia-smi inventory",
            str(nvidia_inventory["stdout"]) + str(nvidia_inventory["stderr"]),
        ),
        ("nvcc --version", str(nvcc["stdout"]) + str(nvcc["stderr"])),
        ("lspci -nn", str(pci["stdout"]) + str(pci["stderr"])),
        ("lsmod", str(modules["stdout"]) + str(modules["stderr"])),
        ("YOLO checks", str(yolo_checks["stdout"]) + str(yolo_checks["stderr"])),
        ("论文锚点检查", str(anchor["stdout"]) + str(anchor["stderr"])),
    ]
    text = "\n\n".join(f"===== {title} =====\n{clean_text(body).rstrip()}" for title, body in sections) + "\n"
    (log_dir / "environment.txt").write_text(text, encoding="utf-8")
    print(json.dumps({"status": "captured", "cuda_available": torch.cuda.is_available()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
