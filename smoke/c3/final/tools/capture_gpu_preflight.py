#!/usr/bin/env python3
"""Capture a privacy-safe C3 GPU preflight snapshot from the real host."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "smoke" / "c3" / "final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="smoke/c3/final/evidence/gpu_preflight.json")
    return parser.parse_args()


def nvidia_smi(query: str) -> list[list[str]]:
    """Run a CSV-formatted nvidia-smi query and return stripped rows."""
    command = ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return [[value.strip() for value in row] for row in csv.reader(StringIO(completed.stdout)) if row]


def main() -> int:
    """Write the host snapshot without process names, usernames, or command lines."""
    args = parse_args()
    output = (REPO_ROOT / args.output).resolve()
    output.relative_to(FINAL_ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)

    gpu_rows = nvidia_smi("gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pstate")
    process_rows = nvidia_smi("compute-apps=gpu_uuid,pid,used_memory")
    process_counts: dict[str, int] = {}
    process_memory: dict[str, int] = {}
    for uuid, _pid, used_memory in process_rows:
        process_counts[uuid] = process_counts.get(uuid, 0) + 1
        process_memory[uuid] = process_memory.get(uuid, 0) + int(used_memory)

    gpus = []
    for index, uuid, name, total, used, utilization, pstate in gpu_rows:
        count = process_counts.get(uuid, 0)
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "utilization_percent": int(utilization),
                "pstate": pstate,
                "compute_process_count": count,
                "compute_process_memory_mib": process_memory.get(uuid, 0),
                "available_for_c3": count == 0,
            }
        )

    payload = {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            "nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pstate "
            "--format=csv,noheader,nounits"
        ),
        "privacy_policy": "Process names, usernames, command lines, and GPU UUIDs are not persisted.",
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_build": torch.version.cuda,
        },
        "gpus": gpus,
        "occupied_gpu_indices": [row["index"] for row in gpus if not row["available_for_c3"]],
        "available_gpu_indices": [row["index"] for row in gpus if row["available_for_c3"]],
        "selection_policy": "Never interrupt an occupied GPU; select only an available index immediately before a run.",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gpu_count": len(gpus), "available": payload["available_gpu_indices"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
