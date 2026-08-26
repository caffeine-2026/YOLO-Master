#!/usr/bin/env python3
"""Run P2 seed824 10/50/500 cells in three exclusive six-GPU waves."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
P2_ROOT = REPO_ROOT / "smoke" / "c3" / "p2"
RUNNER = P2_ROOT / "tools" / "run_scaling.py"
GPU_BY_EXPERIMENT = {
    ("neu", "full_sft"): "0",
    ("neu", "frozen_backbone"): "1",
    ("neu", "vpeft"): "2",
    ("deeppcb", "full_sft"): "3",
    ("deeppcb", "frozen_backbone"): "4",
    ("deeppcb", "vpeft"): "5",
}
METHOD_TAGS = {"full_sft": "full", "frozen_backbone": "frozen", "vpeft": "vpeft"}


def main() -> int:
    protocol = yaml.safe_load((P2_ROOT / "config" / "protocol.yaml").read_text(encoding="utf-8"))
    reuse = json.loads((P2_ROOT / "evidence" / "p1_100_reuse_audit.json").read_text(encoding="utf-8"))
    if protocol["stage"] != "seed824_initial_scaling" or protocol["training"]["seeds_this_stage"] != [824]:
        raise ValueError("The P2 stage is not locked to seed824")
    if protocol["multiseed_gate"]["auto_run_seed825_826"] is not False:
        raise ValueError("The seed825/826 prohibition changed")
    if reuse["decision"] != "REUSE_P1_100_SEED824" or reuse["reused_cells"] != 6:
        raise ValueError("P1 100-cell reuse has not passed")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 6:
        raise RuntimeError("Six CUDA GPUs are required")
    for device in range(6):
        if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 4090":
            raise RuntimeError(f"GPU {device} is not the required RTX 4090")

    planned = []
    for size in (10, 50, 500):
        for (dataset, method), device in GPU_BY_EXPERIMENT.items():
            run_id = f"{dataset}_{size}_{METHOD_TAGS[method]}_seed824"
            planned.append((size, dataset, method, device, run_id))
            for root in (P2_ROOT / "logs", P2_ROOT / "artifacts"):
                if (root / run_id).exists():
                    raise FileExistsError(f"Refusing to overwrite {run_id}")

    for size in (10, 50, 500):
        processes = []
        print(f"[P2] starting sample_size={size} wave", flush=True)
        for _, dataset, method, device, run_id in [row for row in planned if row[0] == size]:
            command = [
                sys.executable,
                str(RUNNER),
                "--dataset",
                dataset,
                "--sample-size",
                str(size),
                "--method",
                method,
                "--seed",
                "824",
                "--device",
                device,
            ]
            print(f"[P2] GPU {device}: {run_id}", flush=True)
            processes.append((run_id, device, subprocess.Popen(command, cwd=REPO_ROOT)))
        failures = []
        for run_id, device, process in processes:
            returncode = process.wait()
            print(f"[P2] GPU {device}: {run_id} returncode={returncode}", flush=True)
            if returncode:
                failures.append((run_id, returncode))
        if failures:
            print(f"[P2] preserving failure evidence: {failures}", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
