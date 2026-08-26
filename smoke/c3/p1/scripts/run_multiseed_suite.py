#!/usr/bin/env python3
"""Run the 12 pending P1 seed825/826 jobs in two six-GPU waves."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
PLAN = P1_ROOT / "config" / "multiseed_plan.yaml"
RUNNER = P1_ROOT / "scripts" / "run_p1.py"

GPU_BY_EXPERIMENT = {
    ("neu_det", "full_sft"): "0",
    ("neu_det", "frozen_backbone"): "1",
    ("neu_det", "vpeft"): "2",
    ("deeppcb", "full_sft"): "3",
    ("deeppcb", "frozen_backbone"): "4",
    ("deeppcb", "vpeft"): "5",
}


def validate_plan(plan: dict[str, object]) -> list[dict[str, object]]:
    if plan.get("status") != "PLANNED_NOT_STARTED":
        raise ValueError("The multiseed plan is not in its immutable pre-run state")
    if plan.get("final_single_seed_epoch") != 100 or plan.get("pending_seeds") != [825, 826]:
        raise ValueError("The multiseed epoch or pending seed set changed")
    seed824 = plan.get("seed824", {})
    if seed824.get("status") != "complete" or seed824.get("rerun") is not False:
        raise ValueError("The plan no longer freezes seed824")
    runs = plan.get("runs", [])
    if not isinstance(runs, list) or len(runs) != 12:
        raise ValueError("Expected exactly 12 pending runs")
    expected = {
        (dataset, method, seed)
        for seed in (825, 826)
        for dataset, method in GPU_BY_EXPERIMENT
    }
    actual = {(run.get("dataset"), run.get("method"), run.get("seed")) for run in runs}
    if actual != expected or any(run.get("status") != "pending" for run in runs):
        raise ValueError("The pending run matrix changed")
    return runs


def validate_gpus() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 6:
        raise RuntimeError("Six CUDA GPUs are required for the locked scheduler")
    for device in range(6):
        if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 4090":
            raise RuntimeError(f"GPU {device} is not the required RTX 4090")


def main() -> int:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    runs = validate_plan(plan)
    validate_gpus()

    for run in runs:
        for root in (P1_ROOT / "logs", P1_ROOT / "artifacts"):
            target = root / str(run["run_id"])
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite planned run: {target.relative_to(REPO_ROOT)}")

    for seed in (825, 826):
        wave = [run for run in runs if run["seed"] == seed]
        processes: list[tuple[dict[str, object], str, subprocess.Popen[bytes]]] = []
        print(f"[multiseed] starting seed{seed} wave ({len(wave)} exclusive GPUs)", flush=True)
        for run in wave:
            device = GPU_BY_EXPERIMENT[(str(run["dataset"]), str(run["method"]))]
            command = [
                sys.executable,
                str(RUNNER),
                "--dataset",
                str(run["dataset"]),
                "--method",
                str(run["method"]),
                "--seed",
                str(seed),
                "--epochs",
                "100",
                "--device",
                device,
            ]
            print(f"[multiseed] GPU {device}: {run['run_id']}", flush=True)
            processes.append((run, device, subprocess.Popen(command, cwd=REPO_ROOT)))

        failures = []
        for run, device, process in processes:
            returncode = process.wait()
            outcome = "passed" if returncode == 0 else "failed"
            print(f"[multiseed] GPU {device}: {run['run_id']} {outcome} ({returncode})", flush=True)
            if returncode != 0:
                failures.append((run["run_id"], returncode))
        if failures:
            print(f"[multiseed] preserving failed wave evidence: {failures}", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
