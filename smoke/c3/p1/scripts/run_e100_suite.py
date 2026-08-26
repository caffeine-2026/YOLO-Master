#!/usr/bin/env python3
"""Run the immutable six-run seed824 100-epoch comparison suite sequentially."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
PLAN = P1_ROOT / "config" / "seed824_e100.yaml"


def main() -> int:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    if plan["execution_policy"] != "restart_all" or plan["epochs"] != 100 or plan["seed"] != 824:
        raise ValueError("The locked seed824 e100 plan is invalid")
    if plan["resume_audit"]["complete_resume_proven"] is not False:
        raise ValueError("The e100 suite must preserve the audited restart_all decision")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("GPU 0 is not the required RTX 4090 CUDA device")

    for run in plan["runs"]:
        for root in (P1_ROOT / "logs", P1_ROOT / "artifacts"):
            target = root / run["run_id"]
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite planned run: {target.relative_to(REPO_ROOT)}")

    runner = P1_ROOT / "scripts" / "run_p1.py"
    for index, run in enumerate(plan["runs"], start=1):
        command = [
            sys.executable,
            str(runner),
            "--dataset",
            run["dataset"],
            "--method",
            run["method"],
            "--seed",
            "824",
            "--epochs",
            "100",
            "--device",
            "0",
        ]
        print(f"[e100 suite {index}/6] starting {run['run_id']}", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            print(f"[e100 suite {index}/6] failed {run['run_id']}: {completed.returncode}", flush=True)
            return completed.returncode
        print(f"[e100 suite {index}/6] passed {run['run_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
