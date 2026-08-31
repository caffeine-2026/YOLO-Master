#!/usr/bin/env python3
"""Run the locked 24-cell selected V-PEFT matrix on explicitly idle GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smoke.c3.completion.tools.run_efficiency import DATASETS, ROOT, SEEDS, SIZES, assert_gpu_idle, selected_candidate

RUNNER = ROOT / "tools" / "run_efficiency.py"
EVENTS = ROOT / "logs" / "final_scheduler.jsonl"
EVENT_LOCK = threading.Lock()


def event(kind: str, **details: object) -> None:
    payload = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": kind, **details}
    with EVENT_LOCK, EVENTS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True)
    return parser.parse_args()


def worker(device: str, jobs: list[tuple[str, int, int, str]]) -> list[tuple[str, int]]:
    results = []
    for dataset, size, seed, run_id in jobs:
        command = [
            sys.executable,
            str(RUNNER),
            "--phase",
            "final",
            "--dataset",
            dataset,
            "--sample-size",
            str(size),
            "--seed",
            str(seed),
            "--device",
            device,
        ]
        event("run_start", run_id=run_id, device=device)
        returncode = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
        event("run_finish", run_id=run_id, device=device, returncode=returncode)
        results.append((run_id, returncode))
    return results


def main() -> int:
    args = parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("At least one unique GPU device is required")
    if EVENTS.exists():
        raise FileExistsError("Refusing to overwrite final scheduler evidence")
    candidate = selected_candidate()
    gpu_evidence = [assert_gpu_idle(device) for device in devices]
    # Largest cells first gives balanced queues while every GPU remains exclusive.
    jobs = [
        (dataset, size, seed, f"final_{dataset}_{size}_vpeft_{candidate}_seed{seed}_e100")
        for size in sorted(SIZES, reverse=True)
        for dataset in DATASETS
        for seed in SEEDS
    ]
    if len(jobs) != 24:
        raise RuntimeError("Final V-PEFT matrix must contain exactly 24 cells")
    for *_, run_id in jobs:
        for path in (ROOT / "logs" / "final" / run_id, ROOT / "artifacts" / "final" / run_id):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {run_id}")
    queues = {device: [] for device in devices}
    for index, job in enumerate(jobs):
        queues[devices[index % len(devices)]].append(job)
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    event(
        "suite_start",
        selected_candidate=candidate,
        selection_path="smoke/c3/completion/results/efficiency_selection.json",
        run_count=len(jobs),
        gpu_preflight=gpu_evidence,
        queues={device: [job[-1] for job in queue] for device, queue in queues.items()},
    )
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker, device, queue) for device, queue in queues.items()]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    failures = sorted((run_id, code) for run_id, code in results if code)
    event("suite_finish", completed=len(results), failures=failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
