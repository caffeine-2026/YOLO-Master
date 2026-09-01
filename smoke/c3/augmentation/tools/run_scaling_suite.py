#!/usr/bin/env python3
"""Run validation-only scaling after the three-seed trigger has passed."""

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

from smoke.c3.augmentation.tools.run_augmentation import ROOT, assert_gpu_idle

RUNNER = ROOT / "tools" / "run_augmentation.py"
LOCK = threading.Lock()


def event(path: Path, kind: str, **details: object) -> None:
    payload = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": kind, **details}
    with LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def worker(event_path: Path, device: str, jobs: list[tuple[str, int, str, int]]) -> list[tuple[str, int]]:
    results = []
    for dataset, sample_size, policy, seed in jobs:
        run_id = f"scaling_{dataset}_{sample_size}_{policy}_seed{seed}_e100"
        command = [
            sys.executable,
            str(RUNNER),
            "--phase",
            "scaling",
            "--dataset",
            dataset,
            "--sample-size",
            str(sample_size),
            "--policy",
            policy,
            "--seed",
            str(seed),
            "--device",
            device,
        ]
        event(event_path, "run_start", run_id=run_id, device=device, command=" ".join(command))
        returncode = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
        event(event_path, "run_finish", run_id=run_id, device=device, returncode=returncode)
        results.append((run_id, returncode))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True)
    args = parser.parse_args()
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("At least one unique GPU is required")
    frozen = json.loads((ROOT / "results" / "frozen_selection.json").read_text(encoding="utf-8"))
    jobs = []
    for dataset, details in frozen["datasets"].items():
        if not details["scaling_trigger_passed"]:
            continue
        for sample_size in (10, 50, 500):
            for policy in dict.fromkeys(("baseline", details["frozen_policy"])):
                jobs.extend((dataset, sample_size, policy, seed) for seed in (824, 825, 826))
    if not jobs:
        raise ValueError("No dataset passed the preregistered scaling trigger")
    event_path = ROOT / "logs" / "scaling_scheduler.jsonl"
    if event_path.exists():
        raise FileExistsError("Refusing to overwrite scaling scheduler evidence")
    preflight = [assert_gpu_idle(device) for device in devices]
    queues = {device: [] for device in devices}
    for index, job in enumerate(jobs):
        queues[devices[index % len(devices)]].append(job)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event(event_path, "suite_start", gpu_preflight=preflight, queues=queues, test_access=False)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker, event_path, device, queue) for device, queue in queues.items()]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    failures = sorted((run_id, code) for run_id, code in results if code)
    event(event_path, "suite_finish", completed=len(results), failures=failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
