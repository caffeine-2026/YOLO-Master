#!/usr/bin/env python3
"""Run the preregistered eight-cell validation-only efficiency search."""

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

from smoke.c3.completion.tools.run_efficiency import CANDIDATES, DATASETS, ROOT, assert_gpu_idle

RUNNER = ROOT / "tools" / "run_efficiency.py"
EVENTS = ROOT / "logs" / "search_scheduler.jsonl"
EVENT_LOCK = threading.Lock()


def event(kind: str, **details: object) -> None:
    payload = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": kind, **details}
    with EVENT_LOCK, EVENTS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices", required=True, help="Comma-separated GPU indices proven idle immediately before launch"
    )
    return parser.parse_args()


def worker(device: str, jobs: list[tuple[str, str, str]]) -> list[tuple[str, int]]:
    results = []
    for dataset, candidate, run_id in jobs:
        command = [
            sys.executable,
            str(RUNNER),
            "--phase",
            "search",
            "--dataset",
            dataset,
            "--sample-size",
            "100",
            "--seed",
            "824",
            "--candidate",
            candidate,
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
        raise FileExistsError("Refusing to overwrite search scheduler evidence")
    gpu_evidence = [assert_gpu_idle(device) for device in devices]
    jobs = [
        (dataset, candidate, f"search_{dataset}_100_{candidate}_seed824")
        for dataset in DATASETS
        for candidate in CANDIDATES
    ]
    for _, _, run_id in jobs:
        for path in (ROOT / "logs" / "search" / run_id, ROOT / "artifacts" / "search" / run_id):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {run_id}")
    queues = {device: [] for device in devices}
    for index, job in enumerate(jobs):
        queues[devices[index % len(devices)]].append(job)
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    event(
        "suite_start",
        protocol="smoke/c3/completion/config/efficiency_search_protocol.yaml",
        test_access=False,
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
