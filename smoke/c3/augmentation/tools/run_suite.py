#!/usr/bin/env python3
"""Run immutable augmentation search or confirmation jobs on explicitly idle GPUs."""

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

from smoke.c3.augmentation.tools.run_augmentation import (
    DATASETS,
    POLICIES,
    ROOT,
    assert_gpu_idle,
    selected_policy,
)

RUNNER = ROOT / "tools" / "run_augmentation.py"
EVENT_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("search", "confirm"), required=True)
    parser.add_argument("--devices", required=True)
    return parser.parse_args()


def event(path: Path, kind: str, **details: object) -> None:
    payload = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": kind, **details}
    with EVENT_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def worker(event_path: Path, phase: str, device: str, jobs: list[tuple[str, str, int, str]]) -> list[tuple[str, int]]:
    results = []
    for dataset, policy, seed, run_id in jobs:
        command = [
            sys.executable,
            str(RUNNER),
            "--phase",
            phase,
            "--dataset",
            dataset,
            "--sample-size",
            "100",
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
    args = parse_args()
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("At least one unique GPU device is required")
    event_path = ROOT / "logs" / f"{args.phase}_scheduler.jsonl"
    if event_path.exists():
        raise FileExistsError(f"Refusing to overwrite {event_path.relative_to(REPO_ROOT)}")
    gpu_evidence = [assert_gpu_idle(device) for device in devices]
    if args.phase == "search":
        jobs = [
            (dataset, policy, 824, f"search_{dataset}_100_{policy}_seed824_e100")
            for dataset in DATASETS
            for policy in POLICIES
        ]
    else:
        jobs = [
            (dataset, policy, seed, f"confirm_{dataset}_100_{policy}_seed{seed}_e100")
            for dataset in DATASETS
            for policy in ("baseline", selected_policy(dataset, "initial"))
            for seed in (825, 826)
        ]
    for *_, run_id in jobs:
        for path in (ROOT / "logs" / args.phase / run_id, ROOT / "artifacts" / args.phase / run_id):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {run_id}")
    queues = {device: [] for device in devices}
    for index, job in enumerate(jobs):
        queues[devices[index % len(devices)]].append(job)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event(
        event_path,
        "suite_start",
        phase=args.phase,
        protocol="smoke/c3/augmentation/config/augmentation_protocol.yaml",
        test_access=False,
        gpu_preflight=gpu_evidence,
        queues={device: [job[-1] for job in queue] for device, queue in queues.items()},
    )
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker, event_path, args.phase, device, queue) for device, queue in queues.items()]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    failures = sorted((run_id, code) for run_id, code in results if code)
    event(event_path, "suite_finish", completed=len(results), failures=failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
