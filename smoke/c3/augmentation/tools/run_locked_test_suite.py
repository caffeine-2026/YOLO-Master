#!/usr/bin/env python3
"""Schedule the post-freeze locked test evaluations on explicitly idle GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smoke.c3.augmentation.tools.run_augmentation import ROOT, assert_gpu_idle
from smoke.c3.p1.scripts import run_p1 as common

RUNNER = ROOT / "tools" / "evaluate_locked_test.py"
LOCK = threading.Lock()


def event(path: Path, kind: str, **details: object) -> None:
    payload = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": kind, **details}
    with LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def worker(event_path: Path, device: str, jobs: list[tuple[str, str, int, int]]) -> list[tuple[str, int]]:
    results = []
    for dataset, policy, sample_size, seed in jobs:
        eval_id = f"test_{dataset}_{sample_size}_{policy}_seed{seed}"
        log_dir = ROOT / "logs" / "locked_test" / eval_id
        if log_dir.exists():
            raise FileExistsError(f"Refusing to overwrite {common.relative(log_dir)}")
        log_dir.mkdir(parents=True)
        command = [
            sys.executable,
            str(RUNNER),
            "--dataset",
            dataset,
            "--policy",
            policy,
            "--sample-size",
            str(sample_size),
            "--seed",
            str(seed),
            "--device",
            device,
        ]
        (log_dir / "command.txt").write_text(
            " ".join(["python", common.relative(RUNNER), *command[2:]]) + "\n", encoding="utf-8"
        )
        common.json_write(log_dir / "environment.json", common.environment_evidence(device))
        event(event_path, "evaluation_start", eval_id=eval_id, device=device, command=" ".join(command))
        exit_code, seconds = common.run_captured(
            command,
            log_dir / "stdout.log",
            log_dir / "stderr.log",
            device,
            log_dir / "resource_samples.csv",
        )
        common.json_write(
            log_dir / "timing.json",
            {"evaluation_seconds": round(seconds, 3), "exit_code": exit_code, "gpu_hours": round(seconds / 3600, 8)},
        )
        common.json_write(log_dir / "resource_usage.json", common.summarize_samples(log_dir / "resource_samples.csv"))
        event(event_path, "evaluation_finish", eval_id=eval_id, device=device, returncode=exit_code)
        results.append((eval_id, exit_code))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--scope", choices=("100", "retry-100-medium", "scaling"), required=True)
    args = parser.parse_args()
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("At least one unique GPU is required")
    frozen = json.loads((ROOT / "results" / "frozen_selection.json").read_text(encoding="utf-8"))
    if frozen.get("test_access_before_selection") is not False:
        raise ValueError("Missing validation-only freeze evidence")
    jobs: list[tuple[str, str, int, int]] = []
    if args.scope == "retry-100-medium":
        details = frozen["datasets"]["deeppcb"]
        jobs = [("deeppcb", details["frozen_policy"], 100, seed) for seed in (824, 825, 826)]
    for dataset, details in frozen["datasets"].items():
        if args.scope == "retry-100-medium":
            continue
        sizes = (100,) if args.scope == "100" else (10, 50, 500)
        if args.scope == "scaling" and not details["scaling_trigger_passed"]:
            continue
        for sample_size in sizes:
            for policy in dict.fromkeys(("baseline", details["frozen_policy"])):
                jobs.extend((dataset, policy, sample_size, seed) for seed in (824, 825, 826))
    if not jobs:
        raise ValueError(f"No locked test jobs were enabled for scope={args.scope}")
    event_path = ROOT / "logs" / f"locked_test_{args.scope}_scheduler.jsonl"
    if event_path.exists():
        raise FileExistsError(f"Refusing to overwrite {common.relative(event_path)}")
    preflight = [assert_gpu_idle(device) for device in devices]
    queues = {device: [] for device in devices}
    for index, job in enumerate(jobs):
        queues[devices[index % len(devices)]].append(job)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event(event_path, "suite_start", gpu_preflight=preflight, jobs=jobs, selection_status=frozen["status"])
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker, event_path, device, queue) for device, queue in queues.items()]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    failures = sorted((eval_id, code) for eval_id, code in results if code)
    event(event_path, "suite_finish", completed=len(results), failures=failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
