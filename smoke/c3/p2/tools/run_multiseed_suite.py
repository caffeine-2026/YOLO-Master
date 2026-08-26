#!/usr/bin/env python3
"""Run only the 36 authorized P2 seed825/826 cells on eight exclusive GPUs."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
P2_ROOT = REPO_ROOT / "smoke" / "c3" / "p2"
RUNNER = P2_ROOT / "tools" / "run_scaling.py"
DATASETS = ("neu", "deeppcb")
SIZES = (10, 50, 500)
METHODS = (("full_sft", "full"), ("frozen_backbone", "frozen"), ("vpeft", "vpeft"))
SEEDS = (825, 826)
GPU_COUNT = 8
EVENTS = P2_ROOT / "logs" / "multiseed_scheduler.jsonl"
EVENT_LOCK = threading.Lock()


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event(kind: str, **details: object) -> None:
    """Append a scheduler event without allowing concurrent writes to interleave."""
    payload = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": kind, **details}
    with EVENT_LOCK, EVENTS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def preflight() -> None:
    """Reject protocol, split, reuse, GPU, or overwrite drift before launching jobs."""
    if EVENTS.exists():
        raise FileExistsError(f"Refusing to overwrite scheduler evidence: {EVENTS.relative_to(REPO_ROOT)}")
    protocol = yaml.safe_load((P2_ROOT / "config" / "protocol.yaml").read_text(encoding="utf-8"))
    gate = json.loads((P2_ROOT / "evidence" / "p2_seed824_validation.json").read_text(encoding="utf-8"))
    if protocol["multiseed_gate"]["decision"] != "MULTISEED_READY=YES":
        raise ValueError("The seed824 multi-seed gate is not ready")
    if gate["seed824_stage_status"] != "PASS" or gate["MULTISEED_READY"] != "YES":
        raise ValueError("The independent seed824 validation did not pass")
    locked = protocol["training"]
    expected = {
        "model": "yolo11n.pt",
        "epochs": 100,
        "batch": 8,
        "imgsz": 640,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0005,
        "scheduler": "cosine",
    }
    if any(locked.get(key) != value for key, value in expected.items()):
        raise ValueError("The locked P2 protocol has drifted")
    for dataset in DATASETS:
        manifest_path = P2_ROOT / "evidence" / f"{dataset}_scaling_split_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "PASS" or manifest["split_seed"] != 824:
            raise ValueError(f"Invalid split manifest: {manifest_path.relative_to(REPO_ROOT)}")
        if not all(manifest["nested_verification"].values()):
            raise ValueError(f"Nested split verification failed for {dataset}")
        for size in SIZES:
            split = manifest["splits"][str(size)]
            train_list = REPO_ROOT / split["train_list"]
            if sha256(train_list) != split["train_list_sha256"]:
                raise ValueError(f"Split hash drift: {train_list.relative_to(REPO_ROOT)}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < GPU_COUNT:
        raise RuntimeError("Eight CUDA GPUs are required")
    for device in range(GPU_COUNT):
        if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 4090":
            raise RuntimeError(f"GPU {device} is not the required RTX 4090")

    for seed in SEEDS:
        for dataset in DATASETS:
            for size in SIZES:
                for _, tag in METHODS:
                    run_id = f"{dataset}_{size}_{tag}_seed{seed}_e100"
                    for root in (P2_ROOT / "logs", P2_ROOT / "artifacts"):
                        if (root / run_id).exists():
                            raise FileExistsError(f"Refusing to overwrite {run_id}")


def jobs_by_gpu() -> dict[int, list[tuple[str, int, str, int, str]]]:
    """Build balanced static queues so every run has one exclusive GPU."""
    queues = {device: [] for device in range(GPU_COUNT)}
    large = []
    small = []
    for seed in SEEDS:
        for dataset in DATASETS:
            for size in SIZES:
                for method, tag in METHODS:
                    job = (dataset, size, method, seed, f"{dataset}_{size}_{tag}_seed{seed}_e100")
                    (large if size == 500 else small).append(job)
    for index, job in enumerate(large):
        queues[index % GPU_COUNT].append(job)
    for index, job in enumerate(small):
        queues[4 + index % 4].append(job)
    if sorted(job for queue in queues.values() for job in queue) != sorted(large + small):
        raise RuntimeError("Internal scheduler matrix mismatch")
    return queues


def worker(device: int, jobs: list[tuple[str, int, str, int, str]]) -> list[tuple[str, int]]:
    """Run one GPU's queue serially and keep all failure evidence."""
    results = []
    for dataset, size, method, seed, run_id in jobs:
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
            str(seed),
            "--device",
            str(device),
        ]
        event("run_start", run_id=run_id, device=device)
        returncode = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
        event("run_finish", run_id=run_id, device=device, returncode=returncode)
        results.append((run_id, returncode))
    return results


def main() -> int:
    preflight()
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    queues = jobs_by_gpu()
    event(
        "suite_start",
        authorized_seeds=list(SEEDS),
        sample_sizes=list(SIZES),
        new_runs=sum(map(len, queues.values())),
        gpu_queues={str(device): [job[-1] for job in jobs] for device, jobs in queues.items()},
    )
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=GPU_COUNT) as executor:
        futures = [executor.submit(worker, device, jobs) for device, jobs in queues.items()]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    failures = sorted((run_id, code) for run_id, code in results if code)
    event("suite_finish", completed=len(results), failures=failures)
    print(f"P2_MULTI_NEW_RUNS={len(results) - len(failures)}/36_PASS", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
