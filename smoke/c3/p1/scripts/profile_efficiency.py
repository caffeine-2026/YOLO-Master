#!/usr/bin/env python3
"""Low-intrusion GPU microprofile of existing P1 checkpoints without saving model changes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
from copy import deepcopy
from pathlib import Path

import torch

from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import init_seeds

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
DATASETS = ("neu_det", "deeppcb")
METHODS = ("full_sft", "frozen_backbone", "vpeft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", default="smoke/c3/p1/evidence/efficiency_profile.json")
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Only repository-relative output paths are accepted")
    resolved = (REPO_ROOT / path).resolve()
    resolved.relative_to(P1_ROOT)
    return resolved


def tensor_bytes(value: object) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def batch_for_dataset(dataset: str) -> dict[str, object]:
    init_seeds(824, deterministic=True)
    resolved = P1_ROOT / "logs" / f"{dataset}_full_sft_seed824" / "resolved_config.yaml"
    cfg = get_cfg(cfg=resolved)
    data_yaml = P1_ROOT / "config" / dataset / "dataset.yaml"
    data = check_det_dataset(str(data_yaml), autodownload=False)
    yolo_dataset = build_yolo_dataset(
        cfg,
        data["train"],
        batch=8,
        data=data,
        mode="val",
        rect=False,
        stride=32,
    )
    loader = build_dataloader(yolo_dataset, batch=8, workers=0, shuffle=False)
    return next(iter(loader))


def clone_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    cloned = deepcopy(batch)
    for key, value in cloned.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.to(device)
    cloned["img"] = cloned["img"].float() / 255
    return cloned


def batch_sha256(batch: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for key in ("img", "cls", "bboxes", "batch_idx"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def loss_scalar(output: object) -> torch.Tensor:
    loss = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"Unexpected training loss type: {type(loss)}")
    return loss.sum()


def profile_run(
    dataset: str,
    method: str,
    cpu_batch: dict[str, object],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    run_id = f"{dataset}_{method}_seed824"
    checkpoint = P1_ROOT / "artifacts" / run_id / "weights" / "last_healthy.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint.relative_to(REPO_ROOT))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = payload["model"].float().to(device).train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.0005)
    batch = clone_to_device(cpu_batch, device)

    def step(timed: bool = False) -> tuple[float, float, float, float] | None:
        optimizer.zero_grad(set_to_none=True)
        if not timed:
            loss_scalar(model(batch)).backward()
            optimizer.step()
            return None
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        loss = loss_scalar(model(batch))
        events[1].record()
        loss.backward()
        events[2].record()
        optimizer.step()
        events[3].record()
        torch.cuda.synchronize(device)
        forward = events[0].elapsed_time(events[1])
        backward = events[1].elapsed_time(events[2])
        optimizer_step = events[2].elapsed_time(events[3])
        return forward, backward, optimizer_step, events[0].elapsed_time(events[3])

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples = [step(timed=True) for _ in range(iterations)]
    assert all(sample is not None for sample in samples)
    measured = [sample for sample in samples if sample is not None]
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    adapter_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters() if "lora_" in name.lower()
    )
    result = {
        "run_id": run_id,
        "dataset": dataset,
        "method": method,
        "batch_sha256": batch_sha256(cpu_batch),
        "batch_size": int(batch["img"].shape[0]),
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "forward_ms_mean": statistics.fmean(sample[0] for sample in measured),
        "backward_ms_mean": statistics.fmean(sample[1] for sample in measured),
        "optimizer_step_ms_mean": statistics.fmean(sample[2] for sample in measured),
        "iteration_ms_mean": statistics.fmean(sample[3] for sample in measured),
        "iteration_ms_std": statistics.stdev(sample[3] for sample in measured) if iterations > 1 else 0.0,
        "baseline_allocated_mib": baseline_allocated / 1024**2,
        "baseline_reserved_mib": baseline_reserved / 1024**2,
        "peak_allocated_mib": peak_allocated / 1024**2,
        "peak_reserved_mib": peak_reserved / 1024**2,
        "peak_incremental_allocated_mib": (peak_allocated - baseline_allocated) / 1024**2,
        "peak_incremental_reserved_mib": (peak_reserved - baseline_reserved) / 1024**2,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "runtime_fp32_parameter_mib": total_parameters * 4 / 1024**2,
        "runtime_fp32_trainable_parameter_mib": trainable_parameters * 4 / 1024**2,
        "gradient_mib": trainable_parameters * 4 / 1024**2,
        "optimizer_state_mib": tensor_bytes(optimizer.state_dict()) / 1024**2,
        "adapter_parameters": adapter_parameters,
        "runtime_fp32_adapter_mib": adapter_parameters * 4 / 1024**2,
    }
    return result


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested efficiency diagnostics")
    device = torch.device(f"cuda:{int(args.device)}")
    output = repo_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output.relative_to(REPO_ROOT)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        cpu_batch = batch_for_dataset(dataset)
        for method in METHODS:
            rows.append(profile_run(dataset, method, cpu_batch, device, args.warmup, args.iterations))
            gc.collect()
            torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "scope": "Read-only checkpoint microprofile; in-memory optimizer steps are never saved.",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "precision": "FP32",
        "augmentation": "disabled for fixed-batch comparability",
        "optimizer": "fresh AdamW for step-time and state-memory diagnostics",
        "limitations": "Microprofile timings diagnose compute overhead but do not replace observed end-to-end pilot timing.",
        "runs": rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "profiles": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
