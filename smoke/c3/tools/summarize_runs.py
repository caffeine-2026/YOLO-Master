#!/usr/bin/env python3
"""汇总 CPU、GPU FP32 与 GPU AMP 三方运行的结构化证据。"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_ROOT = REPO_ROOT / "smoke" / "c3"
RUN_ROOT = REPO_ROOT / "runs" / "detect" / "runs" / "vpeft_smoke"
CPU_RUNS = (
    ("V-PEFT", "neu_det_vpeft_cpu_seed824"),
    ("全参数 Full-SFT", "neu_det_full_sft_cpu_seed824"),
    ("冻结主干", "neu_det_frozen_backbone_cpu_seed824"),
)
GPU_FP32_RUNS = (
    ("V-PEFT", "neu_det_vpeft_gpu_fp32_seed824"),
    ("全参数 Full-SFT", "neu_det_full_sft_gpu_fp32_seed824"),
    ("冻结主干", "neu_det_frozen_backbone_gpu_fp32_seed824"),
)
GPU_AMP_RUNS = (
    ("V-PEFT", "neu_det_vpeft_gpu_seed824"),
    ("全参数 Full-SFT", "neu_det_full_sft_gpu_seed824"),
    ("冻结主干", "neu_det_frozen_backbone_gpu_seed824"),
)
ADAPTER_MARKERS = ("lora_", "hada_", "lokr_", "oft_", "boft_", "ia3_", "hra_")


def first_metric(row: dict[str, str], name: str) -> float:
    return float(row[name])


def summarize(label: str, run_id: str) -> dict[str, object]:
    log_dir = SMOKE_ROOT / "logs" / run_id
    run_dir = RUN_ROOT / run_id
    result = json.loads((log_dir / "result.json").read_text(encoding="utf-8"))
    with (log_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        metric = next(csv.DictReader(stream))
    checkpoint = run_dir / "weights" / "last_healthy.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = payload["model"]
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    adapter = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if any(marker in name for marker in ADAPTER_MARKERS)
    )
    metadata = getattr(model, "lora_runtime_metadata", {}) or {}
    placement_plan = metadata.get("placement_plan") or {}
    train_log = (log_dir / "train.log").read_text(encoding="utf-8")
    speed_matches = re.findall(r"30/30\s+([0-9.]+)it/s", train_log)
    return {
        "label": label,
        "run_id": run_id,
        "status": result["status"],
        "exit_code": result["exit_code"],
        "device": result["device"],
        "wall_seconds": result["wall_seconds"],
        "train_iterations_per_second": float(speed_matches[-1]) if speed_matches else None,
        "peak_rss_mib": result["resource_summary"]["peak_rss_mib"],
        "peak_gpu_memory_mib": result["resource_summary"]["peak_gpu_memory_mib"],
        "numerical_recovery": result.get("numerical_recovery"),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": round(trainable / total * 100, 4),
        "adapter_parameters": adapter,
        "vpeft": {
            "enabled": bool(getattr(model, "lora_enabled", False)),
            "effective_backend": metadata.get("effective_backend"),
            "plan_status": placement_plan.get("status"),
            "planned_target_count": len(placement_plan.get("targets", [])),
            "applied_target_count": len(metadata.get("target_modules", [])),
        },
        "validation": {
            "images": 180,
            "instances": 399,
            "precision": first_metric(metric, "metrics/precision(B)"),
            "recall": first_metric(metric, "metrics/recall(B)"),
            "map50": first_metric(metric, "metrics/mAP50(B)"),
            "map50_95": first_metric(metric, "metrics/mAP50-95(B)"),
        },
        "checkpoint_sha256": next(
            item["sha256"] for item in result["artifacts"] if str(item["path"]).endswith("weights/last_healthy.pt")
        ),
        "evidence_dir": f"smoke/c3/logs/{run_id}",
    }


def main() -> int:
    cpu_rows = [summarize(label, run_id) for label, run_id in CPU_RUNS]
    gpu_fp32_rows = [summarize(label, run_id) for label, run_id in GPU_FP32_RUNS]
    gpu_amp_rows = [summarize(label, run_id) for label, run_id in GPU_AMP_RUNS]
    cpu_report = {
        "schema_version": 2,
        "scope": "NEU-DET 5-shot、1 epoch、batch 1、imgsz 320、seed 824 的 CPU 功能验证",
        "warning": "该结果只验证流程连通性；单轮 CPU 运行不能用于得出精度或 GPU 性能结论。",
        "runs": cpu_rows,
    }
    gpu_report = {
        "schema_version": 2,
        "scope": "RTX 4090 GPU 0、NEU-DET 5-shot、1 epoch、batch 1、imgsz 320、seed 824",
        "gpu_fp32": {
            "status": "PASS",
            "amp": False,
            "warning": "三种运行均未检测到自动数值恢复；单轮指标仍不能作为收敛或优劣结论。",
            "runs": gpu_fp32_rows,
        },
        "gpu_amp": {
            "status": "DEGRADED",
            "amp": True,
            "warning": "三种运行都触发一次非有限值恢复，训练器随后关闭 AMP 重试；耗时和指标不得作为纯 AMP 基准。",
            "runs": gpu_amp_rows,
        },
        "gpu_memory_measurement": "每秒采样 GPU 0 的设备级 memory.used；数值包含空闲基线，可能漏过不足一秒的瞬时峰值。",
    }
    (SMOKE_ROOT / "evidence" / "cpu_smoke_comparison.json").write_text(
        json.dumps(cpu_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (SMOKE_ROOT / "evidence" / "gpu_smoke_comparison.json").write_text(
        json.dumps(gpu_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "summarized", "runs": 9}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
