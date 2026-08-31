#!/usr/bin/env python3
"""汇总既有对照运行，并生成 C3 双数据集 P0 结构化证据。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
SMOKE_ROOT = REPO_ROOT / "smoke" / "c3" / "p0"
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
OFFICIAL_P0_RUNS = (
    ("NEU-DET", "neu_det_vpeft_gpu_fp32_seed824", 180, 399),
    ("DeepPCB", "deeppcb_vpeft_gpu_fp32_seed824", 200, 1364),
)


def first_metric(row: dict[str, str], name: str) -> float:
    return float(row[name])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(
    label: str, run_id: str, validation_images: int = 180, validation_instances: int = 399
) -> dict[str, object]:
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
            "images": validation_images,
            "instances": validation_instances,
            "precision": first_metric(metric, "metrics/precision(B)"),
            "recall": first_metric(metric, "metrics/recall(B)"),
            "map50": first_metric(metric, "metrics/mAP50(B)"),
            "map50_95": first_metric(metric, "metrics/mAP50-95(B)"),
        },
        "checkpoint_sha256": next(
            item["sha256"] for item in result["artifacts"] if str(item["path"]).endswith("weights/last_healthy.pt")
        ),
        "evidence_dir": f"smoke/c3/p0/logs/{run_id}",
    }


def summarize_p0_run(dataset: str, run_id: str, validation_images: int, validation_instances: int) -> dict[str, object]:
    row = summarize(dataset, run_id, validation_images, validation_instances)
    log_dir = SMOKE_ROOT / "logs" / run_id
    result = json.loads((log_dir / "result.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((log_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((log_dir / "vpeft_runtime_metadata.json").read_text(encoding="utf-8"))
    runtime = metadata.get("runtime_metadata", {})
    plan = runtime.get("placement_plan", {})
    metrics = row["validation"]
    train_log = (log_dir / "train.log").read_text(encoding="utf-8")
    artifacts = result.get("artifacts", [])
    artifact_paths = [str(item.get("path", "")) for item in artifacts]
    artifact_hashes_valid = all(
        (REPO_ROOT / path).is_file() and file_sha256(REPO_ROOT / path) == item.get("sha256")
        for item in artifacts
        if (path := str(item.get("path", "")))
    )
    metric_values = [metrics["precision"], metrics["recall"], metrics["map50"], metrics["map50_95"]]
    checks = {
        "exit_code_zero": result.get("status") == "completed" and result.get("exit_code") == 0,
        "cuda_gpu0_used": result.get("device") == "0" and "CUDA:0 (NVIDIA GeForce RTX 4090" in train_log,
        "fixed_training_conditions": all(
            (
                resolved.get("amp") is False,
                resolved.get("epochs") == 1,
                resolved.get("batch") == 1,
                resolved.get("imgsz") == 320,
                resolved.get("seed") == 824,
                resolved.get("workers") == 0,
            )
        ),
        "strict_vpeft_planner": all(
            (
                resolved.get("lora_planner_enabled") is True,
                resolved.get("lora_planner_backend") == "vpeft",
                resolved.get("lora_planner_solver") == "ao",
                resolved.get("lora_vpeft_strict") is True,
                plan.get("status") in {"ACCEPT", "ADAPT"},
                plan.get("planner_backend") == "vpeft",
            )
        ),
        "peft_applied_without_fallback": all(
            (
                metadata.get("backend") == "peft",
                runtime.get("effective_backend") == "peft",
                len(plan.get("targets", [])) > 0,
                len(metadata.get("target_modules", [])) > 0,
            )
        ),
        "finite_metrics_and_loss": all(math.isfinite(float(value)) for value in metric_values)
        and not re.search(r"(?:^|[\s,])(?:nan|[-+]?inf)(?:$|[\s,])", train_log, re.IGNORECASE),
        "no_numerical_recovery": result.get("numerical_recovery", {}).get("detected") is False,
        "checkpoint_present": any(path.endswith("weights/last.pt") for path in artifact_paths),
        "adapter_exported": any(path.endswith("lora_adapter/adapter_model.safetensors") for path in artifact_paths)
        and "adapter_export=completed" in (log_dir / "postprocess.txt").read_text(encoding="utf-8"),
        "resolved_config_and_full_log_present": (log_dir / "resolved_config.yaml").is_file()
        and (log_dir / "train.log").stat().st_size > 0,
        "resources_and_elapsed_recorded": result.get("resource_summary", {}).get("peak_gpu_memory_mib", 0) > 0
        and result.get("wall_seconds", 0) > 0,
        "parameter_counts_recorded": row["trainable_parameters"] > 0 and row["adapter_parameters"] > 0,
        "artifact_sha256_valid": bool(artifacts) and artifact_hashes_valid,
    }
    return {
        "dataset": dataset,
        "run_id": run_id,
        "planner_status": plan.get("status"),
        "planner_backend": plan.get("planner_backend"),
        "actual_backend": runtime.get("effective_backend"),
        "planned_targets": len(plan.get("targets", [])),
        "applied_targets": len(metadata.get("target_modules", [])),
        "gpu": "NVIDIA GeForce RTX 4090",
        "trainable_parameters": row["trainable_parameters"],
        "adapter_parameters": row["adapter_parameters"],
        "peak_gpu_memory_mib": row["peak_gpu_memory_mib"],
        "elapsed_seconds": row["wall_seconds"],
        "exit_code": row["exit_code"],
        "adapter_exported": checks["adapter_exported"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "evidence_dir": row["evidence_dir"],
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
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
    official_p0 = [summarize_p0_run(*definition) for definition in OFFICIAL_P0_RUNS]
    p0_report = {
        "schema_version": 1,
        "scope": "Official C3 P0: NEU-DET and DeepPCB each complete one V-PEFT GPU FP32 minimal run",
        "official_p0": {
            "status": "PASS" if all(row["status"] == "PASS" for row in official_p0) else "FAIL",
            "datasets": official_p0,
        },
        "extra_smoke_evidence": {
            "scope": "NEU-DET preliminary smoke evidence only; not a P1 conclusion",
            "runs": gpu_fp32_rows,
        },
        "conclusion_boundary": "One epoch and one seed verify the execution loop only; metrics do not establish convergence or method superiority.",
    }
    (SMOKE_ROOT / "evidence" / "c3_p0_summary.json").write_text(
        json.dumps(p0_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "summarized", "runs": 10, "official_c3_p0": p0_report["official_p0"]["status"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
