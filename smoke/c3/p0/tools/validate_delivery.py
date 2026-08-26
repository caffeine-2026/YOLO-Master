#!/usr/bin/env python3
"""验证 C3 双数据集 P0 的数据、运行证据和可发布路径策略。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_ROOT = REPO_ROOT / "smoke" / "c3"
CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
DEEPPCB_CLASSES = ("open", "short", "mousebite", "spur", "copper", "pin-hole")
REQUIRED_DELIVERY_FILES = (
    "README.md",
    "docs/ADMISSION_20260825.md",
    "docs/C3_P0_FINAL_REPORT.md",
    "PEFT_RUN_GUIDE.md",
    "config/vpeft_smoke.yaml",
    "config/datasets/deeppcb.yaml",
    "config/datasets/deeppcb_fewshot.yaml",
    "tools/prepare_deeppcb.py",
    "tools/validate_deeppcb_data.py",
    "logs/README.md",
    "evidence/gpu_smoke_comparison.json",
    "evidence/cpu_smoke_comparison.json",
    "evidence/deeppcb_source_manifest.json",
    "evidence/deeppcb_manifest.json",
    "evidence/deeppcb_data_validation.json",
    "evidence/c3_p0_summary.json",
    "evidence/environment.json",
    "evidence/environment_cpu_fallback.json",
)
RUN_EXPECTATIONS = {
    "neu_det_vpeft_cpu_seed824": ("cpu", False, False),
    "neu_det_full_sft_cpu_seed824": ("cpu", False, False),
    "neu_det_frozen_backbone_cpu_seed824": ("cpu", False, False),
    "neu_det_vpeft_gpu_seed824": ("0", True, True),
    "neu_det_full_sft_gpu_seed824": ("0", True, True),
    "neu_det_frozen_backbone_gpu_seed824": ("0", True, True),
    "neu_det_vpeft_gpu_fp32_seed824": ("0", False, False),
    "neu_det_full_sft_gpu_fp32_seed824": ("0", False, False),
    "neu_det_frozen_backbone_gpu_fp32_seed824": ("0", False, False),
    "deeppcb_vpeft_gpu_fp32_seed824": ("0", False, False),
}
OFFICIAL_P0_RUNS = {
    "neu_det_vpeft_gpu_fp32_seed824": "smoke/c3/config/datasets/neu_det_fewshot.yaml",
    "deeppcb_vpeft_gpu_fp32_seed824": "smoke/c3/config/datasets/deeppcb_fewshot.yaml",
}
REQUIRED_RUN_FILES = (
    "command.txt",
    "train.log",
    "resolved_config.yaml",
    "metrics.csv",
    "resources.csv",
    "time.txt",
    "result.json",
    "artifact_index.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/neu_det_fewshot_yolo")
    parser.add_argument("--output", default=None, help="可选的仓库相对 JSON 输出路径。")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_yaml() -> list[str]:
    results = []
    for path in sorted((SMOKE_ROOT / "config").rglob("*.yaml")):
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, dict):
            raise TypeError(f"YAML 顶层必须是映射：{path.relative_to(REPO_ROOT)}")
        results.append(path.relative_to(REPO_ROOT).as_posix())
    return results


def validate_dataset(dataset: Path, classes: tuple[str, ...] = CLASSES) -> dict[str, object]:
    split_summary = {}
    for split in ("train", "val", "test"):
        images = sorted((dataset / "images" / split).glob("*.jpg"))
        labels = sorted((dataset / "labels" / split).glob("*.txt"))
        if len(images) != len(labels):
            raise ValueError(f"{split} 图像/标签数量不一致：{len(images)} != {len(labels)}")
        object_counts: Counter[int] = Counter()
        for label in labels:
            for line_number, row in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                fields = row.split()
                if len(fields) != 5:
                    raise ValueError(f"标签列数错误：{label}:{line_number}")
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
                if class_id not in range(len(classes)):
                    raise ValueError(f"类别 ID 越界：{label}:{line_number}")
                if not all(0.0 <= value <= 1.0 for value in coordinates):
                    raise ValueError(f"坐标越界：{label}:{line_number}")
                if coordinates[2] <= 0 or coordinates[3] <= 0:
                    raise ValueError(f"检测框宽高必须大于零：{label}:{line_number}")
                object_counts[class_id] += 1
        split_summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": sum(object_counts.values()),
            "objects_by_class": {classes[index]: object_counts[index] for index in range(len(classes))},
        }
    manifest = dataset / "split_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return {"splits": split_summary, "manifest_sha256": sha256(manifest)}


def validate_deeppcb_evidence() -> dict[str, object]:
    source = json.loads((SMOKE_ROOT / "evidence" / "deeppcb_source_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((SMOKE_ROOT / "evidence" / "deeppcb_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((SMOKE_ROOT / "evidence" / "deeppcb_data_validation.json").read_text(encoding="utf-8"))
    if source.get("source_repository") != "https://github.com/tangsanli5201/DeepPCB":
        raise ValueError("DeepPCB 数据源不是已核验的原作者仓库")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("source_commit", ""))):
        raise ValueError("DeepPCB source commit 无效")
    if source.get("tested_image_count") != 1500 or source.get("annotation_count") != 1500:
        raise ValueError("DeepPCB 原始图像/标注数量不符合源审计")
    if tuple(source.get("class_names", [])) != DEEPPCB_CLASSES:
        raise ValueError("DeepPCB 源类别映射不符合预期")
    if source.get("original_split") != {"trainval": 1000, "test": 500, "overlap": 0}:
        raise ValueError("DeepPCB 官方划分审计不符合预期")
    pairing = source.get("pairing_audit", {})
    if any(pairing.get(key) != 0 for key in ("missing_tested_images", "missing_annotations", "missing_templates")):
        raise ValueError("DeepPCB 原始图像/标注/模板存在缺失")
    key_hashes = source.get("key_file_sha256", {})
    if len(key_hashes) < 4 or not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in key_hashes.values()):
        raise ValueError("DeepPCB 关键源文件 SHA-256 不完整")
    if manifest.get("source", {}).get("commit") != source.get("source_commit"):
        raise ValueError("DeepPCB 转换清单与源提交不一致")
    if manifest.get("derived_split_policy", {}).get("seed") != 824:
        raise ValueError("DeepPCB 转换 seed 不是 824")
    if manifest.get("split_image_counts") != {"train": 8, "val": 200, "test": 500}:
        raise ValueError("DeepPCB few-shot 划分数量不符合证据")
    if manifest.get("split_overlap_count") != 0:
        raise ValueError("DeepPCB train/val/test 存在重叠")
    if validation.get("status") != "PASS" or validation.get("split_overlap_count") != 0:
        raise ValueError("DeepPCB 数据 smoke 未通过")
    checks = validation.get("annotation_checks", {})
    if not checks or not all(checks.values()):
        raise ValueError("DeepPCB bbox 静态检查未全部通过")
    dataloader = validation.get("dataloader_smoke", {})
    if not dataloader.get("loaded") or dataloader.get("image_shape") != [1, 3, 320, 320]:
        raise ValueError("DeepPCB dataloader smoke 未通过")
    return {
        "source_commit": source["source_commit"],
        "source_images": source["tested_image_count"],
        "source_annotations": source["annotation_count"],
        "classes": source["class_names"],
        "split_image_counts": manifest["split_image_counts"],
        "data_validation": validation["status"],
    }


def validate_official_p0_run(run_name: str, expected_data: str) -> dict[str, object]:
    run_dir = SMOKE_ROOT / "logs" / run_name
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "vpeft_runtime_metadata.json").read_text(encoding="utf-8"))
    runtime = metadata.get("runtime_metadata", {})
    plan = runtime.get("placement_plan", {})
    train_log = (run_dir / "train.log").read_text(encoding="utf-8")
    if result.get("status") != "completed" or result.get("exit_code") != 0:
        raise ValueError(f"{run_name} 退出状态不符合 P0")
    expected_settings = {
        "data": expected_data,
        "device": "0",
        "amp": False,
        "epochs": 1,
        "batch": 1,
        "imgsz": 320,
        "seed": 824,
        "workers": 0,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_planner_enabled": True,
        "lora_planner_backend": "vpeft",
        "lora_planner_solver": "ao",
        "lora_vpeft_strict": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": resolved.get(key)}
        for key, expected in expected_settings.items()
        if resolved.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{run_name} 正式配置不符合 P0：{mismatches}")
    planned_targets = len(plan.get("targets", []))
    applied_targets = len(metadata.get("target_modules", []))
    if plan.get("status") not in {"ACCEPT", "ADAPT"} or plan.get("planner_backend") != "vpeft":
        raise ValueError(f"{run_name} strict V-PEFT 规划未生效")
    if metadata.get("backend") != "peft" or runtime.get("effective_backend") != "peft":
        raise ValueError(f"{run_name} 实际后端不是 peft")
    if planned_targets <= 0 or applied_targets <= 0:
        raise ValueError(f"{run_name} 规划/实际目标数必须大于零")
    if "CUDA:0 (NVIDIA GeForce RTX 4090" not in train_log:
        raise ValueError(f"{run_name} 没有实际使用 GPU 0 的日志证据")
    recovery = result.get("numerical_recovery", {})
    if recovery.get("detected") is not False or recovery.get("observed_epoch_attempts") != 1:
        raise ValueError(f"{run_name} 混入数值恢复或 epoch 尝试数异常")
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or not all(math.isfinite(float(value)) for value in rows[0].values()):
        raise ValueError(f"{run_name} 指标或 loss 非有限，或不是单 epoch")
    params = re.search(r"Trainable:\s*([0-9,]+).*Adapter Params:\s*([0-9,]+)", train_log)
    if not params or int(params.group(1).replace(",", "")) <= 0 or int(params.group(2).replace(",", "")) <= 0:
        raise ValueError(f"{run_name} 缺少可训练参数/适配器参数记录")
    if result.get("resource_summary", {}).get("peak_gpu_memory_mib", 0) <= 0 or result.get("wall_seconds", 0) <= 0:
        raise ValueError(f"{run_name} 缺少显存峰值或总耗时")
    postprocess = (run_dir / "postprocess.txt").read_text(encoding="utf-8")
    if "adapter_export=completed" not in postprocess:
        raise ValueError(f"{run_name} 独立 adapter 导出失败")
    artifact_index = (run_dir / "artifact_index.txt").read_text(encoding="utf-8")
    artifacts = result.get("artifacts", [])
    suffixes = [str(item.get("path", "")) for item in artifacts]
    required_suffixes = ("weights/best.pt", "weights/last.pt", "lora_adapter/adapter_model.safetensors")
    if not all(any(path.endswith(suffix) for path in suffixes) for suffix in required_suffixes):
        raise ValueError(f"{run_name} checkpoint/adapter 产物不完整")
    for item in artifacts:
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute():
            raise ValueError(f"{run_name} 产物索引包含绝对路径")
        artifact = (REPO_ROOT / relative).resolve()
        try:
            artifact.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError(f"{run_name} 产物路径越出仓库") from exc
        if (
            not artifact.is_file()
            or artifact.stat().st_size != item.get("size_bytes")
            or sha256(artifact) != item.get("sha256")
        ):
            raise ValueError(f"{run_name} 产物不存在或 SHA-256 不一致：{relative}")
        if str(item.get("sha256")) not in artifact_index:
            raise ValueError(f"{run_name} artifact_index 缺少产物哈希：{relative}")
    return {
        "planner_status": plan["status"],
        "planner_backend": plan["planner_backend"],
        "actual_backend": runtime["effective_backend"],
        "planned_targets": planned_targets,
        "applied_targets": applied_targets,
        "trainable_parameters": int(params.group(1).replace(",", "")),
        "adapter_parameters": int(params.group(2).replace(",", "")),
        "peak_gpu_memory_mib": result["resource_summary"]["peak_gpu_memory_mib"],
        "elapsed_seconds": result["wall_seconds"],
        "exit_code": result["exit_code"],
        "status": "PASS",
    }


def validate_publication_paths() -> list[str]:
    forbidden = re.compile("/" + "home/" + r"[^/]+/|[A-Za-z]:" + r"\\Users\\")
    deprecated_labels = re.compile(r"\b(?:" + "C" + r"0|G" + r"0(?:-[A-Z0-9]+)?)\b")
    checked = []
    for path in sorted(item for item in SMOKE_ROOT.rglob("*") if item.is_file()):
        if path.suffix.lower() not in {".md", ".yaml", ".yml", ".json", ".txt", ".csv", ".log", ".py"}:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        if forbidden.search(text):
            raise ValueError(f"发现本机用户绝对路径：{path.relative_to(REPO_ROOT)}")
        if deprecated_labels.search(text):
            raise ValueError(f"发现未定义的旧验收标签：{path.relative_to(REPO_ROOT)}")
        checked.append(path.relative_to(REPO_ROOT).as_posix())
    return checked


def validate_delivery_artifacts() -> dict[str, object]:
    missing = [path for path in REQUIRED_DELIVERY_FILES if not (SMOKE_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少交付文件：{missing}")
    admission_path = SMOKE_ROOT / "docs" / "ADMISSION_20260825.md"
    final_report_path = SMOKE_ROOT / "docs" / "C3_P0_FINAL_REPORT.md"
    docs_markdown = sorted((SMOKE_ROOT / "docs").glob("*.md"))
    if docs_markdown != sorted((admission_path, final_report_path)):
        raise ValueError(f"docs 中只允许历史验收文档和 C3 P0 最终报告：{docs_markdown}")
    admission = admission_path.read_text(encoding="utf-8")
    required_admission_sections = (
        "环境安装",
        "基线/最小任务",
        "复现命令",
        "配置文件",
        "完整日志",
        "结果证据",
        "设计说明",
        "风险与降级",
    )
    missing_sections = [section for section in required_admission_sections if section not in admission]
    if missing_sections:
        raise ValueError(f"统一交付文档缺少必需章节：{missing_sections}")
    final_report = final_report_path.read_text(encoding="utf-8")
    if "Official C3 P0" not in final_report or "NEU-DET" not in final_report or "DeepPCB" not in final_report:
        raise ValueError("C3 P0 最终报告未明确双数据集官方范围")

    runs = {}
    for run_name, (expected_device, expected_amp, expected_recovery) in RUN_EXPECTATIONS.items():
        run_dir = SMOKE_ROOT / "logs" / run_name
        missing_run_files = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
        if missing_run_files:
            raise FileNotFoundError(f"{run_name} 缺少证据：{missing_run_files}")
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        if result.get("status") != "completed" or result.get("exit_code") != 0:
            raise ValueError(f"{run_name} 未正常完成")
        if result.get("device") != expected_device:
            raise ValueError(f"{run_name} 设备标签不符合预期：{result.get('device')} != {expected_device}")
        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        if bool(resolved.get("amp")) is not expected_amp:
            raise ValueError(f"{run_name} AMP 最终配置不符合预期")
        peak_gpu_memory = result.get("resource_summary", {}).get("peak_gpu_memory_mib")
        if expected_device == "cpu" and peak_gpu_memory is not None:
            raise ValueError(f"{run_name} 不得把 CPU 资源写成 GPU 显存")
        if expected_device != "cpu" and (peak_gpu_memory is None or peak_gpu_memory <= 0):
            raise ValueError(f"{run_name} 缺少 GPU 显存实测值")
        recovery = result.get("numerical_recovery", {})
        if recovery.get("detected") is not expected_recovery:
            raise ValueError(f"{run_name} 数值恢复状态不符合预期")
        if expected_recovery and recovery.get("recovery_attempts") != 1:
            raise ValueError(f"{run_name} AMP 降级运行必须准确记录一次恢复")
        if not (run_dir / "train.log").stat().st_size:
            raise ValueError(f"{run_name} 完整日志为空")
        runs[run_name] = {
            "status": result["status"],
            "exit_code": result["exit_code"],
            "device": result["device"],
            "amp": bool(resolved.get("amp")),
            "peak_gpu_memory_mib": peak_gpu_memory,
            "numerical_recovery_detected": recovery.get("detected"),
            "recovery_attempts": recovery.get("recovery_attempts"),
        }

    vpeft_files = ("adapter_config.json", "vpeft_runtime_metadata.json", "postprocess.txt")
    for run_name in (name for name in RUN_EXPECTATIONS if "vpeft" in name):
        vpeft_dir = SMOKE_ROOT / "logs" / run_name
        missing_vpeft = [name for name in vpeft_files if not (vpeft_dir / name).is_file()]
        if missing_vpeft:
            raise FileNotFoundError(f"{run_name} 缺少 V-PEFT 专用证据：{missing_vpeft}")
        metadata = json.loads((vpeft_dir / "vpeft_runtime_metadata.json").read_text(encoding="utf-8"))
        runtime = metadata.get("runtime_metadata", {})
        plan = runtime.get("placement_plan", {})
        if plan.get("status") not in {"ACCEPT", "ADAPT"}:
            raise ValueError(f"{run_name} V-PEFT 规划状态不是 ACCEPT/ADAPT")
        if runtime.get("effective_backend") != "peft" or metadata.get("backend") != "peft":
            raise ValueError(f"{run_name} V-PEFT 实际后端不是 peft")
        if len(plan.get("targets", [])) != 59 or len(metadata.get("target_modules", [])) != 52:
            raise ValueError(f"{run_name} V-PEFT 规划/应用目标数不是 59/52")
        if "adapter_export=completed" not in (vpeft_dir / "postprocess.txt").read_text(encoding="utf-8"):
            raise ValueError(f"{run_name} V-PEFT 独立适配器未成功导出")

    official_p0_runs = {
        run_name: validate_official_p0_run(run_name, expected_data)
        for run_name, expected_data in OFFICIAL_P0_RUNS.items()
    }
    p0_summary = json.loads((SMOKE_ROOT / "evidence" / "c3_p0_summary.json").read_text(encoding="utf-8"))
    p0 = p0_summary.get("official_p0", {})
    datasets = p0.get("datasets", [])
    if p0.get("status") != "PASS" or len(datasets) != 2:
        raise ValueError("Official C3 P0 汇总不是双数据集 PASS")
    if {row.get("dataset") for row in datasets if row.get("status") == "PASS"} != {"NEU-DET", "DeepPCB"}:
        raise ValueError("Official C3 P0 汇总缺少 NEU-DET/DeepPCB PASS")

    gpu_comparison = json.loads((SMOKE_ROOT / "evidence" / "gpu_smoke_comparison.json").read_text(encoding="utf-8"))
    if gpu_comparison.get("gpu_fp32", {}).get("status") != "PASS":
        raise ValueError("GPU FP32 汇总状态必须是 PASS")
    if gpu_comparison.get("gpu_amp", {}).get("status") != "DEGRADED":
        raise ValueError("GPU AMP 汇总状态必须是 DEGRADED")
    if len(gpu_comparison["gpu_fp32"].get("runs", [])) != 3 or len(gpu_comparison["gpu_amp"].get("runs", [])) != 3:
        raise ValueError("GPU FP32/AMP 汇总都必须包含三种方案")

    environment = json.loads((SMOKE_ROOT / "evidence" / "environment.json").read_text(encoding="utf-8"))
    cuda = environment["cuda"]
    required_cuda_fields = (
        "gpu_execution_ready",
        "pci_nvidia_gpu_count",
        "required_device_nodes_present",
        "nvidia_smi_exit_code",
        "torch_cuda_available",
        "allocation_test",
        "current_classification",
    )
    missing_cuda_fields = [field for field in required_cuda_fields if field not in cuda]
    if missing_cuda_fields:
        raise KeyError(f"环境证据缺少 GPU 判定字段：{missing_cuda_fields}")
    if not all(
        (
            cuda["gpu_execution_ready"],
            cuda["required_device_nodes_present"],
            cuda["nvidia_smi_exit_code"] == 0,
            cuda["torch_cuda_available"],
            cuda["device_count"] > 0,
            cuda["allocation_test"]["passed"],
        )
    ):
        raise ValueError("最终环境证据未通过 GPU 入口门")
    gpu_gate = {
        "gpu_execution_ready": cuda["gpu_execution_ready"],
        "classification": cuda["current_classification"],
        "pci_nvidia_gpu_count": cuda["pci_nvidia_gpu_count"],
        "required_device_nodes_present": cuda["required_device_nodes_present"],
        "nvidia_smi_exit_code": cuda["nvidia_smi_exit_code"],
        "torch_cuda_available": cuda["torch_cuda_available"],
        "allocation_test_passed": cuda["allocation_test"]["passed"],
    }
    return {
        "required_files": list(REQUIRED_DELIVERY_FILES),
        "runs": runs,
        "official_p0_runs": official_p0_runs,
        "official_p0_status": p0["status"],
        "gpu_gate": gpu_gate,
    }


def main() -> int:
    args = parse_args()
    dataset = (REPO_ROOT / args.dataset).resolve()
    try:
        dataset.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("数据集路径必须位于仓库内") from exc
    report = {
        "schema_version": 3,
        "status": "passed",
        "yaml_files": validate_yaml(),
        "dataset": validate_dataset(dataset),
        "deeppcb_dataset": validate_dataset(REPO_ROOT / "datasets" / "deeppcb_fewshot_yolo", DEEPPCB_CLASSES),
        "deeppcb_evidence": validate_deeppcb_evidence(),
        "delivery": validate_delivery_artifacts(),
        "publishable_text_files": validate_publication_paths(),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = (REPO_ROOT / args.output).resolve()
        try:
            output.relative_to(SMOKE_ROOT)
        except ValueError as exc:
            raise ValueError("验证报告必须写入 smoke/c3 目录") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
