#!/usr/bin/env python3
"""验证 C3 配置、NEU-DET 标签、清单和可发布路径策略。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_ROOT = REPO_ROOT / "smoke" / "c3"
CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
REQUIRED_DELIVERY_FILES = (
    "README.md",
    "docs/ADMISSION_20260825.md",
    "PEFT_RUN_GUIDE.md",
    "config/vpeft_smoke.yaml",
    "logs/README.md",
    "evidence/gpu_smoke_comparison.json",
    "evidence/cpu_smoke_comparison.json",
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


def validate_dataset(dataset: Path) -> dict[str, object]:
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
                if class_id not in range(len(CLASSES)):
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
            "objects_by_class": {CLASSES[index]: object_counts[index] for index in range(len(CLASSES))},
        }
    manifest = dataset / "split_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return {"splits": split_summary, "manifest_sha256": sha256(manifest)}


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
    docs_markdown = sorted((SMOKE_ROOT / "docs").glob("*.md"))
    if docs_markdown != [admission_path]:
        raise ValueError(f"docs 中只允许统一交付文档：{docs_markdown}")
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
        "schema_version": 2,
        "status": "passed",
        "yaml_files": validate_yaml(),
        "dataset": validate_dataset(dataset),
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
