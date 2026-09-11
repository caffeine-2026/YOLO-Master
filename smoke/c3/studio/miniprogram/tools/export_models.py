"""Merge canonical C3 V-PEFT checkpoints and export static ONNX artifacts for WeChat."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
STUDIO_ROOT = REPO_ROOT / "smoke" / "c3" / "studio"
DEFAULT_OUTPUT = STUDIO_ROOT / "miniprogram" / "dist" / "models"
sys.path.insert(0, str(STUDIO_ROOT))

from utils.load_results import canonical_checkpoint

DATASET_SLUGS = {"NEU-DET": "neu", "DeepPCB": "deeppcb"}
PARITY_SAMPLES = {
    "NEU-DET": REPO_ROOT / "datasets" / "raw" / "neu_det_source" / "NEU-DET" / "IMAGES" / "scratches_205.jpg",
    "DeepPCB": REPO_ROOT / "datasets" / "deeppcb_fewshot_yolo" / "images" / "val" / "13000036_test.jpg",
}
IOS_NPU_UNVERIFIED_OPS = frozenset({"Shape", "Transpose"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()


def detection_rows(result) -> list[tuple[int, float, np.ndarray]]:
    if result.boxes is None:
        return []
    return [
        (int(class_id), float(score), np.asarray(box, dtype=np.float64))
        for class_id, score, box in zip(
            result.boxes.cls.detach().cpu().tolist(),
            result.boxes.conf.detach().cpu().tolist(),
            result.boxes.xyxy.detach().cpu().tolist(),
        )
    ]


def validate_prediction_parity(checkpoint: Path, exported: Path, dataset: str) -> dict[str, object]:
    from ultralytics import YOLO

    sample = PARITY_SAMPLES[dataset]
    if not sample.is_file():
        raise FileNotFoundError(f"Parity sample is missing: {sample.relative_to(REPO_ROOT)}")
    source = YOLO(checkpoint).predict(sample, device="cpu", imgsz=640, conf=0.25, verbose=False)[0]
    target = YOLO(exported, task="detect").predict(sample, device="cpu", imgsz=640, conf=0.25, verbose=False)[0]
    source_rows = detection_rows(source)
    target_rows = detection_rows(target)
    if len(source_rows) != len(target_rows):
        raise RuntimeError(f"Parity detection count differs: PyTorch={len(source_rows)}, ONNX={len(target_rows)}")
    maximum_score_delta = 0.0
    maximum_box_delta = 0.0
    for source_row, target_row in zip(source_rows, target_rows):
        if source_row[0] != target_row[0]:
            raise RuntimeError(f"Parity class differs: PyTorch={source_row[0]}, ONNX={target_row[0]}")
        maximum_score_delta = max(maximum_score_delta, abs(source_row[1] - target_row[1]))
        maximum_box_delta = max(maximum_box_delta, float(np.max(np.abs(source_row[2] - target_row[2]))))
    if maximum_score_delta >= 1e-3 or maximum_box_delta >= 0.5:
        raise RuntimeError(
            f"Parity tolerance exceeded: score={maximum_score_delta:.6f}, bbox={maximum_box_delta:.4f}px"
        )
    return {
        "sample": str(sample.relative_to(REPO_ROOT)),
        "detections": len(source_rows),
        "max_score_delta": maximum_score_delta,
        "max_bbox_delta_px": maximum_box_delta,
        "status": "PASS",
    }


def export_one(dataset: str, output_dir: Path, force: bool) -> dict[str, object]:
    import onnx

    from ultralytics import YOLO

    checkpoint, _ = canonical_checkpoint(dataset, "V-PEFT")
    source_hash = sha256(checkpoint)
    filename = f"{DATASET_SLUGS[dataset]}_vpeft_640.onnx"
    destination = output_dir / filename
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {destination}; pass --force after reviewing it")

    with tempfile.TemporaryDirectory(prefix=f"c3-{DATASET_SLUGS[dataset]}-export-") as directory:
        temporary_checkpoint = Path(directory) / "merged_source.pt"
        shutil.copy2(checkpoint, temporary_checkpoint)
        model = YOLO(temporary_checkpoint)
        names = dict(model.model.names)
        if not model.merge_lora():
            raise RuntimeError(f"Failed to merge the active V-PEFT adapter for {dataset}")
        exported_path = Path(
            model.export(
                format="onnx",
                imgsz=640,
                batch=1,
                dynamic=False,
                simplify=False,
                opset=12,
                nms=False,
                device="cpu",
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported_path, destination)

    if sha256(checkpoint) != source_hash:
        raise RuntimeError(f"Source checkpoint changed during export: {checkpoint}")
    graph = onnx.load(destination)
    onnx.checker.check_model(graph)
    inputs = [
        {"name": value.name, "shape": [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]}
        for value in graph.graph.input
    ]
    outputs = [
        {"name": value.name, "shape": [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]}
        for value in graph.graph.output
    ]
    operators = sorted({node.op_type for node in graph.graph.node})
    parity = validate_prediction_parity(checkpoint, destination, dataset)
    return {
        "id": f"{DATASET_SLUGS[dataset]}-vpeft-640",
        "dataset": dataset,
        "method": "V-PEFT",
        "title": f"{dataset} · V-PEFT",
        "file": filename,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "source_checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "source_checkpoint_sha256": source_hash,
        "input_size": 640,
        "input_name": inputs[0]["name"],
        "output_name": outputs[0]["name"],
        "inputs": inputs,
        "outputs": outputs,
        "opset": [{"domain": entry.domain, "version": entry.version} for entry in graph.opset_import],
        "operators": operators,
        "ios_npu_unverified_operators": sorted(IOS_NPU_UNVERIFIED_OPS.intersection(operators)),
        "labels": [names[index] for index in sorted(names)],
        "parity": parity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *DATASET_SLUGS), default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output.resolve()
    datasets = list(DATASET_SLUGS) if args.dataset == "all" else [args.dataset]
    records = [export_one(dataset, output_dir, args.force) for dataset in datasets]
    manifest = {
        "schema": "c3-edge-model-manifest/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "export": {"format": "onnx", "opset": 12, "dynamic": False, "nms": False, "imgsz": 640},
        "models": records,
    }
    manifest_path = output_dir.parent / "model-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "models": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
