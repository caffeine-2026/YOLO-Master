#!/usr/bin/env python3
"""Validate a converted DeepPCB YOLO dataset and load one real batch before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
CLASSES = ("open", "short", "mousebite", "spur", "copper", "pin-hole")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Repository-relative DeepPCB dataset YAML.")
    parser.add_argument("--manifest", type=Path, required=True, help="Repository-relative conversion manifest.")
    parser.add_argument("--output", type=Path, required=True, help="Repository-relative JSON validation output.")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=1)
    return parser.parse_args()


def inside_repo(path: Path) -> Path:
    resolved = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside repository: {path}") from exc
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_files(root: Path) -> tuple[dict[str, object], dict[str, set[str]]]:
    summary = {}
    hashes = {}
    for split in ("train", "val", "test"):
        images = sorted((root / "images" / split).glob("*.jpg"))
        labels = sorted((root / "labels" / split).glob("*.txt"))
        if not images or len(images) != len(labels):
            raise ValueError(f"{split} image/label mismatch: {len(images)} != {len(labels)}")
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise ValueError(f"{split} image/label stems do not match")
        object_counts: Counter[int] = Counter()
        image_class_counts: Counter[int] = Counter()
        split_hashes = set()
        for image_path, label_path in zip(images, labels):
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image size: {image_path}")
            image_hash = sha256(image_path)
            if image_hash in split_hashes:
                raise ValueError(f"Duplicate image content inside {split}: {image_path}")
            split_hashes.add(image_hash)
            present = set()
            rows = label_path.read_text(encoding="utf-8").splitlines()
            if not rows:
                raise ValueError(f"Empty label: {label_path}")
            for line_number, row in enumerate(rows, 1):
                fields = row.split()
                if len(fields) != 5:
                    raise ValueError(f"Invalid YOLO row: {label_path}:{line_number}")
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
                if class_id not in range(len(CLASSES)) or not all(math.isfinite(value) for value in coordinates):
                    raise ValueError(f"Invalid class or non-finite coordinate: {label_path}:{line_number}")
                x, y, box_width, box_height = coordinates
                if box_width <= 0 or box_height <= 0:
                    raise ValueError(f"Non-positive box: {label_path}:{line_number}")
                if not (0 <= x <= 1 and 0 <= y <= 1 and x - box_width / 2 >= -1e-6 and x + box_width / 2 <= 1 + 1e-6):
                    raise ValueError(f"Horizontal box outside image: {label_path}:{line_number}")
                if not (y - box_height / 2 >= -1e-6 and y + box_height / 2 <= 1 + 1e-6):
                    raise ValueError(f"Vertical box outside image: {label_path}:{line_number}")
                object_counts[class_id] += 1
                present.add(class_id)
            image_class_counts.update(present)
        summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": sum(object_counts.values()),
            "objects_by_class": {CLASSES[index]: object_counts[index] for index in range(len(CLASSES))},
            "positive_images_by_class": {CLASSES[index]: image_class_counts[index] for index in range(len(CLASSES))},
        }
        hashes[split] = split_hashes
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = hashes[left] & hashes[right]
        if overlap:
            raise ValueError(f"Duplicate image content across {left}/{right}: {len(overlap)}")
    return summary, hashes


def load_batch(data_yaml: Path, imgsz: int, batch_size: int) -> dict[str, object]:
    data = check_det_dataset(str(data_yaml), autodownload=False)
    if tuple(data["names"].values()) != CLASSES:
        raise ValueError(f"Unexpected DeepPCB class names: {data['names']}")
    cfg = get_cfg(
        overrides={
            "task": "detect",
            "imgsz": imgsz,
            "rect": False,
            "cache": False,
            "single_cls": False,
            "workers": 0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
        }
    )
    dataset = build_yolo_dataset(cfg, data["train"], batch_size, data, mode="val", stride=32)
    loader = build_dataloader(dataset, batch_size, workers=0, shuffle=False)
    batch = next(iter(loader))
    images = batch["img"]
    classes = sorted({int(value) for value in batch["cls"].view(-1).tolist()})
    return {
        "loaded": True,
        "image_shape": list(images.shape),
        "image_dtype": str(images.dtype),
        "class_ids": classes,
        "box_count": int(batch["bboxes"].shape[0]),
    }


def main() -> int:
    args = parse_args()
    data_yaml = inside_repo(args.data)
    manifest_path = inside_repo(args.manifest)
    output = inside_repo(args.output)
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if tuple(config.get("names", {}).values()) != CLASSES:
        raise ValueError(f"Dataset YAML classes must be {CLASSES}")
    dataset_root = inside_repo(Path(config["path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits, _ = validate_files(dataset_root)
    if manifest.get("split_image_counts") != {name: value["images"] for name, value in splits.items()}:
        raise ValueError("Conversion manifest split counts do not match the dataset")
    batch = load_batch(data_yaml, args.imgsz, args.batch)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "data": args.data.as_posix(),
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "classes": list(CLASSES),
        "splits": splits,
        "split_overlap_count": 0,
        "annotation_checks": {
            "nonempty": True,
            "five_columns": True,
            "class_ids_in_range": True,
            "finite_coordinates": True,
            "positive_box_size": True,
            "boxes_inside_images": True,
        },
        "dataloader_smoke": batch,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
