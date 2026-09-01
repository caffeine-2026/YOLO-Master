#!/usr/bin/env python3
"""Audit train/validation industrial-image characteristics without reading locked test data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "smoke" / "c3" / "augmentation"
SPECS = {
    "neu": {
        "train_list": REPO_ROOT / "smoke/c3/p2/config/splits/neu_100_seed824.txt",
        "val_dir": REPO_ROOT / "datasets/neu_det_yolo/images/val",
        "names": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
    },
    "deeppcb": {
        "train_list": REPO_ROOT / "smoke/c3/p2/config/splits/deeppcb_100_seed824.txt",
        "val_dir": REPO_ROOT / "datasets/deeppcb_yolo/images/val",
        "names": ["open", "short", "mousebite", "spur", "copper", "pin-hole"],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_path(image: Path) -> Path:
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def summarize(spec: dict) -> dict:
    images = [REPO_ROOT / line for line in spec["train_list"].read_text().splitlines() if line.strip()]
    shapes: Counter[str] = Counter()
    channel_deltas = []
    areas = []
    widths = []
    heights = []
    class_areas: dict[int, list[float]] = defaultdict(list)
    boxes_per_image = []
    samples = []
    for image_path in images:
        image = np.asarray(Image.open(image_path))
        shapes[str(tuple(image.shape))] += 1
        if image.ndim == 3:
            channel_deltas.append(float(np.abs(image.astype(np.float32) - image[..., :1]).max()))
        rows = [line.split() for line in label_path(image_path).read_text().splitlines() if line.strip()]
        boxes_per_image.append(len(rows))
        for row in rows:
            cls, width, height = int(row[0]), float(row[3]), float(row[4])
            area = width * height
            areas.append(area)
            widths.append(width)
            heights.append(height)
            class_areas[cls].append(area)
        if len(samples) < 8:
            samples.append({"path": image_path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(image_path)})
    quantiles = np.quantile(areas, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist()
    return {
        "scope": "fixed 100-shot train membership plus validation directory metadata; locked test not read",
        "train_images": len(images),
        "validation_images": len(list(spec["val_dir"].glob("*"))),
        "image_shapes": dict(shapes),
        "maximum_channel_difference": max(channel_deltas, default=0.0),
        "grayscale_pixels_confirmed": max(channel_deltas, default=0.0) == 0.0,
        "boxes": len(areas),
        "boxes_per_image_mean": float(np.mean(boxes_per_image)),
        "empty_train_images": sum(count == 0 for count in boxes_per_image),
        "normalized_box_area_quantiles": dict(zip(("min", "p10", "p25", "p50", "p75", "p90", "max"), quantiles)),
        "normalized_box_width_median": float(np.median(widths)),
        "normalized_box_height_median": float(np.median(heights)),
        "per_class": {
            spec["names"][index]: {
                "boxes": len(class_areas[index]),
                "normalized_area_median": float(np.median(class_areas[index])),
            }
            for index in range(len(spec["names"]))
        },
        "train_list": spec["train_list"].relative_to(REPO_ROOT).as_posix(),
        "train_list_sha256": sha256(spec["train_list"]),
        "sample_images": samples,
    }


def main() -> int:
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "locked_test_read": False,
        "datasets": {name: summarize(spec) for name, spec in SPECS.items()},
        "policy_implications": {
            "neu": "200x200 grayscale texture; allow bounded orientation and photometric variation.",
            "deeppcb": "640x640 near-binary imagery with very small defects; restrict geometry and blur/noise.",
            "both": "Disable hue, saturation, mosaic, mixup, cutmix, and copy-paste in all tested policies.",
        },
    }
    output = ROOT / "evidence" / "dataset_characteristics.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": output.relative_to(REPO_ROOT).as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
