#!/usr/bin/env python3
"""Convert a locally obtained MVTec AD tree into a smoke-only YOLO defect-detection dataset.

This conversion is not the official unsupervised MVTec AD protocol. It derives bounding boxes from pixel masks and
re-splits anomalous test images so a supervised detector has positive samples during training.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Extracted MVTec AD root containing category dirs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=824)
    parser.add_argument(
        "--train-ratio", type=float, default=0.6, help="Positive-image train share within each defect group."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Positive-image val share within each defect group."
    )
    parser.add_argument(
        "--train-shots-per-group",
        type=int,
        default=0,
        help="0 keeps the ratio-based positive train split.",
    )
    parser.add_argument("--min-component-area", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def images_under(path: Path) -> list[Path]:
    return sorted(item for item in path.glob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def load_opencv() -> ModuleType:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required; install repository requirements before converting MVTec AD") from exc
    return cv2


def split_group(
    items: list[Path], rng: random.Random, train_ratio: float, val_ratio: float, shots: int
) -> dict[str, list[Path]]:
    items = sorted(items)
    rng.shuffle(items)
    if shots:
        train_end = min(shots, len(items))
    else:
        train_end = max(1, int(len(items) * train_ratio)) if items else 0
    remaining = len(items) - train_end
    desired_val = int(len(items) * val_ratio)
    val_size = min(desired_val, max(0, remaining - 1))
    return {
        "train": items[:train_end],
        "val": items[train_end : train_end + val_size],
        "test": items[train_end + val_size :],
    }


def boxes_from_mask(mask_path: Path, min_area: int, cv2: ModuleType) -> list[tuple[int, int, int, int]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    binary = (mask > 0).astype("uint8")
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area >= min_area:
            boxes.append((x, y, width, height))
    return boxes


def mask_for(image: Path, category: Path, defect_type: str) -> Path:
    return category / "ground_truth" / defect_type / f"{image.stem}_mask.png"


def write_example(
    image: Path,
    mask: Path | None,
    output: Path,
    split: str,
    category: str,
    defect: str,
    min_area: int,
    cv2: ModuleType,
) -> int:
    source_split = image.parents[1].name
    target_name = f"{category}__{source_split}__{defect}__{image.name}"
    target_image = output / "images" / split / target_name
    target_label = output / "labels" / split / f"{Path(target_name).stem}.txt"
    shutil.copy2(image, target_image)
    if mask is None:
        target_label.write_text("", encoding="utf-8")
        return 0

    pixels = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise ValueError(f"Cannot read image: {image}")
    height, width = pixels.shape[:2]
    rows = []
    for x, y, box_width, box_height in boxes_from_mask(mask, min_area, cv2):
        x_center = (x + box_width / 2.0) / width
        y_center = (y + box_height / 2.0) / height
        rows.append(f"0 {x_center:.6f} {y_center:.6f} {box_width / width:.6f} {box_height / height:.6f}")
    if not rows:
        raise ValueError(f"No components with area >= {min_area} in {mask}")
    target_label.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def main() -> int:
    args = parse_args()
    if not 0 < args.train_ratio < 1 or not 0 <= args.val_ratio < 1 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Require 0 < train_ratio < 1 and train_ratio + val_ratio < 1")
    source = args.source.resolve()
    categories = sorted(path for path in source.iterdir() if path.is_dir() and (path / "train/good").is_dir())
    if not categories:
        raise FileNotFoundError(f"No MVTec AD categories with train/good found under {source}")
    cv2 = load_opencv()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    positive_groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    negative_groups: dict[str, list[Path]] = defaultdict(list)
    for category in categories:
        negative_groups[category.name].extend(images_under(category / "train/good"))
        negative_groups[category.name].extend(images_under(category / "test/good"))
        for defect_dir in sorted((category / "test").iterdir()):
            if defect_dir.is_dir() and defect_dir.name != "good":
                positive_groups[(category.name, defect_dir.name)].extend(images_under(defect_dir))

    split_items: dict[str, list[tuple[Path, Path | None, str, str]]] = {name: [] for name in ("train", "val", "test")}
    for group_index, ((category_name, defect_type), items) in enumerate(sorted(positive_groups.items())):
        category = source / category_name
        parts = split_group(
            items,
            random.Random(args.seed + group_index),
            args.train_ratio,
            args.val_ratio,
            args.train_shots_per_group,
        )
        for split, paths in parts.items():
            for image in paths:
                mask = mask_for(image, category, defect_type)
                if not mask.is_file():
                    raise FileNotFoundError(mask)
                split_items[split].append((image, mask, category_name, defect_type))

    for group_index, (category_name, items) in enumerate(sorted(negative_groups.items())):
        parts = split_group(items, random.Random(args.seed + 10_000 + group_index), 0.6, 0.2, 0)
        for split, paths in parts.items():
            split_items[split].extend((image, None, category_name, "good") for image in paths)

    manifest_counts: dict[str, Counter[str]] = {}
    for split, items in split_items.items():
        counts: Counter[str] = Counter(images=len(items), boxes=0, positive_images=0, negative_images=0)
        for image, mask, category_name, defect_type in sorted(items, key=lambda item: (item[2], item[3], item[0].name)):
            box_count = write_example(
                image, mask, output, split, category_name, defect_type, args.min_component_area, cv2
            )
            counts["boxes"] += box_count
            counts["positive_images" if mask else "negative_images"] += 1
        manifest_counts[split] = counts

    manifest = {
        "dataset": "MVTec AD converted to supervised YOLO detection",
        "warning": "Smoke-only conversion; not valid for official unsupervised MVTec AD benchmarking.",
        "source": str(source),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "train_shots_per_group": args.train_shots_per_group,
        "min_component_area": args.min_component_area,
        "categories": [path.name for path in categories],
        "names": {"0": "defect"},
        "counts": {split: dict(counts) for split, counts in manifest_counts.items()},
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
