#!/usr/bin/env python3
"""Convert the official NEU-DET Pascal VOC archive to a deterministic YOLO detection dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing IMAGES/ and ANNOTATIONS/.")
    parser.add_argument("--output", type=Path, required=True, help="YOLO dataset output directory.")
    parser.add_argument("--seed", type=int, default=824)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--train-shots-per-class",
        type=int,
        default=0,
        help="Keep only this many training images per filename class; 0 keeps the full training split.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def source_sha256(path: Path) -> str | None:
    archive = path.parents[1] / "NEU-DET.zip"
    if not archive.is_file():
        return None
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_group(stem: str) -> str:
    name, separator, suffix = stem.rpartition("_")
    if not separator or not suffix.isdigit() or name not in CLASS_TO_ID:
        raise ValueError(f"Unexpected NEU-DET image name: {stem}")
    return name


def stratified_split(
    images: list[Path], seed: int, train_ratio: float, val_ratio: float, train_shots_per_class: int
) -> dict[str, list[Path]]:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Require 0 < train_ratio < 1 and train_ratio + val_ratio < 1")
    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        groups[image_group(image.stem)].append(image)
    if set(groups) != set(CLASSES):
        raise ValueError(f"Expected classes {CLASSES}, found {sorted(groups)}")

    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for class_index, class_name in enumerate(CLASSES):
        class_images = sorted(groups[class_name])
        random.Random(seed + class_index).shuffle(class_images)
        train_end = int(len(class_images) * train_ratio)
        val_end = train_end + int(len(class_images) * val_ratio)
        train_images = class_images[:train_end]
        if train_shots_per_class:
            if train_shots_per_class > len(train_images):
                raise ValueError(
                    f"Requested {train_shots_per_class} {class_name} shots, but only {len(train_images)} are available"
                )
            train_images = train_images[:train_shots_per_class]
        splits["train"].extend(train_images)
        splits["val"].extend(class_images[train_end:val_end])
        splits["test"].extend(class_images[val_end:])
    for split_paths in splits.values():
        split_paths.sort(key=lambda path: path.name)
    return splits


def voc_to_yolo(xml_path: Path) -> tuple[list[str], Counter[str]]:
    root = ET.parse(xml_path).getroot()
    width = float(root.findtext("size/width", default="0"))
    height = float(root.findtext("size/height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    rows: list[str] = []
    object_counts: Counter[str] = Counter()
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        if name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class {name!r} in {xml_path}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox in {xml_path}")
        xmin = max(0.0, min(width, float(box.findtext("xmin", default="0"))))
        ymin = max(0.0, min(height, float(box.findtext("ymin", default="0"))))
        xmax = max(0.0, min(width, float(box.findtext("xmax", default="0"))))
        ymax = max(0.0, min(height, float(box.findtext("ymax", default="0"))))
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid box ({xmin}, {ymin}, {xmax}, {ymax}) in {xml_path}")
        x_center = (xmin + xmax) / (2.0 * width)
        y_center = (ymin + ymax) / (2.0 * height)
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        rows.append(f"{CLASS_TO_ID[name]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        object_counts[name] += 1
    if not rows:
        raise ValueError(f"No objects found in {xml_path}")
    return rows, object_counts


def write_dataset(args: argparse.Namespace) -> dict:
    source = args.source.resolve()
    image_dir = source / "IMAGES"
    annotation_dir = source / "ANNOTATIONS"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(f"Expected {image_dir} and {annotation_dir}")

    images = sorted(image_dir.glob("*.jpg"))
    xml_files = sorted(annotation_dir.glob("*.xml"))
    if len(images) != 1800 or len(xml_files) != 1800:
        raise ValueError(f"Expected 1800 images/XML files, found {len(images)} images and {len(xml_files)} XML files")

    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    splits = stratified_split(images, args.seed, args.train_ratio, args.val_ratio, args.train_shots_per_class)
    split_objects: dict[str, Counter[str]] = {}
    split_images: dict[str, Counter[str]] = {}
    for split, split_paths in splits.items():
        object_counts: Counter[str] = Counter()
        image_counts: Counter[str] = Counter()
        for image_path in split_paths:
            xml_path = annotation_dir / f"{image_path.stem}.xml"
            if not xml_path.is_file():
                raise FileNotFoundError(xml_path)
            rows, counts = voc_to_yolo(xml_path)
            shutil.copy2(image_path, output / "images" / split / image_path.name)
            (output / "labels" / split / f"{image_path.stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            object_counts.update(counts)
            image_counts[image_group(image_path.stem)] += 1
        split_objects[split] = object_counts
        split_images[split] = image_counts

    manifest = {
        "dataset": "NEU-DET",
        "source": str(source),
        "source_archive_sha256": source_sha256(source),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "train_shots_per_class": args.train_shots_per_class,
        "classes": list(CLASSES),
        "image_counts": {split: dict(split_images[split]) for split in splits},
        "object_counts": {split: dict(split_objects[split]) for split in splits},
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    manifest = write_dataset(args)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
