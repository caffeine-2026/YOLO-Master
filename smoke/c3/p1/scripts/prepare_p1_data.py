#!/usr/bin/env python3
"""Create and validate the immutable seed-824 C3 P1 small-sample splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
SEED = 824
SAMPLE_SIZE = 100
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

DATASETS = {
    "neu_det": {
        "display_name": "NEU-DET",
        "root": REPO_ROOT / "datasets" / "neu_det_yolo",
        "names": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
        "strategy": "source_class_stratified",
        "allocation": [17, 17, 17, 17, 16, 16],
    },
    "deeppcb": {
        "display_name": "DeepPCB",
        "root": REPO_ROOT / "datasets" / "deeppcb_yolo",
        "names": ["open", "short", "mousebite", "spur", "copper", "pin-hole"],
        "strategy": "uniform_without_replacement",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace only P1 split/evidence files, never source data.")
    return parser.parse_args()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def label_for(image: Path, root: Path) -> Path:
    return root / "labels" / image.parent.name / f"{image.stem}.txt"


def parse_label(path: Path, class_count: int) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing label: {relative(path)}")
    classes: list[int] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"{relative(path)}:{line_number}: expected 5 YOLO fields")
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
        x, y, width, height = values
        if not 0 <= class_id < class_count:
            raise ValueError(f"{relative(path)}:{line_number}: invalid class {class_id}")
        if width <= 0 or height <= 0:
            raise ValueError(f"{relative(path)}:{line_number}: non-positive box")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{relative(path)}:{line_number}: normalized box is outside [0, 1]")
        if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6:
            raise ValueError(f"{relative(path)}:{line_number}: x bounds exceed image")
        if y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
            raise ValueError(f"{relative(path)}:{line_number}: y bounds exceed image")
        classes.append(class_id)
    return classes


def select_images(key: str, settings: dict[str, object], candidates: list[Path]) -> list[Path]:
    rng = random.Random(SEED)
    if settings["strategy"] == "uniform_without_replacement":
        return sorted(rng.sample(candidates, SAMPLE_SIZE))

    selected: list[Path] = []
    names = settings["names"]
    allocation = settings["allocation"]
    assert isinstance(names, list) and isinstance(allocation, list)
    for class_name, count in zip(names, allocation):
        pool = [path for path in candidates if path.stem.startswith(f"{class_name}_")]
        if len(pool) < count:
            raise ValueError(f"{key}: class {class_name} has only {len(pool)} source images")
        selected.extend(rng.sample(pool, count))
    if len(selected) != SAMPLE_SIZE or len(set(selected)) != SAMPLE_SIZE:
        raise AssertionError(f"{key}: selection is not exactly {SAMPLE_SIZE} unique images")
    return sorted(selected)


def distribution(images: list[Path], root: Path, class_count: int) -> tuple[Counter[int], Counter[int], int]:
    positive_images: Counter[int] = Counter()
    objects: Counter[int] = Counter()
    empty_labels = 0
    for image in images:
        classes = parse_label(label_for(image, root), class_count)
        if not classes:
            empty_labels += 1
        objects.update(classes)
        positive_images.update(set(classes))
    return positive_images, objects, empty_labels


def split_set(root: Path, split: str) -> set[Path]:
    return {path.resolve() for path in image_files(root / "images" / split)}


def write_once(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing P1 evidence: {relative(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_manifest(key: str, settings: dict[str, object], overwrite: bool) -> dict[str, object]:
    root = settings["root"]
    names = settings["names"]
    assert isinstance(root, Path) and isinstance(names, list)
    train_source = image_files(root / "images" / "train")
    val = image_files(root / "images" / "val")
    test = image_files(root / "images" / "test")
    selected = select_images(key, settings, train_source)

    selected_set = {path.resolve() for path in selected}
    overlaps = {
        "train_val": len(selected_set & split_set(root, "val")),
        "train_test": len(selected_set & split_set(root, "test")),
        "val_test": len(split_set(root, "val") & split_set(root, "test")),
    }
    if any(overlaps.values()):
        raise ValueError(f"{key}: split overlap detected: {overlaps}")

    positive_images, objects, empty_labels = distribution(selected, root, len(names))
    for split_images in (val, test):
        distribution(split_images, root, len(names))

    positive_values = [positive_images[index] for index in range(len(names))]
    imbalance_ratio = max(positive_values) / min(positive_values) if min(positive_values) else float("inf")
    distribution_status = "PASS" if imbalance_ratio <= 1.5 else "FAIL"

    list_path = P1_ROOT / "config" / key / f"train_seed{SEED}.txt"
    list_text = "\n".join(relative(path) for path in selected) + "\n"
    write_once(list_path, list_text, overwrite)
    membership_sha256 = hashlib.sha256(list_text.encode()).hexdigest()

    manifest = {
        "schema_version": 1,
        "dataset": settings["display_name"],
        "seed": SEED,
        "selection_strategy": settings["strategy"],
        "requested_train_images": SAMPLE_SIZE,
        "source": {
            "root": relative(root),
            "train_images": len(train_source),
            "val_images": len(val),
            "test_images": len(test),
        },
        "selected_train_images": len(selected),
        "train_list": relative(list_path),
        "train_list_sha256": membership_sha256,
        "class_names": names,
        "class_distribution": {
            "positive_images": {str(index): positive_images[index] for index in range(len(names))},
            "objects": {str(index): objects[index] for index in range(len(names))},
            "positive_image_max_min_ratio": round(imbalance_ratio, 6),
            "status": distribution_status,
            "threshold": "max/min <= 1.5",
        },
        "empty_train_labels": empty_labels,
        "split_overlap": overlaps,
        "selected_images": [relative(path) for path in selected],
        "validation": {
            "exact_sample_count": len(selected) == SAMPLE_SIZE,
            "unique_membership": len(selected_set) == SAMPLE_SIZE,
            "image_label_pairs": True,
            "yolo_boxes_valid": True,
            "no_split_overlap": not any(overlaps.values()),
            "class_distribution_acceptable": distribution_status == "PASS",
        },
    }
    manifest["status"] = "PASS" if all(manifest["validation"].values()) else "FAIL"
    manifest_path = P1_ROOT / "evidence" / f"{key}_split_manifest.json"
    write_once(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", overwrite)
    return manifest


def build_model_boundary(overwrite: bool) -> dict[str, object]:
    model_yaml = REPO_ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11.yaml"
    checkpoint = REPO_ROOT / "yolo11n.pt"
    model = yaml.safe_load(model_yaml.read_text(encoding="utf-8"))
    backbone = model.get("backbone", [])
    head = model.get("head", [])
    if len(backbone) != 11 or len(head) != 13:
        raise ValueError("Unexpected YOLO11 model boundary; freeze=11 must be reviewed")
    entries = []
    for index, entry in enumerate(backbone + head):
        entries.append({"index": index, "section": "backbone" if index < len(backbone) else "head", "module": str(entry[2])})
    payload = {
        "schema_version": 1,
        "model_yaml": relative(model_yaml),
        "model_yaml_sha256": sha256(model_yaml),
        "pretrained_checkpoint": relative(checkpoint),
        "pretrained_sha256": sha256(checkpoint),
        "backbone_top_level_count": len(backbone),
        "freeze_argument": 11,
        "frozen_modules": [f"model.{index}" for index in range(11)],
        "trainable_top_level_modules": [f"model.{index}" for index in range(11, 24)],
        "top_level_modules": entries,
        "status": "PASS",
    }
    path = P1_ROOT / "evidence" / "model_boundary.json"
    write_once(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n", overwrite)
    return payload


def main() -> int:
    args = parse_args()
    manifests = {key: build_manifest(key, settings, args.overwrite) for key, settings in DATASETS.items()}
    boundary = build_model_boundary(args.overwrite)
    status = "PASS" if all(item["status"] == "PASS" for item in manifests.values()) else "FAIL"
    plan = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "sample_size_per_dataset": SAMPLE_SIZE,
        "datasets": manifests,
        "model_boundary": boundary,
        "status": status,
        "policy": "All three methods reuse the exact same per-dataset membership; results cannot trigger resampling.",
    }
    plan_path = P1_ROOT / "evidence" / "p1_data_plan.json"
    write_once(plan_path, json.dumps(plan, ensure_ascii=False, indent=2) + "\n", args.overwrite)

    lines = [
        "# C3 P1 小样本数据计划",
        "",
        "P1 pilot 对两个数据集统一使用 100 张训练图、固定 seed=824；同一数据集的三种方法复用完全相同的成员列表。",
        "",
        "| 数据集 | 来源训练图 | P1 训练图 | val | test | 抽样 | 正样本图分布 | 状态 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for key in ("neu_det", "deeppcb"):
        item = manifests[key]
        dist = item["class_distribution"]["positive_images"]
        lines.append(
            f"| {item['dataset']} | {item['source']['train_images']} | {item['selected_train_images']} | "
            f"{item['source']['val_images']} | {item['source']['test_images']} | {item['selection_strategy']} | "
            f"{' / '.join(str(dist[str(i)]) for i in range(6))} | {item['status']} |"
        )
    lines.extend(
        [
            "",
            "类别分布验收采用各类别正样本图数量 max/min ≤ 1.5。NEU-DET 按来源类别分层；DeepPCB 为多标签数据，使用确定性均匀抽样后审计分布。未依据任何方法结果重划 split。",
            "",
            f"数据计划状态：`{status}`。",
            "",
        ]
    )
    write_once(P1_ROOT / "evidence" / "P1_DATA_PLAN.md", "\n".join(lines), args.overwrite)
    print(json.dumps({"status": status, "sample_size": SAMPLE_SIZE}, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
