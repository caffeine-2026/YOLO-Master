#!/usr/bin/env python3
"""Convert the original DeepPCB repository into deterministic YOLO detection datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
CLASSES = ("open", "short", "mousebite", "spur", "copper", "pin-hole")
SOURCE_REPOSITORY = "https://github.com/tangsanli5201/DeepPCB"


@dataclass(frozen=True)
class Sample:
    """One official DeepPCB tested-image/annotation/template tuple."""

    sample_id: str
    tested: Path
    template: Path
    annotation: Path
    official_split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Original DeepPCB checkout or its PCBData directory."
    )
    parser.add_argument("--output", type=Path, required=True, help="Independent YOLO dataset output directory.")
    parser.add_argument("--seed", type=int, default=824)
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Validation share taken only from official trainval."
    )
    parser.add_argument(
        "--train-shots-per-class",
        type=int,
        default=0,
        help="Minimum positive training images per class; 0 keeps the full deterministic train pool.",
    )
    parser.add_argument("--manifest-output", type=Path, help="Optional second copy of the generated manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing derived output only.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path must stay inside the repository: {path}") from exc


def derived_dataset_path(path: Path) -> Path:
    """Limit replaceable outputs to non-raw children of the repository datasets directory."""
    resolved = path.resolve()
    datasets_root = (REPO_ROOT / "datasets").resolve()
    try:
        relative = resolved.relative_to(datasets_root)
    except ValueError as exc:
        raise ValueError(f"Derived output must stay under {datasets_root}: {path}") from exc
    if not relative.parts or relative.parts[0] == "raw":
        raise ValueError("Derived output must not be the datasets root or any raw-data directory")
    return resolved


def evidence_path(path: Path) -> Path:
    """Limit published manifest writes to the C3 evidence directory."""
    resolved = path.resolve()
    evidence_root = (REPO_ROOT / "smoke" / "c3" / "evidence").resolve()
    try:
        resolved.relative_to(evidence_root)
    except ValueError as exc:
        raise ValueError(f"Manifest output must stay under {evidence_root}: {path}") from exc
    return resolved


def source_commit(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    )
    return result.stdout.strip()


def resolve_source(source: Path) -> tuple[Path, Path]:
    source = source.resolve()
    if (source / "PCBData").is_dir():
        checkout, data_root = source, source / "PCBData"
    elif source.name == "PCBData" and (source.parent / ".git").is_dir():
        checkout, data_root = source.parent, source
    else:
        raise FileNotFoundError(f"Expected an original DeepPCB checkout or PCBData directory: {source}")
    for required in (checkout / "README.md", checkout / "LICENSE", data_root / "trainval.txt", data_root / "test.txt"):
        if not required.is_file():
            raise FileNotFoundError(required)
    return checkout, data_root


def within(root: Path, value: Path) -> Path:
    resolved = value.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"DeepPCB split path escapes PCBData: {value}") from exc
    return resolved


def read_official_split(data_root: Path, split: str) -> list[Sample]:
    split_file = data_root / f"{split}.txt"
    samples = []
    seen = set()
    for line_number, line in enumerate(split_file.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Invalid official split row: {split_file}:{line_number}")
        image_stub = within(data_root, data_root / fields[0])
        annotation = within(data_root, data_root / fields[1])
        tested = image_stub.with_name(f"{image_stub.stem}_test{image_stub.suffix}")
        template = image_stub.with_name(f"{image_stub.stem}_temp{image_stub.suffix}")
        if image_stub.stem in seen:
            raise ValueError(f"Duplicate sample ID in {split_file}: {image_stub.stem}")
        seen.add(image_stub.stem)
        for path in (tested, template, annotation):
            if not path.is_file():
                raise FileNotFoundError(path)
        samples.append(Sample(image_stub.stem, tested, template, annotation, split))
    return samples


def parse_annotation(sample: Sample) -> tuple[list[str], Counter[str], set[int]]:
    with Image.open(sample.tested) as image:
        width, height = image.size
        image.verify()
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {sample.tested}")

    rows = []
    counts: Counter[str] = Counter()
    class_ids = set()
    for line_number, line in enumerate(sample.annotation.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid annotation row: {sample.annotation}:{line_number}")
        x1, y1, x2, y2 = (float(value) for value in fields[:4])
        source_class = int(fields[4])
        if source_class not in range(1, len(CLASSES) + 1):
            raise ValueError(f"Class ID out of range: {sample.annotation}:{line_number}")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"Box outside {width}x{height} image: {sample.annotation}:{line_number} ({x1}, {y1}, {x2}, {y2})"
            )
        class_id = source_class - 1
        rows.append(
            f"{class_id} {(x1 + x2) / (2 * width):.6f} {(y1 + y2) / (2 * height):.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
        counts[CLASSES[class_id]] += 1
        class_ids.add(class_id)
    if not rows:
        raise ValueError(f"Empty DeepPCB annotation: {sample.annotation}")
    return rows, counts, class_ids


def deterministic_splits(
    trainval: list[Sample],
    official_test: list[Sample],
    seed: int,
    val_ratio: float,
    shots: int,
    classes: dict[str, set[int]],
) -> dict[str, list[Sample]]:
    if not 0 < val_ratio < 1:
        raise ValueError("Require 0 < val_ratio < 1")
    shuffled = sorted(trainval, key=lambda sample: sample.sample_id)
    random.Random(seed).shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    if val_size <= 0 or val_size >= len(shuffled):
        raise ValueError("Validation split would be empty or consume all official trainval samples")
    val = shuffled[:val_size]
    train_pool = shuffled[val_size:]

    if shots:
        if shots < 1:
            raise ValueError("--train-shots-per-class must be positive or zero")
        candidates = list(train_pool)
        random.Random(seed + 1).shuffle(candidates)
        selected = []
        coverage: Counter[int] = Counter()
        for sample in candidates:
            if any(coverage[class_id] < shots for class_id in classes[sample.sample_id]):
                selected.append(sample)
                coverage.update(classes[sample.sample_id])
            if all(coverage[class_id] >= shots for class_id in range(len(CLASSES))):
                break
        missing = {CLASSES[index]: max(0, shots - coverage[index]) for index in range(len(CLASSES))}
        missing = {name: count for name, count in missing.items() if count}
        if missing:
            raise ValueError(f"Official trainval cannot satisfy few-shot coverage: {missing}")
        train = selected
    else:
        train = train_pool

    splits = {"train": train, "val": val, "test": list(official_test)}
    split_ids = {name: {sample.sample_id for sample in samples} for name, samples in splits.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_ids[left] & split_ids[right]
        if overlap:
            raise ValueError(f"Split overlap {left}/{right}: {sorted(overlap)[:5]}")
    return {name: sorted(samples, key=lambda sample: sample.sample_id) for name, samples in splits.items()}


def write_dataset(args: argparse.Namespace) -> dict[str, object]:
    checkout, data_root = resolve_source(args.source)
    trainval = read_official_split(data_root, "trainval")
    official_test = read_official_split(data_root, "test")
    if len(trainval) != 1000 or len(official_test) != 500:
        raise ValueError(f"Expected official 1000/500 split, found {len(trainval)}/{len(official_test)}")
    all_samples = trainval + official_test
    if len({sample.sample_id for sample in all_samples}) != 1500:
        raise ValueError("Official trainval/test sample IDs overlap")

    parsed = {}
    source_objects: Counter[str] = Counter()
    class_presence = {}
    for sample in all_samples:
        rows, counts, class_ids = parse_annotation(sample)
        parsed[sample.sample_id] = rows
        source_objects.update(counts)
        class_presence[sample.sample_id] = class_ids
    if set(source_objects) != set(CLASSES):
        raise ValueError(f"Missing source classes: {sorted(set(CLASSES) - set(source_objects))}")

    all_tested = {path.stem[: -len("_test")] for path in data_root.rglob("*_test.jpg")}
    all_annotations = {path.stem for path in data_root.rglob("*.txt") if path.parent.name.endswith("_not")}
    official_ids = {sample.sample_id for sample in all_samples}
    if all_tested != official_ids or all_annotations != official_ids:
        raise ValueError("Official split does not exactly cover all tested images and annotations")
    all_templates = {path.stem[: -len("_temp")] for path in data_root.rglob("*_temp.jpg")}
    extra_templates = sorted(all_templates - official_ids)
    missing_templates = sorted(official_ids - all_templates)
    if missing_templates:
        raise ValueError(f"Missing templates: {missing_templates[:5]}")

    splits = deterministic_splits(
        trainval, official_test, args.seed, args.val_ratio, args.train_shots_per_class, class_presence
    )
    output = derived_dataset_path(args.output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace derived data")
        shutil.rmtree(output)
    for split in splits:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_objects = {}
    split_presence = {}
    for split, samples in splits.items():
        object_counts: Counter[str] = Counter()
        presence_counts: Counter[str] = Counter()
        for sample in samples:
            target_image = output / "images" / split / sample.tested.name
            target_label = output / "labels" / split / f"{sample.tested.stem}.txt"
            shutil.copy2(sample.tested, target_image)
            target_label.write_text("\n".join(parsed[sample.sample_id]) + "\n", encoding="utf-8")
            for row in parsed[sample.sample_id]:
                object_counts[CLASSES[int(row.split()[0])]] += 1
            for class_id in class_presence[sample.sample_id]:
                presence_counts[CLASSES[class_id]] += 1
        split_objects[split] = dict(object_counts)
        split_presence[split] = dict(presence_counts)

    membership = "\n".join(
        f"{split}:{sample.sample_id}" for split in ("train", "val", "test") for sample in splits[split]
    )
    key_files = (checkout / "README.md", checkout / "LICENSE", data_root / "trainval.txt", data_root / "test.txt")
    manifest = {
        "schema_version": 1,
        "dataset": "DeepPCB",
        "purpose": "YOLO detection smoke conversion using defective tested images only",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": source_commit(checkout),
            "checkout": repo_relative(checkout),
            "dataset_root": repo_relative(data_root),
            "research_use_note": "Original README states that the dataset can only be used for research purposes.",
            "key_file_sha256": {repo_relative(path): sha256(path) for path in key_files},
            "tested_images": len(all_tested),
            "template_images": len(all_templates),
            "annotations": len(all_annotations),
            "extra_unpaired_templates": extra_templates,
        },
        "official_split": {"trainval": len(trainval), "test": len(official_test)},
        "derived_split_policy": {
            "seed": args.seed,
            "val_ratio_within_official_trainval": args.val_ratio,
            "train_shots_per_class": args.train_shots_per_class,
            "test_preserved_from_official_split": True,
            "few_shot_definition": "minimum positive training images containing each class; images may be multi-label",
        },
        "classes": list(CLASSES),
        "source_object_counts": dict(source_objects),
        "split_image_counts": {split: len(samples) for split, samples in splits.items()},
        "split_object_counts": split_objects,
        "split_positive_image_counts": split_presence,
        "split_overlap_count": 0,
        "split_membership_sha256": hashlib.sha256((membership + "\n").encode()).hexdigest(),
    }
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (output / "split_manifest.json").write_text(rendered, encoding="utf-8")
    if args.manifest_output:
        manifest_output = evidence_path(args.manifest_output)
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(rendered, encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    manifest = write_dataset(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
