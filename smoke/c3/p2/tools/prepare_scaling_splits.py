#!/usr/bin/env python3
"""Build deterministic P1-anchored nested 10/50/100/500 P2 splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
P2_ROOT = REPO_ROOT / "smoke" / "c3" / "p2"
SPLIT_SEED = 824
SIZES = (10, 50, 100, 500)
DATASETS = {
    "neu": {
        "key": "neu_det",
        "label": "NEU-DET",
        "root": REPO_ROOT / "datasets" / "neu_det_yolo",
        "p1_config": P1_ROOT / "config" / "neu_det",
        "manifest": P2_ROOT / "evidence" / "neu_scaling_split_manifest.json",
    },
    "deeppcb": {
        "key": "deeppcb",
        "label": "DeepPCB",
        "root": REPO_ROOT / "datasets" / "deeppcb_yolo",
        "p1_config": P1_ROOT / "config" / "deeppcb",
        "manifest": P2_ROOT / "evidence" / "deeppcb_scaling_split_manifest.json",
    },
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    objects: tuple[int, ...]

    @property
    def classes(self) -> frozenset[int]:
        return frozenset(self.objects)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def membership_sha(paths: list[Path]) -> str:
    payload = "\n".join(sorted(path.relative_to(REPO_ROOT).as_posix() for path in paths)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def label_path(image: Path, root: Path) -> Path:
    return root / "labels" / "train" / f"{image.stem}.txt"


def load_record(image: Path, root: Path, class_count: int) -> ImageRecord:
    label = label_path(image, root)
    if not label.is_file():
        raise FileNotFoundError(label.relative_to(REPO_ROOT))
    objects = []
    for line in label.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Malformed label: {label.relative_to(REPO_ROOT)}")
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:5]]
        if not 0 <= class_id < class_count or not all(0 <= value <= 1 for value in coordinates):
            raise ValueError(f"Invalid YOLO label: {label.relative_to(REPO_ROOT)}")
        objects.append(class_id)
    return ImageRecord(image, tuple(objects))


def stable_tie(record: ImageRecord) -> int:
    value = f"{SPLIT_SEED}:{record.path.relative_to(REPO_ROOT).as_posix()}".encode()
    return int(hashlib.sha256(value).hexdigest()[:16], 16)


def balanced_order(
    candidates: list[ImageRecord], class_count: int, initial: list[ImageRecord] | None = None
) -> list[ImageRecord]:
    image_counts: Counter[int] = Counter()
    object_counts: Counter[int] = Counter()
    for record in initial or []:
        image_counts.update(record.classes)
        object_counts.update(record.objects)
    remaining = list(candidates)
    selected = []
    while remaining:

        def score(record: ImageRecord) -> tuple[float, float, float, int]:
            uncovered = sum(image_counts[class_id] == 0 for class_id in record.classes)
            image_balance = sum(1 / (1 + image_counts[class_id]) for class_id in record.classes)
            object_balance = sum(1 / (1 + object_counts[class_id]) for class_id in record.objects)
            return uncovered, image_balance, object_balance, -stable_tie(record)

        choice = max(remaining, key=score)
        remaining.remove(choice)
        selected.append(choice)
        image_counts.update(choice.classes)
        object_counts.update(choice.objects)
    return selected


def distribution(records: list[ImageRecord], names: dict[int, str]) -> dict[str, object]:
    image_counts: Counter[int] = Counter()
    object_counts: Counter[int] = Counter()
    for record in records:
        image_counts.update(record.classes)
        object_counts.update(record.objects)
    uncovered = [names[index] for index in names if image_counts[index] == 0]
    covered_class_count = len(names) - len(uncovered)
    return {
        "images": len(records),
        "objects": sum(object_counts.values()),
        "objects_per_image": sum(object_counts.values()) / len(records),
        "class_coverage": covered_class_count,
        "class_count": len(names),
        "all_classes_covered": covered_class_count == len(names),
        "uncovered_classes": uncovered,
        "images_per_class": {names[index]: image_counts[index] for index in names},
        "objects_per_class": {names[index]: object_counts[index] for index in names},
        "10-shot_class_coverage_limitation": uncovered if len(records) == 10 and uncovered else None,
    }


def write_list(path: Path, records: list[ImageRecord]) -> None:
    path.write_text(
        "".join(f"{record.path.relative_to(REPO_ROOT).as_posix()}\n" for record in records), encoding="utf-8"
    )


def split_membership(directory: Path) -> tuple[int, str]:
    paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return len(paths), membership_sha(paths)


def prepare_dataset(tag: str, config: dict[str, object]) -> dict[str, object]:
    root = Path(config["root"])
    p1_config = Path(config["p1_config"])
    p1_data = yaml.safe_load((p1_config / "dataset.yaml").read_text(encoding="utf-8"))
    names = {int(key): str(value) for key, value in p1_data["names"].items()}
    class_count = len(names)
    all_images = sorted(
        path for path in (root / "images" / "train").iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    record_by_relative = {
        image.relative_to(REPO_ROOT).as_posix(): load_record(image, root, class_count) for image in all_images
    }
    p1_list = p1_config / "train_seed824.txt"
    p1_relative = [line for line in p1_list.read_text(encoding="utf-8").splitlines() if line]
    p1_records = [record_by_relative[path] for path in p1_relative]
    if len(p1_records) != 100 or len(set(p1_records)) != 100:
        raise ValueError(f"Invalid P1 100-image anchor for {tag}")

    inner_order = balanced_order(p1_records, class_count)
    records_by_size = {10: inner_order[:10], 50: inner_order[:50], 100: p1_records}
    p1_set = set(p1_records)
    extras = balanced_order(
        [record for record in record_by_relative.values() if record not in p1_set], class_count, p1_records
    )
    records_by_size[500] = p1_records + extras[:400]

    split_dir = P2_ROOT / "config" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {}
    for size in SIZES:
        list_path = split_dir / f"{tag}_{size}_seed824.txt"
        data_path = split_dir / f"{tag}_{size}.yaml"
        if list_path.exists() or data_path.exists():
            raise FileExistsError(f"Refusing to overwrite P2 split output for {tag}/{size}")
        if size == 100:
            list_path.write_bytes(p1_list.read_bytes())
        else:
            write_list(list_path, records_by_size[size])
        data = {
            "path": ".",
            "train": list_path.relative_to(REPO_ROOT).as_posix(),
            "val": p1_data["val"],
            "test": p1_data["test"],
            "names": names,
        }
        data_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        split_rows[str(size)] = {
            "train_list": list_path.relative_to(REPO_ROOT).as_posix(),
            "train_list_sha256": sha256(list_path),
            "membership_sha256": membership_sha([record.path for record in records_by_size[size]]),
            "distribution": distribution(records_by_size[size], names),
            "selected_images": [record.path.relative_to(REPO_ROOT).as_posix() for record in records_by_size[size]],
        }

    sets = {size: set(records_by_size[size]) for size in SIZES}
    nested = {
        "10_subset_50": sets[10] < sets[50],
        "50_subset_100": sets[50] < sets[100],
        "100_subset_500": sets[100] < sets[500],
        "exact_counts": all(len(sets[size]) == size for size in SIZES),
    }
    val_count, val_sha = split_membership(root / "images" / "val")
    test_count, test_sha = split_membership(root / "images" / "test")
    reuse = {
        "p1_train_list": p1_list.relative_to(REPO_ROOT).as_posix(),
        "p1_train_list_sha256": sha256(p1_list),
        "p2_100_train_list_sha256": split_rows["100"]["train_list_sha256"],
        "byte_identical_train_list": p1_list.read_bytes() == (split_dir / f"{tag}_100_seed824.txt").read_bytes(),
        "membership_identical": set(p1_relative)
        == {record.path.relative_to(REPO_ROOT).as_posix() for record in records_by_size[100]},
        "validation_path_identical": p1_data["val"]
        == yaml.safe_load((split_dir / f"{tag}_100.yaml").read_text())["val"],
        "test_path_identical": p1_data["test"] == yaml.safe_load((split_dir / f"{tag}_100.yaml").read_text())["test"],
    }
    reuse["eligible_by_split"] = all(value for key, value in reuse.items() if isinstance(value, bool))
    manifest = {
        "schema_version": 1,
        "dataset": config["label"],
        "split_seed": SPLIT_SEED,
        "selection_strategy": "P1-100 hard anchor; deterministic greedy class-coverage/balance ordering",
        "source_train_images": len(all_images),
        "class_names": names,
        "nested_verification": nested,
        "fixed_evaluation": {
            "val": p1_data["val"],
            "val_images": val_count,
            "val_membership_sha256": val_sha,
            "test": p1_data["test"],
            "test_images": test_count,
            "test_membership_sha256": test_sha,
        },
        "splits": split_rows,
        "p1_100_reuse_audit": reuse,
        "status": "PASS" if all(nested.values()) and reuse["eligible_by_split"] else "FAIL",
    }
    Path(config["manifest"]).parent.mkdir(parents=True, exist_ok=True)
    Path(config["manifest"]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare_run_configs() -> None:
    for tag, dataset in DATASETS.items():
        destination = P2_ROOT / "config" / "runs" / tag
        destination.mkdir(parents=True, exist_ok=True)
        for method in ("full_sft", "frozen_backbone", "vpeft"):
            output = destination / f"{method}.yaml"
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output.relative_to(REPO_ROOT)}")
            source = Path(dataset["p1_config"]) / f"{method}.yaml"
            values = yaml.safe_load(source.read_text(encoding="utf-8"))
            values["data"] = f"smoke/c3/p2/config/splits/{tag}_100.yaml"
            values["epochs"] = 100
            values["seed"] = 824
            output.write_text(yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> int:
    manifests = {tag: prepare_dataset(tag, config) for tag, config in DATASETS.items()}
    prepare_run_configs()
    if not all(manifest["status"] == "PASS" for manifest in manifests.values()):
        raise RuntimeError("Nested split validation failed")
    print("NEU_NESTED_SPLIT=PASS")
    print("DEEPPCB_NESTED_SPLIT=PASS")
    print("P1_100_REUSE_SPLIT_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
