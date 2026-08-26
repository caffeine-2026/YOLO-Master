#!/usr/bin/env python3
"""Create fixed-sample four-column P1 qualitative comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
CONFIDENCE = 0.25
PANEL_SIZE = 640
TITLE_HEIGHT = 44
COLORS = (
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
)
DATASETS = {
    "neu_det": {
        "label": "NEU-DET",
        "output": "neu_det",
        "test": REPO_ROOT / "datasets" / "neu_det_yolo" / "images" / "test",
        "labels": REPO_ROOT / "datasets" / "neu_det_yolo" / "labels" / "test",
    },
    "deeppcb": {
        "label": "DeepPCB",
        "output": "deeppcb",
        "test": REPO_ROOT / "datasets" / "deeppcb_yolo" / "images" / "test",
        "labels": REPO_ROOT / "datasets" / "deeppcb_yolo" / "labels" / "test",
    },
}
METHODS = (
    ("Full-SFT", "full"),
    ("Frozen", "frozen"),
    ("V-PEFT", "vpeft"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="6")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_samples(test_dir: Path) -> list[Path]:
    images = sorted(path for path in test_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) < 10:
        raise ValueError(f"Need at least 10 test images in {test_dir.relative_to(REPO_ROOT)}")
    # Ten equally spaced lexicographic positions are fixed before inference and
    # depend only on the test membership, never on model predictions.
    indices = [round(index * (len(images) - 1) / 9) for index in range(10)]
    return [images[index] for index in indices]


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    canvas = np.full((PANEL_SIZE + TITLE_HEIGHT, PANEL_SIZE, 3), 255, dtype=np.uint8)
    canvas[TITLE_HEIGHT:] = cv2.resize(image, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, title, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def draw_box(image: np.ndarray, box: tuple[float, float, float, float], label: str, class_id: int) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    p1 = (max(0, min(width - 1, round(x1))), max(0, min(height - 1, round(y1))))
    p2 = (max(0, min(width - 1, round(x2))), max(0, min(height - 1, round(y2))))
    color = COLORS[class_id % len(COLORS)]
    thickness = max(1, round(min(width, height) / 240))
    cv2.rectangle(image, p1, p2, color, thickness, cv2.LINE_AA)
    scale = max(0.35, min(width, height) / 1000)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    text_y = max(text_size[1] + 4, p1[1])
    cv2.rectangle(image, (p1[0], text_y - text_size[1] - 4), (p1[0] + text_size[0] + 4, text_y + 2), color, -1)
    cv2.putText(image, label, (p1[0] + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def ground_truth_panel(image_path: Path, label_dir: Path, names: dict[int, str]) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read {image_path.relative_to(REPO_ROOT)}")
    height, width = image.shape[:2]
    label_path = label_dir / f"{image_path.stem}.txt"
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id_text, x_text, y_text, w_text, h_text, *_ = line.split()
        class_id = int(class_id_text)
        x, y, box_width, box_height = map(float, (x_text, y_text, w_text, h_text))
        box = (
            (x - box_width / 2) * width,
            (y - box_height / 2) * height,
            (x + box_width / 2) * width,
            (y + box_height / 2) * height,
        )
        draw_box(image, box, f"GT {names[class_id]}", class_id)
    return add_title(image, "Ground Truth")


def prediction_panel(result: object, title: str, names: dict[int, str]) -> np.ndarray:
    image = result.orig_img.copy()
    boxes = result.boxes
    if boxes is not None:
        for coordinates, confidence, class_id_value in zip(
            boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), strict=True
        ):
            class_id = int(class_id_value)
            draw_box(image, tuple(coordinates), f"{names[class_id]} {confidence:.2f}", class_id)
    return add_title(image, title)


def main() -> int:
    args = parse_args()
    final_root = P1_ROOT / "visualizations" / "final"
    manifest_path = final_root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path.relative_to(REPO_ROOT)}")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "selection_rule": "10 equally spaced positions in the lexicographically sorted fixed test split",
        "reference_seed": 824,
        "reference_seed_rationale": "pre-designated completed seed; no metric-based checkpoint selection",
        "confidence_threshold": CONFIDENCE,
        "imgsz": PANEL_SIZE,
        "device": args.device,
        "datasets": {},
    }

    for dataset, config in DATASETS.items():
        output_dir = final_root / str(config["output"])
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir.relative_to(REPO_ROOT)}")
        output_dir.mkdir(parents=True)
        data_config = yaml.safe_load((P1_ROOT / "config" / dataset / "dataset.yaml").read_text(encoding="utf-8"))
        names = {int(key): str(value) for key, value in data_config["names"].items()}
        samples = fixed_samples(Path(config["test"]))
        dataset_tag = "neu" if dataset == "neu_det" else "deeppcb"
        checkpoints = {
            title: P1_ROOT / "artifacts" / f"{dataset_tag}_{method}_seed824_e100" / "weights" / "best.pt"
            for title, method in METHODS
        }
        predictions = {}
        for title, checkpoint in checkpoints.items():
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint.relative_to(REPO_ROOT))
            model = YOLO(checkpoint)
            predictions[title] = model.predict(
                source=[str(path) for path in samples],
                conf=CONFIDENCE,
                imgsz=PANEL_SIZE,
                device=args.device,
                verbose=False,
            )

        output_rows = []
        for index, image_path in enumerate(samples):
            panels = [ground_truth_panel(image_path, Path(config["labels"]), names)]
            panels.extend(prediction_panel(predictions[title][index], title, names) for title, _ in METHODS)
            comparison = cv2.hconcat(panels)
            output = output_dir / f"{index + 1:02d}_{image_path.stem}.png"
            if not cv2.imwrite(str(output), comparison, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
                raise RuntimeError(f"Failed to write {output.relative_to(REPO_ROOT)}")
            output_rows.append(
                {
                    "source": image_path.relative_to(REPO_ROOT).as_posix(),
                    "source_sha256": sha256(image_path),
                    "comparison": output.relative_to(REPO_ROOT).as_posix(),
                    "comparison_sha256": sha256(output),
                }
            )
        manifest["datasets"][dataset] = {
            "label": config["label"],
            "sample_count": len(samples),
            "checkpoints": {
                title: {"path": checkpoint.relative_to(REPO_ROOT).as_posix(), "sha256": sha256(checkpoint)}
                for title, checkpoint in checkpoints.items()
            },
            "samples": output_rows,
        }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("QUALITATIVE_COMPARISONS=20/20")
    print(f"CONFIDENCE_THRESHOLD={CONFIDENCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
