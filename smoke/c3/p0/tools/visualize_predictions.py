#!/usr/bin/env python3
"""Create deterministic GT, V-PEFT prediction, comparison, and overview evidence for C3 P0."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[4]
SMOKE_ROOT = REPO_ROOT / "smoke" / "c3" / "p0"
RUN_ROOT = REPO_ROOT / "runs" / "detect" / "runs" / "vpeft_smoke"
PLOT_NAMES = (
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png",
)
PALETTE = (
    (230, 57, 70),
    (29, 130, 230),
    (30, 170, 95),
    (235, 145, 30),
    (150, 80, 210),
    (20, 170, 175),
)
DATASETS = {
    "neu_det": {
        "display_name": "NEU-DET",
        "data": "smoke/c3/p0/config/datasets/neu_det_fewshot.yaml",
        "run_id": "neu_det_vpeft_gpu_fp32_seed824",
        "prediction_json": "smoke/c3/p0/evidence/neu_det_visual_predictions.json",
        "overview": "smoke/c3/p0/visualizations/neu_det_overview.jpg",
    },
    "deeppcb": {
        "display_name": "DeepPCB",
        "data": "smoke/c3/p0/config/datasets/deeppcb_fewshot.yaml",
        "run_id": "deeppcb_vpeft_gpu_fp32_seed824",
        "prediction_json": "smoke/c3/p0/evidence/deeppcb_visual_predictions.json",
        "overview": "smoke/c3/p0/visualizations/deeppcb_overview.jpg",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=824)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true", help="Replace only the C3 visualization output tree.")
    return parser.parse_args()


def repo_path(value: str | Path, *, must_exist: bool = False) -> Path:
    path = (REPO_ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside the repository: {value}") from exc
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def fitted_font(text: str, max_width: int, start_size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Return the largest bundled font size that keeps a label inside its panel."""
    for size in range(start_size, 9, -1):
        candidate = font(size, bold=bold)
        bounds = candidate.getbbox(text)
        if bounds[2] - bounds[0] <= max_width:
            return candidate
    return font(10, bold=bold)


def load_dataset(data_yaml: Path) -> tuple[Path, list[str], str]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = repo_path(config["path"], must_exist=True)
    split = "test" if config.get("test") else "val"
    image_dir = repo_path(root / config[split], must_exist=True)
    names_config = config.get("names", {})
    if isinstance(names_config, dict):
        names = [str(names_config[index]) for index in sorted(names_config)]
    else:
        names = [str(name) for name in names_config]
    if not names:
        raise ValueError(f"No class names in {data_yaml}")
    return image_dir, names, split


def choose_samples(image_dir: Path, count: int, seed: int) -> list[Path]:
    images = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) < count:
        raise ValueError(f"Requested {count} samples but only {len(images)} exist in {image_dir}")
    selected = random.Random(seed).sample(images, count)
    return sorted(selected, key=lambda path: path.name)


def read_ground_truth(image_path: Path, image_size: tuple[int, int], names: list[str]) -> list[dict[str, object]]:
    dataset_root = image_path.parents[2]
    split = image_path.parent.name
    label_path = dataset_root / "labels" / split / f"{image_path.stem}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    width, height = image_size
    objects = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label: {label_path}:{line_number}")
        class_id = int(fields[0])
        if class_id not in range(len(names)):
            raise ValueError(f"Class ID out of range: {label_path}:{line_number}")
        x, y, box_width, box_height = (float(value) for value in fields[1:])
        if not (box_width > 0 and box_height > 0):
            raise ValueError(f"Non-positive GT box: {label_path}:{line_number}")
        x1 = (x - box_width / 2) * width
        y1 = (y - box_height / 2) * height
        x2 = (x + box_width / 2) * width
        y2 = (y + box_height / 2) * height
        if not (-1e-3 <= x1 < x2 <= width + 1e-3 and -1e-3 <= y1 < y2 <= height + 1e-3):
            raise ValueError(f"GT box outside image: {label_path}:{line_number}")
        objects.append(
            {
                "class_id": class_id,
                "class": names[class_id],
                "bbox_xyxy": [
                    round(max(0.0, x1), 3),
                    round(max(0.0, y1), 3),
                    round(min(width, x2), 3),
                    round(min(height, y2), 3),
                ],
            }
        )
    return objects


def prediction_objects(result, names: list[str], image_path: Path) -> list[dict[str, object]]:
    objects = []
    boxes = result.boxes
    if boxes is None:
        return objects
    for xyxy, confidence, class_value in zip(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu()):
        class_id = int(class_value.item())
        if class_id not in range(len(names)):
            raise ValueError(f"Prediction class ID {class_id} is invalid for {image_path}")
        objects.append(
            {
                "image": relative(image_path),
                "class_id": class_id,
                "class": names[class_id],
                "confidence": round(float(confidence.item()), 6),
                "bbox_xyxy": [round(float(value), 3) for value in xyxy.tolist()],
            }
        )
    return objects


def draw_objects(image: Image.Image, objects: list[dict[str, object]], title: str, *, predictions: bool) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    header = max(38, min(64, round(height * 0.09)))
    canvas = Image.new("RGB", (width, height + header), (248, 248, 248))
    canvas.paste(image, (0, header))
    drawing = ImageDraw.Draw(canvas)
    title_font = fitted_font(title, width - 24, max(16, min(28, header // 2)), bold=True)
    label_font = font(max(12, min(20, round(min(width, height) * 0.035))), bold=True)
    drawing.text((12, max(4, header // 5)), title, fill=(20, 20, 20), font=title_font)
    line_width = max(2, round(min(width, height) / 180))
    for item in objects:
        class_id = int(item["class_id"])
        color = PALETTE[class_id % len(PALETTE)]
        x1, y1, x2, y2 = (float(value) for value in item["bbox_xyxy"])
        y1 += header
        y2 += header
        drawing.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text = str(item["class"])
        if predictions:
            text += f" {float(item['confidence']):.2f}"
        text_box = drawing.textbbox((0, 0), text, font=label_font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(header, y1 - text_height - 6)
        label_x = max(0, min(x1, width - text_width - 8))
        drawing.rectangle((label_x, label_y, label_x + text_width + 8, label_y + text_height + 6), fill=color)
        drawing.text((label_x + 4, label_y + 2), text, fill=(255, 255, 255), font=label_font)
    if predictions and not objects:
        note = "No detections at conf=0.25"
        note_font = fitted_font(note, width - 32, max(14, min(24, round(min(width, height) * 0.045))), bold=True)
        note_box = drawing.textbbox((0, 0), note, font=note_font)
        drawing.rectangle((8, header + 8, note_box[2] + 24, header + note_box[3] + 20), fill=(40, 40, 40))
        drawing.text((16, header + 12), note, fill=(255, 255, 255), font=note_font)
    return canvas


def save_jpeg(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=94, subsampling=0)


def comparison_image(gt: Image.Image, prediction: Image.Image) -> Image.Image:
    if gt.size != prediction.size:
        raise ValueError(f"GT/prediction render size mismatch: {gt.size} != {prediction.size}")
    divider = 8
    canvas = Image.new("RGB", (gt.width * 2 + divider, gt.height), (35, 35, 35))
    canvas.paste(gt, (0, 0))
    canvas.paste(prediction, (gt.width + divider, 0))
    return canvas


def make_overview(comparisons: list[tuple[str, Path]], destination: Path, dataset_name: str) -> list[str]:
    chosen = comparisons[:6]
    tile_width = 900
    title_height = 70
    gap = 14
    columns = 2
    rows = (len(chosen) + columns - 1) // columns
    rendered = []
    for sample_id, path in chosen:
        with Image.open(path) as source:
            scale = tile_width / source.width
            tile = source.convert("RGB").resize((tile_width, round(source.height * scale)), Image.Resampling.LANCZOS)
        rendered.append((sample_id, tile))
    tile_height = max(tile.height for _, tile in rendered)
    canvas = Image.new(
        "RGB",
        (columns * tile_width + (columns + 1) * gap, title_height + rows * tile_height + (rows + 1) * gap),
        (242, 242, 242),
    )
    drawing = ImageDraw.Draw(canvas)
    drawing.text(
        (gap, 16), f"{dataset_name}: fixed-seed GT vs V-PEFT predictions", fill=(20, 20, 20), font=font(28, bold=True)
    )
    for index, (sample_id, tile) in enumerate(rendered):
        column = index % columns
        row = index // columns
        x = gap + column * (tile_width + gap)
        y = title_height + gap + row * (tile_height + gap)
        canvas.paste(tile, (x, y))
        drawing.text(
            (x + 8, y + 6),
            sample_id,
            fill=(255, 255, 255),
            stroke_width=3,
            stroke_fill=(0, 0, 0),
            font=font(18, bold=True),
        )
    save_jpeg(canvas, destination)
    return [sample_id for sample_id, _ in chosen]


def existing_plots(run_dir: Path) -> list[str]:
    plots = []
    for name in PLOT_NAMES:
        path = run_dir / name
        if path.is_file():
            plots.append(relative(path))
    plots.extend(relative(path) for path in sorted(run_dir.glob("val_batch*_labels.jpg")))
    plots.extend(relative(path) for path in sorted(run_dir.glob("val_batch*_pred.jpg")))
    return plots


def prepare_output(overwrite: bool) -> Path:
    output = SMOKE_ROOT / "visualizations"
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Visualization output already exists: {relative(output)}; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def visualize_dataset(
    key: str, settings: dict[str, str], args: argparse.Namespace, output_root: Path
) -> dict[str, object]:
    data_yaml = repo_path(settings["data"], must_exist=True)
    image_dir, names, split = load_dataset(data_yaml)
    selected = choose_samples(image_dir, args.samples, args.seed)
    run_dir = RUN_ROOT / settings["run_id"]
    checkpoint = run_dir / "weights" / "best.pt"
    adapter_dir = run_dir / "lora_adapter"
    adapter_model = adapter_dir / "adapter_model.safetensors"
    resolved_config = SMOKE_ROOT / "logs" / settings["run_id"] / "resolved_config.yaml"
    for required in (checkpoint, adapter_model, adapter_dir / "adapter_config.json", resolved_config):
        if not required.is_file():
            raise FileNotFoundError(required)

    model = YOLO(checkpoint)
    loaded = model.model
    adapter_tensors = [name for name, _ in loaded.named_parameters() if "lora_" in name]
    if not getattr(loaded, "lora_enabled", False) or getattr(loaded, "lora_backend", None) != "peft":
        raise RuntimeError(f"{settings['display_name']} best.pt does not contain the expected active PEFT model")
    if len(getattr(loaded, "lora_target_modules", []) or []) <= 0 or not adapter_tensors:
        raise RuntimeError(f"{settings['display_name']} best.pt has no applied adapter targets/tensors")
    model_names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    if model_names != names:
        raise ValueError(f"Checkpoint classes differ from dataset YAML: {model.names} != {names}")

    results = []
    for image_path in selected:
        image_results = model.predict(
            source=str(image_path),
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            save=False,
            save_txt=False,
            save_conf=False,
            plots=False,
            verbose=False,
            project=str(output_root),
            name=f"_{key}_inference",
        )
        if len(image_results) != 1:
            raise RuntimeError(f"Expected one inference result for {image_path}, got {len(image_results)}")
        results.append(image_results[0])
    if len(results) != len(selected):
        raise RuntimeError(f"Expected {len(selected)} inference results, got {len(results)}")
    parameter_device = next(model.model.parameters()).device
    if parameter_device.type != "cuda" or parameter_device.index != int(args.device):
        raise RuntimeError(f"Inference did not use requested CUDA device {args.device}: {parameter_device}")

    dataset_output = output_root / key
    for directory in ("original", "ground_truth", "predictions", "comparison"):
        (dataset_output / directory).mkdir(parents=True, exist_ok=True)
    sample_records = []
    flat_predictions = []
    comparisons = []
    for image_path, result in zip(selected, results):
        if Path(result.path).resolve() != image_path.resolve():
            raise RuntimeError(f"Inference result order/path mismatch: {result.path} != {image_path}")
        sample_id = image_path.stem
        with Image.open(image_path) as source:
            source.load()
            original = source.convert("RGB")
        gt_objects = read_ground_truth(image_path, original.size, names)
        predictions = prediction_objects(result, names, image_path)
        flat_predictions.extend(predictions)
        original_path = dataset_output / "original" / f"{sample_id}.jpg"
        gt_path = dataset_output / "ground_truth" / f"{sample_id}.jpg"
        prediction_path = dataset_output / "predictions" / f"{sample_id}.jpg"
        comparison_path = dataset_output / "comparison" / f"{sample_id}.jpg"
        shutil.copy2(image_path, original_path)
        gt_render = draw_objects(original, gt_objects, "Ground Truth (GT)", predictions=False)
        prediction_render = draw_objects(original, predictions, "V-PEFT Prediction", predictions=True)
        save_jpeg(gt_render, gt_path)
        save_jpeg(prediction_render, prediction_path)
        save_jpeg(comparison_image(gt_render, prediction_render), comparison_path)
        comparisons.append((sample_id, comparison_path))
        sample_records.append(
            {
                "dataset": settings["display_name"],
                "split": split,
                "sample_id": sample_id,
                "sample_path": relative(image_path),
                "gt_object_count": len(gt_objects),
                "prediction_count": len(predictions),
                "outputs": {
                    "original": relative(original_path),
                    "ground_truth": relative(gt_path),
                    "prediction": relative(prediction_path),
                    "comparison": relative(comparison_path),
                },
                "output_sha256": {
                    "original": sha256(original_path),
                    "ground_truth": sha256(gt_path),
                    "prediction": sha256(prediction_path),
                    "comparison": sha256(comparison_path),
                },
            }
        )

    overview = repo_path(settings["overview"])
    overview_sample_ids = make_overview(comparisons, overview, settings["display_name"])
    prediction_json = repo_path(settings["prediction_json"])
    prediction_payload = {
        "schema_version": 1,
        "dataset": settings["display_name"],
        "split": split,
        "seed": args.seed,
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "adapter": relative(adapter_dir),
        "adapter_model_sha256": sha256(adapter_model),
        "model_loading_api": "ultralytics.YOLO(best.pt); best.pt contains the active PEFT wrapper and trained head",
        "inference": {
            "device": args.device,
            "actual_device": str(parameter_device),
            "imgsz": args.imgsz,
            "conf": args.conf,
        },
        "prediction_record_fields": ["image", "class_id", "class", "confidence", "bbox_xyxy"],
        "samples": [
            {
                "image": record["sample_path"],
                "sample_id": record["sample_id"],
                "prediction_count": record["prediction_count"],
            }
            for record in sample_records
        ],
        "predictions": flat_predictions,
    }
    prediction_json.write_text(json.dumps(prediction_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "dataset": settings["display_name"],
        "split": split,
        "data_config": relative(data_yaml),
        "seed": args.seed,
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "adapter": relative(adapter_dir),
        "adapter_model_sha256": sha256(adapter_model),
        "resolved_config": relative(resolved_config),
        "model_loading_api": "ultralytics.YOLO(best.pt)",
        "peft_backend": getattr(loaded, "lora_backend", None),
        "applied_targets": len(getattr(loaded, "lora_target_modules", []) or []),
        "adapter_tensor_count": len(adapter_tensors),
        "inference": {
            "requested_device": args.device,
            "actual_device": str(parameter_device),
            "gpu_name": torch.cuda.get_device_name(parameter_device),
            "imgsz": args.imgsz,
            "conf": args.conf,
        },
        "samples": sample_records,
        "prediction_json": relative(prediction_json),
        "overview": relative(overview),
        "overview_sample_ids": overview_sample_ids,
        "existing_run_plots": existing_plots(run_dir),
    }


def main() -> int:
    args = parse_args()
    if args.samples != 10:
        raise ValueError("C3 visual evidence requires exactly 10 fixed samples per dataset")
    if args.seed != 824 or args.imgsz != 320 or abs(args.conf - 0.25) > 1e-12 or args.device != "0":
        raise ValueError("C3 visual evidence is locked to seed=824, device=0, imgsz=320, conf=0.25")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError("CUDA GPU 0 is required for C3 visual inference")
    output_root = prepare_output(args.overwrite)
    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    datasets = [visualize_dataset(key, DATASETS[key], args, output_root) for key in keys]
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Human-reviewable fixed-sample GT and real V-PEFT inference evidence; no training performed",
        "selection_policy": "Exactly 10 images sampled from each official test split with independent random.Random(824); no result-based replacement",
        "model_loading": "The repository YOLO(best.pt) API loads each checkpoint's active PEFT wrapper and trained detection head; exported adapter directories are recorded but not merged or modified.",
        "inference": {"device": args.device, "imgsz": args.imgsz, "conf": args.conf, "seed": args.seed},
        "datasets": datasets,
        "verification": {
            "ground_truth_loaded_from_yolo_labels": True,
            "prediction_generated_by_checkpoint": True,
            "comparison_generated_for_all_samples": all(len(item["samples"]) == 10 for item in datasets),
            "no_training_performed": True,
        },
    }
    manifest_path = SMOKE_ROOT / "evidence" / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": relative(manifest_path),
                "datasets": {item["dataset"]: len(item["samples"]) for item in datasets},
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
