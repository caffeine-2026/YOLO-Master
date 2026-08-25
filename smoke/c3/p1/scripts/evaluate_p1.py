#!/usr/bin/env python3
"""Evaluate one completed P1 checkpoint on the locked test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def repo_path(value: str, *, must_exist: bool = False) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"Only repository-relative paths are accepted: {value}")
    resolved = (REPO_ROOT / candidate).resolve()
    resolved.relative_to(REPO_ROOT)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(value)
    return resolved


def main() -> int:
    args = parse_args()
    model_path = repo_path(args.model, must_exist=True)
    data_path = repo_path(args.data, must_exist=True)
    save_dir = repo_path(args.save_dir)
    output = repo_path(args.output)
    if save_dir.exists():
        raise FileExistsError(f"Refusing to overwrite test evaluation directory: {args.save_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    metrics = model.val(
        data=data_path.as_posix(),
        split="test",
        device=args.device,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=0,
        quantize=32,
        plots=True,
        save_json=False,
        save_dir=save_dir.as_posix(),
    )
    values = {key: float(value) for key, value in metrics.results_dict.items()}
    payload = {
        "schema_version": 1,
        "split": "test",
        "device": args.device,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "precision": values["metrics/precision(B)"],
        "recall": values["metrics/recall(B)"],
        "map50": values["metrics/mAP50(B)"],
        "map50_95": values["metrics/mAP50-95(B)"],
        "fitness": values.get("fitness"),
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "save_dir": args.save_dir,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
