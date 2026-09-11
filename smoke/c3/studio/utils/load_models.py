"""Lazy, single-model inference manager for final P1 checkpoints."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

from utils.load_results import canonical_checkpoint
from utils.paths import relative_path


@dataclass
class InferenceOutput:
    original: Image.Image
    annotated: Image.Image
    detections: pd.DataFrame
    status: str
    device: str
    latency_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


def _external_gpu0_processes() -> list[tuple[int, int]]:
    """Return compute processes on physical GPU 0, excluding this Studio process."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    records = []
    for line in result.stdout.splitlines():
        try:
            pid, memory = (int(value.strip()) for value in line.split(",", maxsplit=1))
        except (TypeError, ValueError):
            continue
        if pid != os.getpid():
            records.append((pid, memory))
    return records


def choose_inference_device() -> tuple[str, str]:
    """Use physical GPU 0 only when available and not occupied by another compute process."""
    if os.environ.get("C3_STUDIO_FORCE_CPU") == "1":
        return "cpu", "CPU forced by C3_STUDIO_FORCE_CPU=1"
    try:
        import torch
    except ImportError:
        return "cpu", "PyTorch unavailable"
    if not torch.cuda.is_available():
        return "cpu", "CUDA unavailable"
    external = _external_gpu0_processes()
    if external:
        used = sum(memory for _, memory in external)
        return "cpu", f"GPU 0 occupied by another process ({used} MiB); using CPU"
    return "0", "GPU 0 available"


class ModelManager:
    """Keep at most one final checkpoint resident and serialize inference calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple[str, str] | None = None
        self._model = None

    def _release(self) -> None:
        self._model = None
        self._key = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _validate_vpeft(model) -> None:
        loaded = model.model
        adapter_tensors = [name for name, _ in loaded.named_parameters() if "lora_" in name]
        if not getattr(loaded, "lora_enabled", False) or getattr(loaded, "lora_backend", None) != "peft":
            raise RuntimeError("Final V-PEFT checkpoint does not contain the expected active PEFT model")
        if len(getattr(loaded, "lora_target_modules", []) or []) <= 0 or not adapter_tensors:
            raise RuntimeError("Final V-PEFT checkpoint has no applied adapter targets/tensors")

    def _load(self, dataset: str, method: str):
        key = (dataset, method)
        if self._key == key and self._model is not None:
            return self._model
        self._release()
        from ultralytics import YOLO

        checkpoint, _ = canonical_checkpoint(dataset, method)
        model = YOLO(checkpoint)
        if method == "V-PEFT":
            self._validate_vpeft(model)
        self._model = model
        self._key = key
        return model

    def infer(
        self,
        image: Image.Image,
        dataset: str,
        method: str,
        confidence: float,
        iou_threshold: float = 0.70,
    ) -> InferenceOutput:
        if image is None:
            raise ValueError("Upload an image before running inference")
        if not 0.01 <= float(confidence) <= 1.0:
            raise ValueError("Confidence threshold must be between 0.01 and 1.00")
        if not 0.01 <= float(iou_threshold) <= 1.0:
            raise ValueError("IoU threshold must be between 0.01 and 1.00")
        original = image.convert("RGB")
        with self._lock:
            model = self._load(dataset, method)
            device, reason = choose_inference_device()
            started = time.perf_counter()
            try:
                results = model.predict(
                    source=np.asarray(original),
                    device=device,
                    imgsz=640,
                    conf=float(confidence),
                    iou=float(iou_threshold),
                    save=False,
                    save_txt=False,
                    save_conf=False,
                    plots=False,
                    verbose=False,
                )
            except RuntimeError as exc:
                if device != "0":
                    raise
                device, reason = "cpu", f"GPU 0 runtime fallback: {exc.__class__.__name__}"
                self._release()
                model = self._load(dataset, method)
                started = time.perf_counter()
                results = model.predict(
                    source=np.asarray(original),
                    device="cpu",
                    imgsz=640,
                    conf=float(confidence),
                    iou=float(iou_threshold),
                    save=False,
                    save_txt=False,
                    save_conf=False,
                    plots=False,
                    verbose=False,
                )
            latency_ms = (time.perf_counter() - started) * 1000.0
            if len(results) != 1:
                raise RuntimeError(f"Expected one inference result, got {len(results)}")
            result = results[0]
            speed = result.speed or {}
            preprocess_ms = float(speed.get("preprocess", 0.0))
            inference_ms = float(speed.get("inference", 0.0))
            postprocess_ms = float(speed.get("postprocess", 0.0))
            annotated = Image.fromarray(result.plot()[..., ::-1])
            names = result.names
            rows = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().tolist()
                scores = result.boxes.conf.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                for box, score, class_id in zip(boxes, scores, classes):
                    rows.append(
                        {
                            "Class": names[int(class_id)],
                            "Confidence": round(float(score), 4),
                            "x1": round(float(box[0]), 1),
                            "y1": round(float(box[1]), 1),
                            "x2": round(float(box[2]), 1),
                            "y2": round(float(box[3]), 1),
                        }
                    )
            detections = pd.DataFrame(rows, columns=["Class", "Confidence", "x1", "y1", "x2", "y2"])
            checkpoint, _ = canonical_checkpoint(dataset, method)
            detection_text = f"{len(rows)} detection(s)" if rows else "No detections"
            status = (
                f"### {detection_text}\n\n"
                f"- **Threshold:** `{float(confidence):.2f}`\n"
                f"- **NMS IoU:** `{float(iou_threshold):.2f}`\n"
                f"- **Device:** `{'GPU 0' if device == '0' else 'CPU'}` — {reason}\n"
                f"- **End-to-end latency:** `{latency_ms:.1f} ms`\n"
                f"- **Stages:** preprocess `{preprocess_ms:.1f} ms` · model `{inference_ms:.1f} ms` · "
                f"postprocess/NMS `{postprocess_ms:.1f} ms`\n"
                f"- **Checkpoint:** `{relative_path(checkpoint)}` (100-epoch final, SHA-256 verified)"
            )
            return InferenceOutput(
                original,
                annotated,
                detections,
                status,
                device,
                latency_ms,
                preprocess_ms,
                inference_ms,
                postprocess_ms,
            )


MODEL_MANAGER = ModelManager()
