"""Loopback-only photo fallback and model artifact server for C3 Edge Lab development."""

from __future__ import annotations

import argparse
import hmac
import io
import json
import os
import sys
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

MINIPROGRAM_ROOT = Path(__file__).resolve().parent
STUDIO_ROOT = MINIPROGRAM_ROOT.parent
DIST_ROOT = MINIPROGRAM_ROOT / "dist"
MODEL_ROOT = DIST_ROOT / "models"
sys.path.insert(0, str(STUDIO_ROOT))

from utils.load_models import MODEL_MANAGER
from utils.load_results import DATASETS, METHODS

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
app = FastAPI(title="C3 Edge Lab Fallback", version="0.1.0", docs_url=None, redoc_url=None)


def require_api_key(x_c3_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("C3_EDGE_API_KEY", "")
    if expected and not hmac.compare_digest(x_c3_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


def manifest_models() -> dict[str, dict]:
    manifest_path = DIST_ROOT / "model-manifest.json"
    if not manifest_path.is_file():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {record["file"]: record for record in payload.get("models", [])}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "c3-edge-fallback",
        "models": len(manifest_models()),
        "loopback_default": True,
    }


@app.get("/v1/models")
def list_models(_: None = Depends(require_api_key)) -> dict[str, object]:
    return {"models": list(manifest_models().values())}


@app.get("/v1/models/{filename}")
def download_model(filename: str, _: None = Depends(require_api_key)) -> FileResponse:
    records = manifest_models()
    record = records.get(filename)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown model artifact")
    path = MODEL_ROOT / filename
    if not path.is_file() or path.parent.resolve() != MODEL_ROOT.resolve():
        raise HTTPException(status_code=404, detail="Model artifact is not present")
    return FileResponse(
        path, media_type="application/octet-stream", filename=filename, headers={"X-SHA256": record["sha256"]}
    )


@app.post("/v1/infer")
async def infer(
    file: Annotated[UploadFile, File()],
    dataset: Annotated[str, Form()],
    method: Annotated[str, Form()] = "V-PEFT",
    confidence: Annotated[float, Form()] = 0.25,
    iou: Annotated[float, Form()] = 0.45,
    _: None = Depends(require_api_key),
) -> dict[str, object]:
    if dataset not in DATASETS or method not in METHODS:
        raise HTTPException(status_code=400, detail="Unsupported dataset or method")
    if not 0.01 <= confidence <= 1.0 or not 0.01 <= iou <= 1.0:
        raise HTTPException(status_code=400, detail="confidence and iou must be between 0.01 and 1.0")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MiB")
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Invalid image") from error
    output = MODEL_MANAGER.infer(image, dataset, method, confidence, iou_threshold=iou)
    detections = [
        {
            "label": row["Class"],
            "confidence": float(row["Confidence"]),
            "x1": float(row["x1"]),
            "y1": float(row["y1"]),
            "x2": float(row["x2"]),
            "y2": float(row["y2"]),
        }
        for row in output.detections.to_dict(orient="records")
    ]
    return {
        "dataset": dataset,
        "method": method,
        "width": image.width,
        "height": image.height,
        "detections": detections,
        "latency_ms": output.latency_ms,
        "device": "GPU 0" if output.device == "0" else "CPU",
        "timings": {
            "preprocess_ms": output.preprocess_ms,
            "inference_ms": output.inference_ms,
            "postprocess_ms": output.postprocess_ms,
            "total_ms": output.latency_ms,
        },
        "confidence": confidence,
        "iou": iou,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7863)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
