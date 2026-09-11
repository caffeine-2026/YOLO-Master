"""Live inference tab."""

from __future__ import annotations

import gradio as gr
from utils.load_models import MODEL_MANAGER
from utils.load_results import DATASETS, METHODS


def run_inference(image, dataset: str, method: str, confidence: float):
    try:
        output = MODEL_MANAGER.infer(image, dataset, method, float(confidence))
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(f"Inference failed safely: {exc}") from exc
    return output.original, output.annotated, output.detections, output.status


def threshold_text(confidence: float) -> str:
    return f"Current confidence threshold: **{float(confidence):.2f}**"


def build_tab() -> None:
    gr.Markdown(
        "## Live Inference\nUpload an image and run real YOLO inference using the SHA-256 verified, 100-epoch final P1 checkpoint. "
        "The V-PEFT checkpoint is loaded through the repository's existing active-PEFT checkpoint path. "
        "GPU 0 is used only when free; otherwise inference falls back to CPU."
    )
    with gr.Row():
        dataset = gr.Dropdown(DATASETS, value="NEU-DET", label="Dataset", interactive=True)
        method = gr.Dropdown(METHODS, value="V-PEFT", label="Model", interactive=True)
        confidence = gr.Slider(0.01, 1.0, value=0.25, step=0.01, label="Confidence threshold")
    threshold = gr.Markdown(threshold_text(0.25))
    confidence.change(threshold_text, confidence, threshold)
    upload = gr.Image(type="pil", label="Upload image", sources=["upload"])
    run_button = gr.Button("Run real inference", variant="primary")
    with gr.Row():
        original = gr.Image(type="pil", label="Original")
        annotated = gr.Image(type="pil", label="Detection result")
    detections = gr.Dataframe(
        headers=["Class", "Confidence", "x1", "y1", "x2", "y2"],
        interactive=False,
        label="Detections",
    )
    status = gr.Markdown("Upload an image to begin. No threshold is lowered automatically.")
    run_button.click(
        run_inference,
        inputs=[upload, dataset, method, confidence],
        outputs=[original, annotated, detections, status],
        concurrency_limit=1,
    )
