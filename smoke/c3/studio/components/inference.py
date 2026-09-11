"""Live inference tab."""

from __future__ import annotations

import time

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


def run_camera_frame(image, dataset: str, method: str, confidence: float, previous_time: float | None):
    """Run one latest-available browser-camera frame without building a backlog."""
    if image is None:
        return None, "Start the camera to begin live inference.", previous_time
    try:
        output = MODEL_MANAGER.infer(image, dataset, method, float(confidence))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return image, f"### Live inference paused\n\n{exc}", previous_time

    completed_at = time.perf_counter()
    observed_fps = (
        1.0 / max(completed_at - float(previous_time), 1e-6)
        if previous_time is not None
        else 1000.0 / max(output.latency_ms, 1e-6)
    )
    processing_fps = 1000.0 / max(output.latency_ms, 1e-6)
    detection_count = len(output.detections.index)
    stats = (
        f"### Live · {observed_fps:.1f} FPS\n\n"
        f"- **Detections:** `{detection_count}`\n"
        f"- **Observed stream FPS:** `{observed_fps:.1f}`\n"
        f"- **Processing capacity:** `{processing_fps:.1f} FPS` (`{output.latency_ms:.1f} ms/frame`)\n"
        f"- **Stages:** preprocess `{output.preprocess_ms:.1f} ms` · model `{output.inference_ms:.1f} ms` · "
        f"postprocess/NMS `{output.postprocess_ms:.1f} ms`\n"
        f"- **Device:** `{'GPU 0' if output.device == '0' else 'CPU'}`\n"
        f"- **Threshold:** `{float(confidence):.2f}`"
    )
    return output.annotated, stats, completed_at


def build_tab() -> None:
    gr.Markdown(
        "## Live Inference\nUpload an image and run real YOLO inference using the SHA-256 verified, 100-epoch final P1 checkpoint. "
        "The V-PEFT checkpoint is loaded through the repository's existing active-PEFT checkpoint path. "
        "The browser-camera mode streams the latest frame for server-side inference. "
        "GPU 0 is used only when free; otherwise inference falls back to CPU."
    )
    with gr.Row():
        dataset = gr.Dropdown(DATASETS, value="NEU-DET", label="Dataset", interactive=True)
        method = gr.Dropdown(METHODS, value="V-PEFT", label="Model", interactive=True)
        confidence = gr.Slider(0.01, 1.0, value=0.25, step=0.01, label="Confidence threshold")
    threshold = gr.Markdown(threshold_text(0.25))
    confidence.change(threshold_text, confidence, threshold)
    with gr.Tabs():
        with gr.Tab("Upload Image"):
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
                concurrency_id="model-inference",
            )

        with gr.Tab("Realtime Camera"):
            gr.Markdown(
                "Start the browser camera below. Frames are sent to this Studio server and the newest available frame is "
                "annotated; this measures server inference, not iPhone Core ML performance."
            )
            camera_clock = gr.State(value=None)
            with gr.Row():
                camera = gr.Image(
                    type="pil",
                    sources=["webcam"],
                    streaming=True,
                    label="Camera",
                )
                camera_result = gr.Image(
                    type="pil",
                    streaming=True,
                    label="Live detection",
                    interactive=False,
                )
            camera_stats = gr.Markdown("Start the camera to begin live inference.")
            camera.stream(
                run_camera_frame,
                inputs=[camera, dataset, method, confidence, camera_clock],
                outputs=[camera_result, camera_stats, camera_clock],
                stream_every=0.10,
                time_limit=300,
                trigger_mode="always_last",
                concurrency_limit=1,
                concurrency_id="model-inference",
                show_progress="hidden",
            )
