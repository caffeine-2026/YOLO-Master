"""Few-shot scaling tab."""

from __future__ import annotations

import gradio as gr
import pandas as pd
from utils.load_results import DATASETS, scaling_data
from utils.plotting import scaling_figure

METRICS = ("mAP50-95", "mAP50", "Accuracy Retention", "Peak GPU Memory", "GPU-hours")


def scaling_outputs(dataset: str, metric: str):
    frame = scaling_data(dataset, metric).copy()
    display = pd.DataFrame(
        {
            "Sample size": frame["sample_size"].astype(int),
            "Method": frame["method"].astype(str),
            "Mean": frame["mean"].round(5),
            "95% CI lower": frame["lower"].round(5),
            "95% CI upper": frame["upper"].round(5),
        }
    )
    return scaling_figure(dataset, metric), display, f"**Current view:** `{dataset}` · `{metric}` · mean ± 95% CI"


def build_tab() -> None:
    gr.Markdown(
        "## Few-shot Scaling\nMulti-seed scaling across 10 / 50 / 100 / 500 samples. "
        "Published P2 mean and confidence intervals are used where available; accuracy retention is computed per seed "
        "against the matching Full-SFT run."
    )
    with gr.Row():
        dataset = gr.Dropdown(DATASETS, value="NEU-DET", label="Dataset", interactive=True)
        metric = gr.Radio(METRICS, value="mAP50-95", label="Metric", interactive=True)
    initial_figure, initial_table, initial_note = scaling_outputs("NEU-DET", "mAP50-95")
    note = gr.Markdown(initial_note)
    figure = gr.Plot(value=initial_figure, label="Scaling curve")
    table = gr.Dataframe(value=initial_table, interactive=False, wrap=True, label="Underlying values")
    dataset.change(scaling_outputs, [dataset, metric], [figure, table, note])
    metric.change(scaling_outputs, [dataset, metric], [figure, table, note])
