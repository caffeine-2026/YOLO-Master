"""Three-way comparison tab."""

from __future__ import annotations

import gradio as gr
from utils.load_results import DATASETS, SAMPLE_SIZES, comparison_table
from utils.plotting import comparison_figure


def comparison_outputs(dataset: str, sample_size: int):
    sample_size = int(sample_size)
    table = comparison_table(dataset, sample_size)
    figure = comparison_figure(dataset, sample_size)
    note = (
        f"**Current cell:** `{dataset}` · `{sample_size}-shot` · multi-seed mean. "
        "Parameter reduction is derived from each row's CSV trainable parameter count relative to Full-SFT."
    )
    return table, figure, note


def build_tab() -> None:
    gr.Markdown(
        "## 3-Way Comparison\nSelect a dataset and few-shot size. The table and all four charts are generated from "
        "`smoke/c3/p2/results/p2_summary.csv`."
    )
    with gr.Row():
        dataset = gr.Dropdown(DATASETS, value="NEU-DET", label="Dataset", interactive=True)
        sample_size = gr.Dropdown(SAMPLE_SIZES, value=100, label="Sample size", interactive=True)
    initial_table, initial_figure, initial_note = comparison_outputs("NEU-DET", 100)
    note = gr.Markdown(initial_note)
    table = gr.Dataframe(value=initial_table, interactive=False, wrap=True, label="Multi-seed comparison")
    figure = gr.Plot(value=initial_figure, label="Accuracy / parameters / memory / time")
    dataset.change(comparison_outputs, [dataset, sample_size], [table, figure, note])
    sample_size.change(comparison_outputs, [dataset, sample_size], [table, figure, note])
