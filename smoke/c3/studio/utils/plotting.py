"""Matplotlib charts for immutable C3 result tables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.load_results import METHODS, comparison_table, scaling_data

COLORS = {"Full-SFT": "#3B82F6", "Frozen Backbone": "#94A3B8", "V-PEFT": "#F59E0B"}


def _style_axes(axis: plt.Axes) -> None:
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#CBD5E1")


def comparison_figure(dataset: str, sample_size: int) -> plt.Figure:
    frame = comparison_table(dataset, int(sample_size))
    colors = [COLORS[name] for name in frame["Method"]]
    panels = [
        ("mAP50-95", "Accuracy (mean)", "mAP50-95"),
        ("Trainable Params", "Trainable parameters", "Parameters"),
        ("Peak GPU Memory (MiB)", "Peak GPU memory", "MiB"),
        ("Training Time (s)", "Training time", "Seconds"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    for axis, (column, title, ylabel) in zip(axes.flat, panels):
        values = frame[column].astype(float).to_numpy()
        bars = axis.bar(frame["Method"], values, color=colors, width=0.64)
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#0F172A")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=10)
        _style_axes(axis)
        labels = [f"{value:.4f}" if column == "mAP50-95" else f"{value:,.0f}" for value in values]
        axis.bar_label(bars, labels=labels, padding=3, fontsize=8, color="#334155")
        axis.margins(y=0.15)
    figure.suptitle(f"{dataset} · {sample_size}-shot · 3-way comparison", fontsize=15, fontweight="bold")
    return figure


def scaling_figure(dataset: str, metric: str) -> plt.Figure:
    frame = scaling_data(dataset, metric)
    figure, axis = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    for method in METHODS:
        part = frame[frame["method"] == method]
        x = part["sample_size"].astype(int).to_numpy()
        y = part["mean"].astype(float).to_numpy()
        lower = np.maximum(0.0, y - part["lower"].astype(float).to_numpy())
        upper = np.maximum(0.0, part["upper"].astype(float).to_numpy() - y)
        axis.errorbar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            marker="o",
            markersize=7,
            linewidth=2.2,
            capsize=4,
            label=method,
            color=COLORS[method],
        )
    axis.set_xscale("log")
    axis.set_xticks([10, 50, 100, 500], labels=["10", "50", "100", "500"])
    axis.set_xlabel("Few-shot training samples")
    ylabel = "Retention vs Full-SFT" if metric == "Accuracy Retention" else metric
    axis.set_ylabel(ylabel)
    if metric == "Accuracy Retention":
        axis.axhline(1.0, color="#64748B", linestyle="--", linewidth=1.2)
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.set_title(f"{dataset} · {metric} scaling (mean ± 95% CI)", loc="left", fontsize=14, fontweight="bold")
    axis.legend(frameon=False, ncol=3, loc="best")
    _style_axes(axis)
    return figure
