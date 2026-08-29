"""V-PEFT planner tab."""

from __future__ import annotations

import html

import gradio as gr
from utils.load_results import planner_data

FLOW = (
    "Input model + dataset",
    "PEFTPlanner",
    "Budget / Architecture Analysis",
    "ACCEPT / ADAPT / REFUSE",
    "Target Selection",
    "V-PEFT / LoRA Injection",
    "Training",
    "Adapter Export",
    "Evaluation",
)


def planner_html() -> str:
    data = planner_data()
    rows = (
        ("Planner status", data["status"]),
        ("Planner backend", data["planner_backend"]),
        ("Actual backend", data["actual_backend"]),
        ("Planned targets", f'{int(data["planned_targets"]):,}'),
        ("Applied targets", f'{int(data["applied_targets"]):,}'),
        ("Trainable params", f'{int(data["trainable_params"]):,}'),
        ("Adapter params", f'{int(data["adapter_params"]):,}'),
        ("Parameter reduction", f'{float(data["parameter_reduction"]):.2%}'),
    )
    cards = "".join(
        f'<div class="planner-stat"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'
        for label, value in rows
    )
    flow_parts = []
    for index, node in enumerate(FLOW):
        flow_parts.append(f'<div class="flow-node">{html.escape(node)}</div>')
        if index < len(FLOW) - 1:
            flow_parts.append('<div class="flow-arrow">→</div>')
    flow = "".join(flow_parts)
    return f"""
    <div class="planner-grid">{cards}</div>
    <h3 class="section-title">Planner-to-evaluation flow</h3>
    <div class="flow-wrap">{flow}</div>
    <div class="source-note">Source: <code>{html.escape(str(data['source']))}</code></div>
    """


def build_tab() -> None:
    gr.Markdown(
        "## V-PEFT Planner\nPlanner facts are read from the canonical V-PEFT run's existing `metrics.json`; no planner outcome is hard-coded."
    )
    gr.HTML(planner_html())
