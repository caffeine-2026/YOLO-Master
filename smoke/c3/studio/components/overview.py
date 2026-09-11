"""Overview tab."""

from __future__ import annotations

import html

import gradio as gr
from utils.load_results import overview_metrics, stage_status


def _retention_cards() -> str:
    metrics = overview_metrics()
    frame = metrics["retention"]
    cards = []
    for dataset in ("NEU-DET", "DeepPCB"):
        part = frame[frame["dataset"] == dataset]
        values = "".join(
            f'<div class="retention-row"><span>{int(row.sample_size)}-shot</span><strong>{row.accuracy_retention:.2%}</strong></div>'
            for row in part.itertuples()
        )
        cards.append(
            f'<section class="insight-card"><div class="eyebrow">V-PEFT accuracy retention</div>'
            f'<h3>{html.escape(dataset)}</h3>{values}</section>'
        )
    return "".join(cards)


def overview_html() -> str:
    status = stage_status()
    metrics = overview_metrics()
    status_cards = "".join(
        f'<div class="status-card"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in (
            ("P0", status["P0"]),
            ("P1", status["P1"]),
            ("P2", status["P2"]),
            ("P2 Matrix", f'{status["matrix_pass"]}/{status["matrix_expected"]} PASS'),
        )
    )
    return f"""
    <div class="hero-panel">
      <div class="eyebrow">C3 Industrial PEFT Studio · v0.1</div>
      <h1>C3｜工业缺陷检测：V-PEFT 小样本实战</h1>
      <p>NEU-DET 与 DeepPCB · Full-SFT / Frozen Backbone / V-PEFT · 3 seeds · 10/50/100/500-shot</p>
      <div class="status-grid">{status_cards}</div>
    </div>
    <div class="overview-grid">
      <section class="insight-card highlight-card">
        <div class="eyebrow">Trainable parameter reduction</div>
        <div class="hero-number">{metrics['parameter_reduction']:.2%}</div>
        <p>V-PEFT 相对 Full-SFT；由现有 P2 retention CSV 自动读取。</p>
      </section>
      <section class="insight-card">
        <div class="eyebrow">Experiment scope</div>
        <h3>2 datasets · 3 methods · 4 scales</h3>
        <p>所有数字来自已完成的 P1/P2 CSV 与验证 JSON；Studio 不训练、不改写实验结果。</p>
      </section>
      {_retention_cards()}
    </div>
    <div class="boundary-callout">
      <strong>Interpretation boundary</strong>
      <span>V-PEFT 不是 universal winner。其显存节省仅约 {metrics['memory_saving_min']:.2%}–{metrics['memory_saving_max']:.2%}，
      训练时间相对 Full-SFT 增加约 {metrics['training_time_min']:.2%}–{metrics['training_time_max']:.2%}。
      优势集中在可训练参数规模与部分小样本精度保持，而非所有精度/速度/显存维度。</span>
    </div>
    """


def build_tab() -> None:
    gr.HTML(overview_html())
