"""C3 Industrial PEFT Studio v0.1."""

from __future__ import annotations

import argparse
import os

# Enforce the product boundary before any optional Torch import: physical GPU 0 is the only visible CUDA device.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("MPLBACKEND", "Agg")

import gradio as gr
from components import comparison, evidence, inference, overview, planner, scaling
from utils.paths import REPO_ROOT, STUDIO_ROOT, find_available_port

CSS = """
:root {
  --ink: #0f172a;
  --muted: #64748b;
  --line: #dbe4ee;
  --paper: #f7f9fc;
  --blue: #2563eb;
  --amber: #f59e0b;
  --green: #16a34a;
}
.gradio-container { background: var(--paper) !important; color: var(--ink); }
.main { max-width: 1480px !important; margin: 0 auto; }
.hero-panel {
  padding: 34px; border-radius: 22px; color: white;
  background: radial-gradient(circle at 88% 5%, rgba(245,158,11,.32), transparent 25%),
              linear-gradient(125deg, #0f172a 0%, #172554 55%, #1d4ed8 100%);
  box-shadow: 0 24px 55px rgba(15,23,42,.16); margin-bottom: 20px;
}
.hero-panel h1 { margin: 7px 0 8px; font-size: clamp(27px, 3vw, 44px); line-height: 1.12; letter-spacing: -.03em; }
.hero-panel p { margin: 0; color: #dbeafe; font-size: 16px; }
.eyebrow { text-transform: uppercase; letter-spacing: .12em; font-weight: 800; font-size: 11px; color: #93c5fd; }
.status-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin-top: 26px; }
.status-card { padding: 14px 16px; border: 1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.08); border-radius: 14px; backdrop-filter: blur(8px); }
.status-card span { display: block; color: #bfdbfe; font-size: 12px; }
.status-card strong { display: block; margin-top: 3px; font-size: 18px; color: #fff; }
.overview-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 15px; }
.insight-card { background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 22px; box-shadow: 0 8px 22px rgba(15,23,42,.04); }
.insight-card .eyebrow { color: var(--blue); }
.insight-card h3 { margin: 6px 0 13px; }
.insight-card p { color: var(--muted); margin-bottom: 0; }
.highlight-card { border-top: 4px solid var(--amber); }
.hero-number { font-size: 49px; font-weight: 900; letter-spacing: -.04em; color: var(--ink); margin-top: 5px; }
.retention-row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #eef2f7; color: var(--muted); }
.retention-row:last-child { border-bottom: 0; }
.retention-row strong { color: var(--ink); }
.boundary-callout { display: flex; gap: 16px; align-items: flex-start; background: #fffbeb; border: 1px solid #fde68a; color: #78350f; padding: 18px 20px; border-radius: 15px; margin-top: 16px; }
.boundary-callout strong { white-space: nowrap; }
.planner-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; }
.planner-stat { background: #fff; border: 1px solid var(--line); border-radius: 15px; padding: 18px; }
.planner-stat span { color: var(--muted); display: block; font-size: 12px; }
.planner-stat strong { font-size: 21px; display: block; margin-top: 4px; }
.section-title { margin: 26px 0 12px; }
.flow-wrap { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; background: #fff; border: 1px solid var(--line); border-radius: 17px; padding: 20px; }
.flow-node { background: #eff6ff; border: 1px solid #bfdbfe; color: #1e3a8a; padding: 10px 12px; border-radius: 10px; font-weight: 650; }
.flow-arrow { color: var(--amber); font-weight: 900; font-size: 18px; }
.source-note { color: var(--muted); margin-top: 12px; font-size: 13px; }
@media (max-width: 760px) {
  .status-grid, .planner-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .overview-grid { grid-template-columns: 1fr; }
  .boundary-callout { display: block; }
}
"""


def create_demo() -> gr.Blocks:
    """Build the full six-tab Studio without loading a model or initializing CUDA."""
    with gr.Blocks(title="C3 Industrial PEFT Studio", fill_width=True) as demo, gr.Tabs():
        with gr.Tab("Overview", id="overview"):
            overview.build_tab()
        with gr.Tab("3-Way Comparison", id="comparison"):
            comparison.build_tab()
        with gr.Tab("Few-shot Scaling", id="scaling"):
            scaling.build_tab()
        with gr.Tab("Live Inference", id="inference"):
            inference.build_tab()
        with gr.Tab("V-PEFT Planner", id="planner"):
            planner.build_tab()
        with gr.Tab("Evidence / Reproduction", id="evidence"):
            evidence.build_tab()
    return demo.queue(default_concurrency_limit=1, max_size=16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch C3 Industrial PEFT Studio")
    parser.add_argument("--port", type=int, choices=(7860, 7861, 7862), help="Loopback port; auto-selects if omitted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    port = args.port or find_available_port()
    demo = create_demo()
    print(f"C3_STUDIO_URL=http://127.0.0.1:{port}", flush=True)
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        show_error=True,
        css=CSS,
        allowed_paths=[str(STUDIO_ROOT / "assets")],
        blocked_paths=[str(REPO_ROOT)],
    )


if __name__ == "__main__":
    main()
