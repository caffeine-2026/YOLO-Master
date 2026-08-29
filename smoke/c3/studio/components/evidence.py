"""Evidence and reproduction tab."""

from __future__ import annotations

import gradio as gr
from utils.load_results import evidence_catalog, source_manifest
from utils.paths import git_info, recorded_refs


def metadata_markdown() -> str:
    git = git_info()
    refs = recorded_refs()
    return f"""
### Provenance

| Field | Value |
|---|---|
| BASE_REF | `{refs['BASE_REF']}` |
| P0 current ref | `{refs['P0_REF']}` |
| P1 current ref | `{refs['P1_REF']}` |
| P2 current ref | `{refs['P2_REF']}` |
| Current commit | `{git['commit']}` |
| Current branch | `{git['branch']}` |

Stage-specific refs are shown as “Not recorded” when no existing evidence file records them.
"""


def sources_markdown() -> str:
    sources = source_manifest()
    sections = []
    for name, paths in sources.items():
        sections.append(f"**{name}**\n\n" + "\n".join(f"- `{path}`" for path in paths))
    return "\n\n".join(sections)


def build_tab() -> None:
    gr.Markdown(
        "## Evidence / Reproduction\nRepository-relative evidence paths only. Commands below validate existing artifacts or launch the Studio; "
        "they do not start training."
    )
    gr.Markdown(metadata_markdown())
    gr.Dataframe(value=evidence_catalog(), interactive=False, wrap=True, label="Evidence index")
    with gr.Accordion("Data sources", open=True):
        gr.Markdown(sources_markdown())
    with gr.Accordion("Safe reproduction commands (not auto-executed)", open=True):
        gr.Code(
            value=(
                ".venv/bin/python smoke/c3/p0/tools/validate_delivery.py\n"
                ".venv/bin/python smoke/c3/p2/tools/validate_p2_final.py\n"
                "CUDA_VISIBLE_DEVICES=0 .venv/bin/python smoke/c3/studio/app.py"
            ),
            language="shell",
            interactive=False,
        )
