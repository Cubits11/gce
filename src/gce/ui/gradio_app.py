from __future__ import annotations

"""
Gradio front-end for the Guardrail Composability Explorer.

This module is intentionally self-contained and stable:

- It accepts a RunBundle via text area or JSON upload.
- It validates via the pydantic RunBundle model.
- It computes a Verdict using the configured backend (cc-framework if present,
  otherwise the local fallback implementation).
- It renders a readable HTML verdict chip, Markdown panels, and downloadable
  JSON/PDF artifacts.

`app.py` should only act as a thin entrypoint that calls `build_interface()`.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
from pydantic import ValidationError

from ..core.cc_surface.api import RunBundle, compute_verdict
from ..exporters.one_pager import verdict_to_json, verdict_to_pdf

# ---------------------------------------------------------------------------
# Presentation constants
# ---------------------------------------------------------------------------

_LABEL_COLORS: Dict[str, str] = {
    "Constructive": "#00A572",  # green-ish
    "Independent": "#F3B61F",   # amber
    "Destructive": "#D7263D",   # red
}

# Repo root: .../gce/
_ROOT_DIR = Path(__file__).resolve().parents[3]
_SAMPLES_DIR = _ROOT_DIR / "examples"
_SAMPLE_PATH = _SAMPLES_DIR / "sample_run_bundle.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_bundle_text(text: str, upload: Optional[Any]) -> str:
    """
    Choose a JSON payload source in the following priority:

    1. Explicit text from the Textbox.
    2. Uploaded file (if provided).
    3. Built-in sample_run_bundle.json (if present).
    4. Otherwise, raise a Gradio error.
    """
    candidate = (text or "").strip()

    # 1) Direct text input wins if present
    if candidate:
        return candidate

    # 2) Try the uploaded file (gr.File or dict-like)
    if upload is not None:
        upload_path: Optional[str] = None
        if isinstance(upload, dict):
            # Gradio FileData dict (newer versions)
            upload_path = upload.get("name") or upload.get("path")
        elif hasattr(upload, "name"):
            # e.g., tempfile.NamedTemporaryFile or similar
            upload_path = getattr(upload, "name")

        if upload_path:
            try:
                return Path(upload_path).read_text(encoding="utf-8")
            except OSError as exc:  # file vanished / permissions / etc.
                raise gr.Error(f"Failed to read uploaded file: {exc}") from exc

    # 3) Fallback to local sample file, if shipped with the package
    if _SAMPLE_PATH.exists():
        try:
            return _SAMPLE_PATH.read_text(encoding="utf-8")
        except OSError:
            # If even this fails, fall through to error
            pass

    # 4) Nothing to work with
    raise gr.Error(
        "No run bundle provided.\n\n"
        "Please either:\n"
        "  • Paste JSON into the text area, or\n"
        "  • Upload a JSON file, or\n"
        "  • Ensure examples/sample_run_bundle.json is present in the repo."
    )


def _format_label_chip(label: str, cc: float) -> str:
    """
    Render the verdict label + CC as a small HTML chip.

    This is pure presentation: the backend only cares about the raw values.
    """
    color = _LABEL_COLORS.get(label, "#4A4A4A")
    label_html = (
        f"<span style='padding:0.35rem 0.75rem;border-radius:999px;"
        f"background:{color};color:white;font-weight:600;'>"
        f"{label}</span>"
    )
    cc_html = f"<span style='font-family:monospace;'>CC={cc:.3f}</span>"
    return (
        "<div style='display:inline-flex;align-items:center;gap:0.5rem;'>"
        f"{label_html}{cc_html}"
        "</div>"
    )


def _list_to_md(items: list[str], header: str) -> str:
    """
    Convert a list of strings to a Markdown section with bullets.
    """
    if not items:
        return f"### {header}\n_No data_"
    bullets = "\n".join(f"- {item}" for item in items)
    return f"### {header}\n{bullets}"


def _compute(bundle_text: str, upload: Optional[Any]) -> Tuple[str, str, str, str, str, str]:
    """
    Core Gradio callback:

    - Resolve bundle JSON from text/upload/sample.
    - Validate into a RunBundle.
    - Compute Verdict through the cc-surface API.
    - Return:
        1) HTML verdict chip,
        2) Recommendation text,
        3) Next Tests markdown,
        4) Checklist markdown,
        5) *Path* to JSON file (for download),
        6) *Path* to PDF (for download).
    """
    try:
        payload_text = _load_bundle_text(bundle_text, upload)
        payload = json.loads(payload_text)
        bundle = RunBundle.model_validate(payload)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Invalid JSON in run bundle: {exc}") from exc
    except ValidationError as exc:
        raise gr.Error(f"RunBundle schema validation failed: {exc}") from exc

    verdict = compute_verdict(bundle)

    label_chip_html = _format_label_chip(verdict.label, verdict.CC)
    recommendation = verdict.recommendation
    next_tests_md = _list_to_md(verdict.next_tests, "Next Tests")
    checklist_md = _list_to_md(verdict.checklist, "Checklist")

    # Include bundle in the JSON export for context.
    json_blob = verdict_to_json(verdict, bundle=bundle)

    # IMPORTANT: DownloadButton expects a *file*, not a raw JSON string.
    # So we persist to a temp file and return its path.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(json_blob.encode("utf-8"))
    tmp.flush()
    tmp.close()
    json_path = tmp.name

    # PDF path is already a real file path from verdict_to_pdf.
    pdf_path = verdict_to_pdf(verdict)

    return (
        label_chip_html,
        recommendation,
        next_tests_md,
        checklist_md,
        json_path,
        str(pdf_path),
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    """
    Construct the Gradio Blocks app but do not launch it.

    This makes the UI re-usable from:
    - gce.ui.app (thin entrypoint)
    - tests or ad-hoc scripts that want an object handle.
    """
    default_text = ""
    if _SAMPLE_PATH.exists():
        try:
            default_text = _SAMPLE_PATH.read_text(encoding="utf-8")
        except OSError:
            default_text = ""

    with gr.Blocks(title="Guardrail Composability Explorer") as demo:
        gr.Markdown(
            """
            # Guardrail Composability Explorer

            Paste or upload a CC run bundle JSON, then compute the composability
            verdict, recommendation, and exportable artifacts.
            """
        )

        with gr.Row():
            bundle_text = gr.Textbox(
                label="Run bundle JSON",
                lines=18,
                value=default_text,
                show_copy_button=True,
            )
            bundle_file = gr.File(
                label="Upload run bundle (.json)",
                file_types=[".json"],
            )

        compute_btn = gr.Button("Compute verdict", variant="primary")

        verdict_chip = gr.HTML(label="Verdict")
        rec_output = gr.Textbox(
            label="Recommendation",
            interactive=False,
            lines=4,
        )
        next_tests = gr.Markdown(label="Next Tests")
        checklist = gr.Markdown(label="Checklist")

        # Download buttons: used as *outputs*; the callback returns file paths.
        download_json = gr.DownloadButton("Export JSON")
        download_pdf = gr.DownloadButton("Download PDF")

        compute_btn.click(
            _compute,
            inputs=[bundle_text, bundle_file],
            outputs=[
                verdict_chip,
                rec_output,
                next_tests,
                checklist,
                download_json,
                download_pdf,
            ],
        )

    return demo


def launch(**kwargs: Any) -> None:
    """
    Convenience launcher. Safe to use from __main__ or external callers:

        python -m gce.ui.app
        python -m gce.ui.gradio_app
    """
    iface = build_interface()
    iface.launch(**kwargs)


if __name__ == "__main__":  # pragma: no cover - manual launch
    launch()
