from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
from pydantic import ValidationError

from ..core.cc_surface.api import RunBundle, compute_verdict
from ..exporters.one_page import verdict_to_json, verdict_to_pdf

_LABEL_COLORS: Dict[str, str] = {
    "Constructive": "#00A572",
    "Independent": "#F3B61F",
    "Destructive": "#D7263D",
}

_SAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
_SAMPLE_PATH = _SAMPLES_DIR / "sample_run_bundle.json"


def _load_bundle_text(text: str, upload: Optional[Any]) -> str:
    candidate = (text or "").strip()
    if not candidate and upload is not None:
        upload_path = None
        if isinstance(upload, dict):
            upload_path = upload.get("name")
        elif hasattr(upload, "name"):
            upload_path = getattr(upload, "name")
        if upload_path:
            candidate = Path(upload_path).read_text(encoding="utf-8")
    if not candidate and _SAMPLE_PATH.exists():
        candidate = _SAMPLE_PATH.read_text(encoding="utf-8")
    if not candidate:
        raise gr.Error("Provide a run bundle via text area, file upload, or keep the sample file present.")
    return candidate


def _format_label(label: str, cc: float) -> str:
    color = _LABEL_COLORS.get(label, "#4A4A4A")
    return (
        f"<div style='display:inline-flex;align-items:center;gap:0.4rem;'>"
        f"<span style='padding:0.35rem 0.75rem;border-radius:999px;background:{color};color:white;font-weight:600;'>"
        f"{label}</span>"
        f"<span style='font-family:monospace;'>CC={cc:.3f}</span>"
        "</div>"
    )


def _list_to_md(items: list[str], header: str) -> str:
    if not items:
        return f"### {header}\n_No data_"
    bullets = "\n".join(f"- {item}" for item in items)
    return f"### {header}\n{bullets}"


def _compute(bundle_text: str, upload: Optional[Any]):
    try:
        payload_text = _load_bundle_text(bundle_text, upload)
        payload = json.loads(payload_text)
        bundle = RunBundle.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise gr.Error(f"Invalid run bundle: {exc}") from exc

    verdict = compute_verdict(bundle)

    label_chip = _format_label(verdict.label, verdict.CC)
    recommendation = verdict.recommendation
    next_tests_md = _list_to_md(verdict.next_tests, "Next Tests")
    checklist_md = _list_to_md(verdict.checklist, "Checklist")
    json_blob = verdict_to_json(verdict)
    pdf_path = verdict_to_pdf(verdict)

    return (
        label_chip,
        recommendation,
        next_tests_md,
        checklist_md,
        json_blob,
        str(pdf_path),
    )


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Guardrail Composability Explorer") as demo:
        gr.Markdown(
            """
            # Guardrail Composability Explorer
            Upload or paste a CC run bundle, then compute the verdict, recommendation, and exports.
            """
        )

        with gr.Row():
            bundle_text = gr.Textbox(
                label="Run bundle JSON",
                lines=18,
                value=_SAMPLE_PATH.read_text(encoding="utf-8") if _SAMPLE_PATH.exists() else "",
            )
            bundle_file = gr.File(label="Upload run bundle", file_types=[".json"])

        compute_btn = gr.Button("Compute verdict", variant="primary")

        verdict_chip = gr.HTML(label="Verdict")
        rec_output = gr.Textbox(label="Recommendation", interactive=False)
        next_tests = gr.Markdown()
        checklist = gr.Markdown()

        download_json = gr.DownloadButton(
            "Export JSON", file_name="gce-verdict.json", variant="secondary"
        )
        download_pdf = gr.DownloadButton(
            "Download PDF", file_name="gce-verdict.pdf", variant="secondary"
        )

        compute_btn.click(
            _compute,
            inputs=[bundle_text, bundle_file],
            outputs=[verdict_chip, rec_output, next_tests, checklist, download_json, download_pdf],
        )

    return demo


def launch(**kwargs: Any) -> None:
    interface = build_interface()
    interface.launch(**kwargs)


if __name__ == "__main__":
    launch()
