from __future__ import annotations

"""
Gradio front-end for the Guardrail Composability Explorer (GCE).

This module is intentionally self-contained and stable:

- Accepts a RunBundle via text area or JSON upload.
- Validates via the pydantic RunBundle model.
- Computes a Verdict using the configured backend
  (cc-framework if present, otherwise the local fallback implementation).
- Renders:
    * a verdict chip (Constructive / Independent / Destructive + CC),
    * a recommendation sentence,
    * a "Next Tests" panel,
    * a "Checklist" panel,
    * an **AI Coach Explanation** panel (LLM-backed when available),
    * JSON + PDF artifacts as downloadable files.

`app.py` is a thin entrypoint that simply calls `build_interface()` and launches.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
from pydantic import ValidationError

from ..core.cc_surface.api import (
    RunBundle,
    Verdict,
    backend_info,
    compute_verdict,
)
from ..exporters.one_pager import verdict_to_json, verdict_to_pdf

# ai_explainer is optional: if it fails to import (e.g. missing openai),
# we transparently fall back to a deterministic offline explainer.
try:  # pragma: no cover - exercised indirectly via UI
    from .. import ai_explainer as _ai_explainer  # type: ignore[import]
except Exception:  # pragma: no cover - safe fallback
    _ai_explainer = None  # type: ignore[assignment]

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
# Helper: input loading
# ---------------------------------------------------------------------------

def _load_bundle_text(text: str, upload: Optional[Any]) -> str:
    """
    Choose a JSON payload source in the following priority:

    1. Explicit text from the Textbox.
    2. Uploaded file (if provided).
    3. Built-in examples/sample_run_bundle.json (if present).
    4. Otherwise, raise a Gradio error.

    This makes the UI resilient in demos: even with no user input, the
    sample bundle allows a one-click "Compute verdict" path.
    """
    candidate = (text or "").strip()

    # 1) Direct text input wins if present.
    if candidate:
        return candidate

    # 2) Try the uploaded file (gr.File or dict-like, depending on Gradio version).
    if upload is not None:
        upload_path: Optional[str] = None
        if isinstance(upload, dict):
            # Gradio's FileData dict (newer versions).
            upload_path = upload.get("name") or upload.get("path")
        elif hasattr(upload, "name"):
            # e.g., tempfile.NamedTemporaryFile or similar.
            upload_path = getattr(upload, "name")

        if upload_path:
            try:
                return Path(upload_path).read_text(encoding="utf-8")
            except OSError as exc:  # file vanished / permissions / etc.
                raise gr.Error(f"Failed to read uploaded file: {exc}") from exc

    # 3) Fallback to local sample file, if shipped with the package.
    if _SAMPLE_PATH.exists():
        try:
            return _SAMPLE_PATH.read_text(encoding="utf-8")
        except OSError:
            # If even this fails, fall through to error.
            pass

    # 4) Nothing to work with.
    raise gr.Error(
        "No run bundle provided.\n\n"
        "Please either:\n"
        "  • Paste JSON into the text area, or\n"
        "  • Upload a JSON file, or\n"
        "  • Ensure examples/sample_run_bundle.json is present in the repo."
    )


# ---------------------------------------------------------------------------
# Helper: deterministic explanation (offline coach)
# ---------------------------------------------------------------------------

def _best_baseline(bundle: RunBundle) -> Tuple[str, float]:
    """
    Compute the best-performing singleton according to the bundle objective.

    This mirrors the logic used in the recommendation engine, but is kept
    local so the UI can generate a concise explanation without importing
    internal helpers.
    """
    if not bundle.J_baselines:
        return "<none>", float("nan")

    items = bundle.J_baselines.items()
    if bundle.objective == "maximize":
        name, val = max(items, key=lambda kv: float(kv[1]))
    else:
        name, val = min(items, key=lambda kv: float(kv[1]))
    return name, float(val)


def _offline_ai_explanation(
    bundle: RunBundle,
    verdict: Verdict,
    *,
    error: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Generate a deterministic, fully offline explanation of the verdict.

    Returns:
        (text, mode_label)

    `mode_label` is a short descriptor that will be displayed alongside the
    explanation (e.g. "offline-template" or "offline-template (LLM error)").
    """
    best_name, best_val = _best_baseline(bundle)
    label = verdict.label
    cc_val = verdict.CC

    baseline_clause = ""
    if bundle.J_baselines:
        baseline_clause = (
            f"The best singleton guardrail in this run was '{best_name}' "
            f"with J ≈ {best_val:.3g}. "
        )

    core_line = (
        f"The composed system using rule '{bundle.rule}' at θ={bundle.theta:.2f} "
        f"achieved J_composed ≈ {bundle.J_composed:.3g}, yielding a "
        f"composability coefficient CC ≈ {cc_val:.2f} and a **{label}** verdict "
        f"for objective='{bundle.objective}'."
    )

    if label == "Constructive":
        verdict_interpretation = (
            "This means the composition is outperforming the best singleton: "
            "you are gaining signal by combining guardrails rather than running "
            "them in isolation."
        )
    elif label == "Destructive":
        verdict_interpretation = (
            "This means the composition is *worse* than the best singleton: "
            "some interaction between guardrails is degrading performance."
        )
    else:  # Independent
        verdict_interpretation = (
            "This means the composition is essentially neutral: performance is "
            "within the expected noise band of the best singleton."
        )

    next_hint = ""
    if verdict.next_tests:
        next_hint = (
            f"\n\nA strong next experiment is:\n- {verdict.next_tests[0]}"
        )

    error_note = ""
    mode_label = "offline-template"
    if error is not None:
        mode_label = "offline-template (LLM error)"
        error_note = (
            "\n\n> Note: The online AI explainer is temporarily unavailable; "
            "this summary was generated by GCE's built-in template logic.\n"
        )

    text = (
        f"{baseline_clause}{core_line} {verdict_interpretation}"
        f"{next_hint}{error_note}"
    )

    return text.strip(), mode_label


# ---------------------------------------------------------------------------
# Helper: AI explainer integration (hybrid coach)
# ---------------------------------------------------------------------------

def _generate_ai_summary(
    bundle: RunBundle,
    verdict: Verdict,
) -> Tuple[str, str]:
    """
    Route to the AI explainer if available, otherwise fall back to the
    deterministic offline explainer.

    Expected contract for `gce.ai_explainer` (if present):

        def explain_verdict(
            bundle: RunBundle,
            verdict: Verdict,
            max_words: int = 220,
        ) -> str | tuple[str, str]:
            ...

    Where the optional second element in the tuple is a mode label such as
    "online-llm" or "offline-fallback". Any exceptions are caught and turned
    into a clean offline explanation.
    """
    if _ai_explainer is not None:
        explain = getattr(_ai_explainer, "explain_verdict", None)
        if callable(explain):
            try:  # pragma: no cover - behaviour exercised via UI, not unit tests
                result = explain(bundle=bundle, verdict=verdict, max_words=240)  # type: ignore[call-arg]
                if isinstance(result, tuple) and result:
                    text = str(result[0])
                    mode = str(result[1]) if len(result) > 1 else "online"
                else:
                    text = str(result)
                    mode = "online"
                if not text.strip():
                    # Defensive: never return an empty explanation.
                    text, mode = _offline_ai_explanation(bundle, verdict)
                return text, mode
            except Exception as exc:
                # LLM or network issues: degrade gracefully to offline template.
                return _offline_ai_explanation(bundle, verdict, error=str(exc))

    # No explainer module or no callable: pure offline mode.
    return _offline_ai_explanation(bundle, verdict)


# ---------------------------------------------------------------------------
# Helper: formatting
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core Gradio callback
# ---------------------------------------------------------------------------

def _compute(
    bundle_text: str,
    upload: Optional[Any],
) -> Tuple[str, str, str, str, str, str, str]:
    """
    Core Gradio callback:

    - Resolve bundle JSON from text/upload/sample.
    - Validate into a RunBundle.
    - Compute Verdict through the cc-surface API.
    - Generate an AI explanation (LLM-backed when available).
    - Materialise JSON + PDF artifacts on disk.

    Returns (in order):
        1) HTML verdict chip,
        2) Recommendation text,
        3) Next Tests markdown,
        4) Checklist markdown,
        5) AI Coach Explanation markdown,
        6) *Path* to JSON file (for DownloadButton),
        7) *Path* to PDF file (for DownloadButton).
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

    # AI coach explanation (hybrid LLM/offline).
    ai_text, ai_mode = _generate_ai_summary(bundle, verdict)
    ai_md = f"**Mode:** `{ai_mode}`\n\n{ai_text}"

    # Include bundle in the JSON export for context.
    json_blob = verdict_to_json(verdict, bundle=bundle)

    # DownloadButton expects a *file*, not a raw JSON string.
    # Persist to a temp file and return its path.
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
        ai_md,
        json_path,
        str(pdf_path),
    )


# ---------------------------------------------------------------------------
# Public interface (Blocks builder + launcher)
# ---------------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    """
    Construct the Gradio Blocks app but do not launch it.

    This makes the UI re-usable from:
    - gce.ui.app (thin entrypoint),
    - tests or ad-hoc scripts that want an object handle,
    - other tools embedding GCE as a component.
    """
    default_text = ""
    if _SAMPLE_PATH.exists():
        try:
            default_text = _SAMPLE_PATH.read_text(encoding="utf-8")
        except OSError:
            default_text = ""

    # Try to fetch backend metadata for a tiny status footer.
    try:
        info = backend_info()
        backend_label = f"Backend: `{info.get('backend', '?')}` ({info.get('provider', '?')})"
    except Exception:  # pragma: no cover - ultra-defensive
        backend_label = ""

    with gr.Blocks(title="Guardrail Composability Explorer") as demo:
        gr.Markdown(
            """
            # Guardrail Composability Explorer

            Paste or upload a **CC run bundle JSON**, and GCE will act as your
            **guardrail composability coach**:
            
            1. Diagnose whether your composition is **Constructive**, **Independent**, or **Destructive**.
            2. Recommend targeted **Next Tests** and a safety **Checklist**.
            3. Produce an **AI Coach Explanation** of what happened and what to do next.
            4. Export a JSON + PDF report for your lab notebook or assignment.
            """
        )

        if backend_label:
            gr.Markdown(f"_{backend_label}_")

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
        ai_explanation = gr.Markdown(
            label="AI Coach Explanation",
            value="Press **Compute verdict** to generate an explanation.",
        )

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
                ai_explanation,
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
