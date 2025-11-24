from __future__ import annotations

"""
One-pager export helpers for Guardrail Composability Explorer (GCE).

This module provides:

- JSON export of a Verdict (with optional RunBundle + metadata)
- Plain-text "one-pager" report
- Simple one-page PDF summary (via ReportLab)
- A tiny `generate_pdf` helper for Gradio demos

Design goals:
- Be robust to missing / partial payloads.
- Fail loudly on truly unsupported payload types.
- Keep outputs stable enough for tests and course submissions.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

from reportlab.lib.pagesizes import letter  # type: ignore[import-untyped]
from reportlab.lib.units import inch  # type: ignore[import-untyped]
from reportlab.pdfgen import canvas  # type: ignore[import-untyped]

from ..core.cc_surface.api import RunBundle, Verdict, compute_verdict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_mapping(payload: Any) -> Dict[str, Any]:
    """
    Convert an arbitrary "payload-like" object into a plain dict.

    Supported inputs:
    - None        → {}
    - Mapping     → dict(payload)
    - Pydantic v2 models (model_dump)
    - Generic objects with a `__dict__` (excluding private attrs)

    Anything else raises TypeError. This keeps JSON export honest instead of
    silently dropping complex objects.
    """
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        # Make a shallow copy so callers can't mutate original objects via the payload.
        return dict(payload)
    if hasattr(payload, "model_dump"):
        # Pydantic v2 style
        return payload.model_dump()  # type: ignore[no-any-return]
    if hasattr(payload, "__dict__"):
        return {
            key: value
            for key, value in payload.__dict__.items()
            if not key.startswith("_")
        }
    raise TypeError(f"Unsupported payload type for normalization: {type(payload)!r}")


def _safe_float(value: Any, *, default: float = float("nan")):  # type: ignore[valid-type]
    """
    Best-effort conversion to float.

    - Returns `default` if conversion fails (TypeError / ValueError).
    - Accepts anything that float(...) would normally accept.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# JSON envelope
# ---------------------------------------------------------------------------


def build_payload(
    bundle: RunBundle | Mapping[str, Any] | None,
    verdict: Verdict,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serialisable payload from a RunBundle + Verdict.

    Structure:
    {
        "generated_at": "<ISO-8601 UTC timestamp>",
        "bundle": {...},            # normalised dict
        "verdict": {...},           # normalised dict
        "metadata": {...}           # (optional)
    }
    """
    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle": _normalize_mapping(bundle) if bundle is not None else {},
        "verdict": _normalize_mapping(verdict),
    }
    if metadata:
        # Make a shallow copy to avoid callers mutating the payload after the fact.
        payload["metadata"] = dict(metadata)
    return payload


def verdict_to_json(
    verdict: Verdict,
    *,
    bundle: RunBundle | Mapping[str, Any] | None = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Serialize a Verdict with optional bundle + metadata to pretty-printed JSON.
    """
    envelope = build_payload(bundle, verdict, metadata=metadata)
    return json.dumps(envelope, indent=2)


# ---------------------------------------------------------------------------
# Plain-text one-pager
# ---------------------------------------------------------------------------


def render_text_report(
    bundle: RunBundle | Mapping[str, Any],
    verdict: Verdict,
    *,
    title: str = "Guardrail One-Pager",
) -> str:
    """
    Render a human-readable one-pager as plain text.

    The report is intentionally minimalist and CLI-friendly:
    - ASCII title underline
    - Rule, θ, objective, label + CC
    - Recommendation paragraph
    - Numbered "Next Tests" and "Checklist" sections
    """
    bundle_payload = _normalize_mapping(bundle)
    verdict_payload = _normalize_mapping(verdict)

    def _list_section(header: str, items: Optional[Sequence[str]]) -> list[str]:
        if not items:
            return [header, "(none)"]
        lines: list[str] = [header]
        for idx, item in enumerate(items, 1):
            lines.append(f"{idx}. {item}")
        return lines

    cc_value = verdict_payload.get("CC")
    cc_display = _safe_float(cc_value)

    lines: list[str] = [title, "=" * len(title)]

    # Rule / θ / objective
    theta_display = bundle_payload.get("theta", "?")
    lines.append(f"Rule: {bundle_payload.get('rule', '?')} (θ={theta_display})")
    lines.append(f"Objective: {bundle_payload.get('objective', '?')}")

    # Label + CC
    lines.append(
        f"Label: {verdict_payload.get('label', '?')} "
        f"(CC={cc_display:.2f})"
    )
    lines.append("")

    # Recommendation
    lines.append("Recommendation")
    lines.append(
        verdict_payload.get("recommendation", "No recommendation available.")
    )
    lines.append("")

    # Next Tests
    lines.extend(_list_section("Next Tests", verdict_payload.get("next_tests")))
    lines.append("")

    # Checklist
    lines.extend(_list_section("Checklist", verdict_payload.get("checklist")))

    # End with a trailing newline (nice for pipes and POSIX tools).
    return "\n".join(lines).strip() + "\n"


def export_one_pager(
    bundle: RunBundle | Mapping[str, Any],
    verdict: Verdict,
    output_path: Path | str,
    *,
    title: str = "Guardrail One-Pager",
) -> Path:
    """
    Write the plain-text one-pager to `output_path` and return the path.

    This is a thin wrapper around `render_text_report` that handles filesystem I/O.
    """
    path = Path(output_path)
    report = render_text_report(bundle, verdict, title=title)
    path.write_text(report, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# PDF one-pager
# ---------------------------------------------------------------------------


def verdict_to_pdf(
    verdict: Verdict,
    *,
    output_path: Optional[Path | str] = None,
    title: str = "Guardrail Composability Verdict",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """
    Render a simple one-page PDF summary using ReportLab.

    Layout (letter, portrait):
    - Title
    - Generated timestamp
    - Label + CC
    - Recommendation (wrapped)
    - Next Tests (bullets)
    - Checklist (bullets)
    - Optional metadata footer (key: value) right-aligned at bottom
    """
    # Resolve output path
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.close()
        path = Path(tmp.name)
    else:
        path = Path(output_path)

    c = canvas.Canvas(str(path), pagesize=letter)
    c.setTitle(title)

    width, height = letter
    margin = inch
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, title)
    y -= 0.4 * inch

    # Timestamp
    c.setFont("Helvetica", 12)
    c.drawString(margin, y, f"Generated: {datetime.now(timezone.utc).isoformat()}")
    y -= 0.3 * inch

    # Label + CC
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"Label: {verdict.label}")
    y -= 0.25 * inch

    c.setFont("Helvetica", 12)
    c.drawString(margin, y, f"CC = {verdict.CC:.3f}")
    y -= 0.3 * inch

    # Recommendation
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Recommendation")
    y -= 0.2 * inch

    text_obj = c.beginText(margin, y)
    text_obj.setFont("Helvetica", 11)
    for line in _wrap_text(verdict.recommendation, width - 2 * margin):
        text_obj.textLine(line)
    c.drawText(text_obj)

    y = text_obj.getY() - 0.2 * inch

    # Next Tests
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Next Tests")
    y -= 0.2 * inch
    y = _draw_bullets(c, list(verdict.next_tests), y, margin, width)

    # Checklist
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Checklist")
    y -= 0.2 * inch
    y = _draw_bullets(c, list(verdict.checklist), y, margin, width)

    # Optional metadata footer
    if metadata:
        footer_y = margin
        c.setFont("Helvetica-Oblique", 9)
        for key, value in metadata.items():
            c.drawRightString(
                width - margin,
                footer_y,
                f"{key}: {value}",
            )
            footer_y += 0.15 * inch

    c.showPage()
    c.save()
    return path


def _wrap_text(text: str, max_width: float, avg_char_width: float = 6.0) -> list[str]:
    """
    Simple word wrap for ReportLab text objects.

    We approximate by:
        max_chars = max_width / avg_char_width

    This does not depend on font metrics but is good enough for short paragraphs.
    """
    if not text:
        return [""]

    max_chars = max(int(max_width / avg_char_width), 1)
    words = text.split()
    lines: list[str] = []
    current: list[str] = []

    for word in words:
        tentative = " ".join(current + [word]) if current else word
        if len(tentative) > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        lines.append(" ".join(current))

    return lines or [""]


def _draw_bullets(
    c: canvas.Canvas,
    items: Sequence[str],
    start_y: float,
    margin: float,
    page_width: float,
) -> float:
    """
    Draw simple bullet-point text, returning the new y-coordinate.

    - Each item is prefixed with "• ".
    - Long items are wrapped using `_wrap_text`.
    - If `items` is empty, we print "(none)".
    """
    y = start_y
    c.setFont("Helvetica", 11)

    if not items:
        c.drawString(margin + 10, y, "(none)")
        return y - 0.2 * inch

    available_width = page_width - (margin + 10) - margin

    for item in items:
        bullet_prefix = "• "
        wrapped_lines = _wrap_text(item, available_width)
        for idx, line in enumerate(wrapped_lines):
            text = bullet_prefix + line if idx == 0 else "  " + line
            c.drawString(margin + 10, y, text)
            y -= 0.2 * inch

    return y


# ---------------------------------------------------------------------------
# Convenience helper for the simple Gradio demo
# ---------------------------------------------------------------------------


def generate_pdf(json_str: str) -> str:
    """
    Convenience helper used by the lightweight Gradio demo.

    Accepts:
        json_str: a RunBundle JSON string.

    Behaviour:
        - Validates the JSON into a RunBundle.
        - Computes the Verdict via the active backend.
        - Writes a temporary PDF via `verdict_to_pdf`.
        - Returns the filesystem path as a string (suitable for gr.File output).
    """
    bundle = RunBundle.model_validate_json(json_str)
    verdict = compute_verdict(bundle)
    pdf_path = verdict_to_pdf(verdict)
    return str(pdf_path)
