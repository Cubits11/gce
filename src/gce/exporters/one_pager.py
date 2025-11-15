
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from ..core.cc_surface.api import Verdict


def verdict_to_json(verdict: Verdict, *, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Serialize a :class:`Verdict` with minimal metadata."""

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict.model_dump(),
    }
    if metadata:
        payload["metadata"] = metadata
    return json.dumps(payload, indent=2)


def verdict_to_pdf(
    verdict: Verdict,
    *,
    output_path: Optional[Path] = None,
    title: str = "Guardrail Composability Verdict",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Render a simple one-page PDF summary using ReportLab."""

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.close()
        output_path = Path(tmp.name)
    else:
        output_path = Path(output_path)

    c = canvas.Canvas(str(output_path), pagesize=letter)
    c.setTitle(title)

    width, height = letter
    margin = inch
    y = height - margin

    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, title)
    y -= 0.4 * inch

    c.setFont("Helvetica", 12)
    c.drawString(margin, y, f"Generated: {datetime.now(timezone.utc).isoformat()}")
    y -= 0.3 * inch

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"Label: {verdict.label}")
    y -= 0.25 * inch

    c.setFont("Helvetica", 12)
    c.drawString(margin, y, f"CC = {verdict.CC:.3f}")
    y -= 0.3 * inch

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Recommendation")
    y -= 0.2 * inch
    text_obj = c.beginText(margin, y)
    text_obj.setFont("Helvetica", 11)
    for line in _wrap_text(verdict.recommendation, width - 2 * margin):
        text_obj.textLine(line)
    c.drawText(text_obj)

    y = text_obj.getY() - 0.2 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Next Tests")
    y -= 0.2 * inch
    _draw_bullets(c, verdict.next_tests, y, margin)
    y -= 0.4 * inch + 0.2 * inch * len(verdict.next_tests)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Checklist")
    y -= 0.2 * inch
    _draw_bullets(c, verdict.checklist, y, margin)

    if metadata:
        y = margin
        c.setFont("Helvetica-Oblique", 9)
        for key, value in metadata.items():
            c.drawRightString(width - margin, y, f"{key}: {value}")
            y += 0.15 * inch

    c.showPage()
    c.save()
    return output_path


def _wrap_text(text: str, max_width: float, avg_char_width: float = 6) -> list[str]:
    """Simple word wrap for ReportLab text objects."""

    max_chars = int(max_width / avg_char_width)
    words = text.split()
    lines = []
    current: list[str] = []
    for word in words:
        tentative = " ".join(current + [word]) if current else word
        if len(tentative) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def _draw_bullets(c: canvas.Canvas, items: list[str], start_y: float, margin: float) -> None:
    y = start_y
    c.setFont("Helvetica", 11)
    for item in items:
        c.drawString(margin + 10, y, f"• {item}")
        y -= 0.2 * inch
