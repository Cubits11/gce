"""Export helpers for Guardrail Composability Explorer."""

from . import one_pager as one_page
from .one_pager import (
    build_payload,
    export_one_pager,
    render_text_report,
    verdict_to_json,
    verdict_to_pdf,
)

__all__ = [
    "one_page",
    "build_payload",
    "export_one_pager",
    "render_text_report",
    "verdict_to_json",
    "verdict_to_pdf",
]


def __getattr__(name: str):  # pragma: no cover - compatibility shim
    if name == "one_page":
        return one_page
    raise AttributeError(name)
