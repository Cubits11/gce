from __future__ import annotations

"""
Thin entrypoint for the Guardrail Composability Explorer UI.

All layout, callbacks, and Gradio wiring live in `gradio_app.py`.
This module just re-exports the interface builder and launcher.

Usage
-----
From the repo root:

    python -m gce.ui.app
"""

from typing import Any

from . import gradio_app as _gradio_app

__all__ = ["build_interface", "launch"]


def build_interface():
    """Return the main Gradio Blocks app defined in `gradio_app`."""
    return _gradio_app.build_interface()


def launch(**kwargs: Any) -> None:
    """
    Launch the Guardrail Composability Explorer UI.

    Example:

        launch(server_name="0.0.0.0", server_port=7860, share=False)
    """
    iface = _gradio_app.build_interface()
    iface.launch(**kwargs)


if __name__ == "__main__":  # pragma: no cover - manual launch only
    launch()
