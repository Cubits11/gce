from __future__ import annotations

"""
Thin entrypoint for the Guardrail Composability Explorer (GCE) UI.

All layout, callbacks, and Gradio wiring live in `gradio_app.py`.
This module only re-exports the interface builder and launcher so that
you can run:

    python -m gce.ui.app

or import:

    from gce.ui.app import build_interface, launch
"""

from typing import Any, TYPE_CHECKING

from . import gradio_app as _gradio_app

if TYPE_CHECKING:  # pragma: no cover - import-time guard for type checkers
    import gradio as gr

__all__ = ["build_interface", "launch"]


def build_interface() -> "gr.Blocks":
    """
    Construct and return the main Gradio Blocks app defined in `gradio_app`.

    This is the preferred way to embed GCE into other scripts or notebooks.
    """
    return _gradio_app.build_interface()


def launch(**kwargs: Any) -> None:
    """
    Launch the Guardrail Composability Explorer UI.

    Parameters are passed through to `gradio.Blocks.launch`, for example:

        launch(server_name="0.0.0.0", server_port=7860, share=False)
    """
    iface = build_interface()
    iface.launch(**kwargs)


if __name__ == "__main__":  # pragma: no cover - manual launch only
    launch()
