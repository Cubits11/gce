from __future__ import annotations

__version__ = "0.1.0"

from .core.cc_surface.api import (
    compute_verdict, analyze_composition, fh_bounds, backend_info, format_verdict
)

__all__ = ["compute_verdict", "analyze_composition", "fh_bounds", "backend_info", "format_verdict"]