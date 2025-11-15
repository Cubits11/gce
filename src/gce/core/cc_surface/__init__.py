from .api import compute_verdict


def _unavailable(name: str):
    def _inner(*args, **kwargs):  # pragma: no cover - defensive fallback
        raise NotImplementedError(
            f"'{name}' is only available when the cc-framework package is installed."
        )

    return _inner


try:  # pragma: no cover - exercised when cc-framework is present
    from cc.core.api import analyze_composition, fh_bounds, backend_info, format_verdict
except Exception:  # pragma: no cover - default fallback for local use
    analyze_composition = _unavailable("analyze_composition")
    fh_bounds = _unavailable("fh_bounds")
    backend_info = _unavailable("backend_info")
    format_verdict = _unavailable("format_verdict")


__all__ = [
    "compute_verdict",
    "analyze_composition",
    "fh_bounds",
    "backend_info",
    "format_verdict",
]
