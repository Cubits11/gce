from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple, Literal, cast

from .composition import compute_cc, classify_cc

# Label type used in the fallback backend
Label = Literal["Constructive", "Independent", "Destructive"]

_BACKEND = "fallback"
_BACKEND_PROVIDER = "gce.core.cc_surface"

try:  # pragma: no cover - exercised indirectly via tests
    # cc-framework backend (optional)
    from cc.core.api import (  # type: ignore[import-untyped]
        RunBundle as _CCRBundle,
        Verdict as _CCVerdict,
        compute_verdict as _cc_compute_verdict,
    )
except Exception:  # pragma: no cover - default code path in tests
    from .validators import RunBundle, Verdict
    from .recommend import make_checklist, make_next_tests, make_recommendation

    def compute_verdict(bundle: RunBundle) -> Verdict:
        """
        Fallback verdict computation using local CC logic.

        - Compute CC from baselines + composed J
        - Classify CC into Constructive / Independent / Destructive
        - Generate recommendation, next_tests, checklist
        """
        CC = compute_cc(bundle.J_baselines, bundle.J_composed, bundle.objective)

        # classify_cc returns a string; cast to the Literal type for mypy
        label = cast(Label, classify_cc(CC))

        recommendation = make_recommendation(bundle, CC, label)
        next_tests = make_next_tests(bundle, CC, label)
        checklist = make_checklist(bundle)
        return Verdict(
            CC=CC,
            label=label,
            recommendation=recommendation,
            next_tests=next_tests,
            checklist=checklist,
        )

else:  # pragma: no cover - injected module in tests / cc-framework present
    # Alias external models for our public surface
    RunBundle = _CCRBundle  # type: ignore[misc]
    Verdict = _CCVerdict  # type: ignore[misc]

    def compute_verdict(bundle: RunBundle) -> Verdict:
        """Delegate verdict computation to cc-framework backend."""
        return _cc_compute_verdict(bundle)

    _BACKEND = "cc-framework"
    _BACKEND_PROVIDER = "cc.core.api"


def _model_dump(obj: Any) -> Dict[str, Any]:
    """
    Normalise various object types into a plain dict.

    Supports:
    - Pydantic models with .model_dump()
    - Mapping instances
    - Objects with __dict__ (excluding private attributes)
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore[no-any-return]
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    raise TypeError(f"Cannot serialise object of type {type(obj)!r}")


def backend_info() -> Dict[str, str]:
    """
    Report which backend is active (fallback vs cc-framework).
    """
    return {"backend": _BACKEND, "provider": _BACKEND_PROVIDER}


def fh_bounds(theta: float, epsilon: float = 0.05) -> Tuple[float, float]:
    """
    Simple symmetric Fréchet–Hoeffding-style bounds around θ on [0, 1].

    Returns (theta - eps, theta + eps) clipped to [0, 1].
    """
    theta_f = float(theta)
    eps = abs(float(epsilon))
    return (max(0.0, theta_f - eps), min(1.0, theta_f + eps))


def analyze_composition(bundle: RunBundle | Mapping[str, Any]) -> Dict[str, Any]:
    """
    Lightweight helper to compute CC + best singleton from a bundle-like payload.

    Accepts either:
    - A RunBundle instance
    - A mapping with keys: theta, J_baselines, J_composed, objective
    """
    payload = _model_dump(bundle)
    J_baselines = payload.get("J_baselines", {}) or {}
    J_composed = float(payload.get("J_composed", 0.0))
    objective = payload.get("objective", "minimize")

    best_singleton: float | None = None
    if J_baselines:
        selector = min if objective == "minimize" else max
        best_singleton = selector(J_baselines.values())

    CC = compute_cc(J_baselines, J_composed, objective)

    return {
        "theta": float(payload.get("theta", 0.0)),
        "objective": objective,
        "best_singleton": best_singleton,
        "CC": CC,
    }


def format_verdict(verdict: Verdict | Mapping[str, Any]) -> str:
    """
    Human-friendly one-line summary of a verdict.

    Example:
        "Constructive (CC=0.93): ... Next: test A, test B."
    """
    payload = _model_dump(verdict)

    cc_raw = payload.get("CC")
    try:
        if cc_raw is None:
            raise TypeError("CC missing")
        cc_val = float(cc_raw)
    except (TypeError, ValueError):
        cc_val = float("nan")

    label = payload.get("label", "?")
    rec = payload.get("recommendation", "")
    next_tests = payload.get("next_tests") or []
    tests_preview = ", ".join(list(next_tests)[:2]) if next_tests else "no follow-ups"

    return f"{label} (CC={cc_val:.2f}): {rec} Next: {tests_preview}."


def compute_verdict_from_params(**params: Any) -> Verdict:
    """
    Convenience helper: build a RunBundle from raw params and compute its verdict.

    This is used by the CLI and quick-check tools.
    """
    bundle = RunBundle(**params)
    return compute_verdict(bundle)
