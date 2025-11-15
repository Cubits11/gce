from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .composition import compute_cc, classify_cc

_BACKEND = "fallback"
_BACKEND_PROVIDER = "gce.core.cc_surface"

try:  # pragma: no cover - exercised indirectly via tests
    from cc.core.api import RunBundle as _CCRBundle
    from cc.core.api import Verdict as _CCVerdict
    from cc.core.api import compute_verdict as _cc_compute_verdict
except Exception:  # pragma: no cover - default code path in tests
    from .validators import RunBundle, Verdict
    from .recommend import make_checklist, make_next_tests, make_recommendation

    def compute_verdict(bundle: RunBundle) -> Verdict:
        CC = compute_cc(bundle.J_baselines, bundle.J_composed, bundle.objective)
        label = classify_cc(CC)
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

else:  # pragma: no cover - injected module in tests
    RunBundle = _CCRBundle
    Verdict = _CCVerdict

    def compute_verdict(bundle: RunBundle) -> Verdict:
        return _cc_compute_verdict(bundle)

    _BACKEND = "cc-framework"
    _BACKEND_PROVIDER = "cc.core.api"


def _model_dump(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    raise TypeError(f"Cannot serialise object of type {type(obj)!r}")


def backend_info() -> Dict[str, str]:
    return {"backend": _BACKEND, "provider": _BACKEND_PROVIDER}


def fh_bounds(theta: float, epsilon: float = 0.05) -> Tuple[float, float]:
    theta_f = float(theta)
    eps = abs(float(epsilon))
    return (max(0.0, theta_f - eps), min(1.0, theta_f + eps))


def analyze_composition(bundle: RunBundle | Mapping[str, Any]) -> Dict[str, Any]:
    payload = _model_dump(bundle)
    J_baselines = payload.get("J_baselines", {})
    J_composed = payload.get("J_composed", 0.0)
    objective = payload.get("objective", "minimize")
    best_singleton = None
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
    payload = _model_dump(verdict)
    cc_val = float(payload.get("CC", float("nan")))
    label = payload.get("label", "?")
    rec = payload.get("recommendation", "")
    next_tests = payload.get("next_tests", [])
    tests_preview = ", ".join(next_tests[:2]) if next_tests else "no follow-ups"
    return f"{label} (CC={cc_val:.2f}): {rec} Next: {tests_preview}."


def compute_verdict_from_params(**params: Any) -> Verdict:
    bundle = RunBundle(**params)
    return compute_verdict(bundle)
