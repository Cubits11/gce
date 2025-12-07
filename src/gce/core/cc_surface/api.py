from __future__ import annotations

"""
Public API surface for the GCE compositional-safety backend.

This module is intentionally thin:

- It exposes the *minimum* public surface that the CLI / UI / tests need:
  - `RunBundle`, `Verdict`, `compute_verdict`
  - lightweight helpers like `fh_bounds`, `analyze_composition`,
    `format_verdict`, and `compute_verdict_from_params`
- It dynamically selects between:
  - a local "fallback" backend implemented in `gce.core.cc_surface`, and
  - the optional `cc-framework` backend (`cc.core.api`) when installed.

The goal is that downstream tooling (CLI, UI, notebooks, reports) can treat
this as a stable facade, without needing to care which backend is active.
"""

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Tuple, Literal, cast

from .composition import compute_cc, classify_cc

# Label type used in the fallback backend
Label = Literal["Constructive", "Independent", "Destructive"]

# These globals are intentionally simple strings so they can be surfaced
# directly in `backend_info()` and in the CLI (`gce-backend-info`).
_BACKEND = "fallback"
_BACKEND_PROVIDER = "gce.core.cc_surface"

# ---------------------------------------------------------------------------
# Backend selection: cc-framework (optional) vs local fallback
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised indirectly via tests / real environments
    # Preferred backend: cc-framework (if available in the environment).
    # We use a broad exception catch here deliberately: in real research
    # environments, partial or mis-matched installs are common, and we want
    # a clean fallback rather than import-time crashes.
    from cc.core.api import (  # type: ignore[import-untyped]
        RunBundle as _CCRBundle,
        Verdict as _CCVerdict,
        compute_verdict as _cc_compute_verdict,
    )
except Exception:  # pragma: no cover - default code path in tests
    # Local fallback backend: everything is implemented in this repo.
    from .validators import RunBundle, Verdict
    from .recommend import make_checklist, make_next_tests, make_recommendation

    def compute_verdict(bundle: RunBundle) -> Verdict:
        """
        Fallback verdict computation using local CC logic.

        Steps
        -----
        1. Compute CC from singleton baselines + composed J via `compute_cc`.
        2. Classify CC into "Constructive" / "Independent" / "Destructive".
        3. Generate:
           - a high-level recommendation string,
           - a list of "next tests" for the safety engineer,
           - a checklist of sanity checks performed / to perform.
        """
        CC = compute_cc(bundle.J_baselines, bundle.J_composed, bundle.objective)

        # `classify_cc` returns a string; cast to the Literal type for mypy.
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

else:  # pragma: no cover - path taken when cc-framework is installed
    # Expose cc-framework's models as our public surface.
    RunBundle = _CCRBundle  # type: ignore[misc]
    Verdict = _CCVerdict  # type: ignore[misc]

    def compute_verdict(bundle: RunBundle) -> Verdict:
        """
        Delegate verdict computation to the cc-framework backend.

        This preserves the same high-level semantics as the fallback
        implementation but defers all details (bootstrapping, bounds,
        recommendation logic) to `cc.core.api`.
        """
        return _cc_compute_verdict(bundle)

    _BACKEND = "cc-framework"
    _BACKEND_PROVIDER = "cc.core.api"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _model_dump(obj: Any) -> Dict[str, Any]:
    """
    Normalise various object types into a plain `dict`.

    Supported inputs (in order of preference)
    -----------------------------------------
    1. Pydantic v2 models with `.model_dump()`.
    2. Dataclasses (converted via `dataclasses.asdict`).
    3. Mapping instances (converted via `dict(...)`).
    4. Objects with a `__dict__` attribute (excluding private attributes).

    This intentionally matches how both the fallback `RunBundle` / `Verdict`
    and the cc-framework equivalents expose their data, without tying the API
    to a specific implementation (Pydantic vs dataclass vs hand-rolled).
    """
    # Pydantic v2-style models
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore[no-any-return]

    # Dataclasses
    if is_dataclass(obj):
        return asdict(obj)

    # Generic mapping
    if isinstance(obj, Mapping):
        return dict(obj)

    # Generic Python object
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}

    raise TypeError(f"Cannot serialise object of type {type(obj)!r}")


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------


def backend_info() -> Dict[str, str]:
    """
    Report which backend is active (fallback vs cc-framework).

    Returns
    -------
    dict
        A dictionary with two keys:

        - ``"backend"``: either ``"fallback"`` or ``"cc-framework"``.
        - ``"provider"``: module path for the backend implementation.

    This is intentionally simple so it can be displayed directly in the CLI:

    .. code-block:: bash

        gce-backend-info
        # backend: cc-framework, provider: cc.core.api
    """
    return {"backend": _BACKEND, "provider": _BACKEND_PROVIDER}


def fh_bounds(theta: float, epsilon: float = 0.05) -> Tuple[float, float]:
    """
    Symmetric Fréchet–Hoeffding-style bounds around θ on [0, 1].

    Parameters
    ----------
    theta : float
        Central value (typically the composition knob θ) assumed to lie in
        the probability range [0, 1]. Values outside this range are allowed
        but will be clipped.
    epsilon : float, default 0.05
        Half-width of the confidence band around θ. Only the absolute value
        is used; negative inputs are treated as ``abs(epsilon)``.

    Returns
    -------
    (float, float)
        A pair ``(lower, upper)`` satisfying:

        - lower = max(0.0, theta - epsilon)
        - upper = min(1.0, theta + epsilon)

    Notes
    -----
    This helper is deliberately minimal: its purpose is to provide a simple,
    stable way for the UI and exporters to show “θ ± ε” style bands without
    dragging in the full CC-framework bounds machinery.
    """
    theta_f = float(theta)
    eps = abs(float(epsilon))
    lower = max(0.0, theta_f - eps)
    upper = min(1.0, theta_f + eps)
    return (lower, upper)


def analyze_composition(bundle: RunBundle | Mapping[str, Any]) -> Dict[str, Any]:
    """
    Lightweight helper to compute CC + best singleton from a bundle-like payload.

    This is intended for:
    - UI components that need a quick summary for display.
    - Notebooks that want a "one-line" view over a run bundle.

    Parameters
    ----------
    bundle : RunBundle or Mapping[str, Any]
        Either:
        - a `RunBundle` instance (fallback or cc-framework), or
        - a mapping with keys:
            - ``"theta"``      : float, composition knob (optional, default 0.0)
            - ``"J_baselines"``: mapping of singleton name -> J value (optional)
            - ``"J_composed"`` : float, J value for the composed system
            - ``"objective"``  : ``"minimize"`` or ``"maximize"`` (optional)

    Returns
    -------
    dict
        A plain dictionary with keys:

        - ``"theta"``        : float
        - ``"objective"``    : as taken from the bundle (or ``"minimize"``)
        - ``"best_singleton"``: best J across baselines according to objective,
                                or ``None`` if no baselines are provided.
        - ``"CC"``           : composability coefficient computed by `compute_cc`.

    Notes
    -----
    - This helper **does not** perform any bootstrapping, bounds, or label
      classification; for that, use `compute_verdict`.
    - It deliberately tolerates partial / minimal bundles so it can be used
      in exploratory notebooks and quick UI previews.
    """
    payload = _model_dump(bundle)

    # J_baselines may be missing or None; normalise to a dict.
    J_baselines = payload.get("J_baselines", {}) or {}
    if not isinstance(J_baselines, Mapping):
        raise TypeError(
            f"Expected 'J_baselines' to be a mapping (got {type(J_baselines)!r})."
        )

    J_composed_raw = payload.get("J_composed", 0.0)
    try:
        J_composed = float(J_composed_raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(
            f"Expected 'J_composed' to be numeric (got {J_composed_raw!r})."
        ) from exc

    objective = payload.get("objective", "minimize")

    # Best singleton: min or max across baseline Js according to the user's
    # objective. If nothing is provided, we surface `None` instead of guessing.
    best_singleton: float | None = None
    if J_baselines:
        # Values may be strings / numpy scalars; convert defensively.
        try:
            baseline_values = [float(v) for v in J_baselines.values()]
        except (TypeError, ValueError):  # pragma: no cover - defensive
            baseline_values = []
        if baseline_values:
            selector = min if objective == "minimize" else max
            best_singleton = selector(baseline_values)

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

    Example
    -------
    >>> s = format_verdict(verdict)
    >>> print(s)
    Constructive (CC=0.93): Composition reduces effective leak. Next: test A, test B.

    Behaviour
    ---------
    - If CC is missing or non-numeric, it prints as ``CC=nan``.
    - If `next_tests` is present and non-empty, up to two entries are shown
      as a short preview; otherwise "no follow-ups" is printed.
    """
    payload = _model_dump(verdict)

    # CC value: robust to missing / non-numeric inputs.
    cc_raw = payload.get("CC")
    try:
        if cc_raw is None:
            raise TypeError("CC missing")
        cc_val = float(cc_raw)
    except (TypeError, ValueError):
        cc_val = float("nan")

    label = payload.get("label", "?")

    # Recommendation text: may be missing in some minimal verdicts.
    rec = payload.get("recommendation") or ""
    if not isinstance(rec, str):  # pragma: no cover - defensive
        rec = str(rec)

    # Next tests: tolerate list of strings, list of dicts, etc.
    next_tests_raw = payload.get("next_tests") or []
    tests_preview: str
    try:
        items = list(next_tests_raw)
    except TypeError:  # not iterable
        items = []

    # Render a short preview string.
    if not items:
        tests_preview = "no follow-ups"
    else:
        # Convert arbitrary items to strings, but keep it short.
        tests_preview = ", ".join(str(x) for x in items[:2])

    return f"{label} (CC={cc_val:.2f}): {rec} Next: {tests_preview}."


def compute_verdict_from_params(**params: Any) -> Verdict:
    """
    Convenience helper: build a `RunBundle` from raw params and compute its verdict.

    This is used by:
    - the CLI (`gce run --theta ...`) for quick checks,
    - notebooks where the user wants to inline a small experiment.

    Parameters
    ----------
    **params : Any
        Keyword arguments forwarded into the `RunBundle` constructor. At a
        minimum, these would include:

        - ``theta``      : float
        - ``J_baselines``: mapping of singleton name -> J value
        - ``J_composed`` : float
        - ``objective``  : "minimize" or "maximize"

    Returns
    -------
    Verdict
        A backend-specific verdict object (fallback or cc-framework).

    Notes
    -----
    - This helper deliberately does *not* catch validation errors: if the
      RunBundle schema rejects the params, that is surfaced directly to the
      caller. This keeps errors honest and debuggable.
    """
    bundle = RunBundle(**params)
    return compute_verdict(bundle)
