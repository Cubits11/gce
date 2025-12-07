from __future__ import annotations

import math
from typing import Mapping, Optional, Literal

# ---------------------------------------------------------------------------
# Public types and constants
# ---------------------------------------------------------------------------

#: Objective for the J metric.
#: - "minimize": smaller J is better (e.g. leakage, risk, cost).
#: - "maximize": larger J is better (e.g. detection power, utility).
Objective = Literal["minimize", "maximize"]

#: Categorical label for composability behaviour.
CCLabel = Literal["Constructive", "Independent", "Destructive"]

#: CC values within ±INDEPENDENT_TOL of 1.0 are treated as "Independent".
INDEPENDENT_TOL: float = 0.05  # ±5%

__all__ = [
    "Objective",
    "CCLabel",
    "INDEPENDENT_TOL",
    "compute_cc",
    "classify_cc",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _best_singleton_value(
    J_baselines: Mapping[str, float],
    objective: Objective,
) -> Optional[float]:
    """
    Return the best singleton J value according to the objective.

    Parameters
    ----------
    J_baselines:
        Mapping from guardrail name → singleton J value. Values may be any
        float; non-finite entries (NaN, ±inf) are ignored.
    objective:
        - ``\"minimize\"`` → smaller J is better.
        - ``\"maximize\"`` → larger J is better.

    Returns
    -------
    float or None
        The best finite singleton value under the objective, or ``None`` if:

        - the mapping is empty, or
        - all entries are non-finite (NaN / ±inf).

    Notes
    -----
    This function is deliberately conservative: if we cannot identify at least
    one finite baseline J, downstream logic treats CC as undefined (NaN).
    """
    if not J_baselines:
        return None

    finite_vals = []
    for value in J_baselines.values():
        v = float(value)
        if math.isfinite(v):
            finite_vals.append(v)

    if not finite_vals:
        return None

    if objective == "maximize":
        return max(finite_vals)
    return min(finite_vals)


# ---------------------------------------------------------------------------
# Core composability coefficient
# ---------------------------------------------------------------------------


def compute_cc(
    J_baselines: Mapping[str, float],
    J_comp: float,
    objective: Objective,
) -> float:
    """
    Compute the composability coefficient (CC) for a composed guardrail.

    Intuition
    ---------
    CC compares the composed guardrail's J-value against the best singleton
    guardrail, under a chosen objective:

    - ``objective='minimize'`` (smaller J is better, e.g. *leakage*):
        ``CC = J_comp / J_best``
    - ``objective='maximize'`` (larger J is better, e.g. *detection*):
        ``CC = J_best / J_comp``

    Interpretation
    --------------
    - ``CC < 1``  → composition is better than best singleton (**Constructive**)
    - ``CC ≈ 1``  → composition is comparable (**Independent**)
    - ``CC > 1``  → composition is worse than best singleton (**Destructive**)

    Edge cases
    ----------
    - **No usable baselines**:
        If there are no baselines, or all baseline J values are non-finite,
        CC is returned as ``NaN``. Classification logic will treat this as
        ``\"Independent\"`` to avoid over-claiming either harm or benefit.

    - **Minimize objective** (smaller is better, e.g. privacy leak):
        ``CC = J_comp / J_best``

        * If ``J_best == 0``:
            - ``J_comp == 0`` → ``CC = 1.0`` (composition matches perfect privacy)
            - otherwise      → ``CC = +inf`` (composition strictly worse)

    - **Maximize objective** (larger is better, e.g. detection power):
        ``CC = J_best / J_comp``

        * If ``J_comp <= 0``:
            - ``J_best <= 0`` → ``CC = 1.0`` (degenerate but symmetric; both bad)
            - otherwise      → ``CC = +inf`` (composition has collapsed)

    Parameters
    ----------
    J_baselines:
        Mapping from guardrail name to its singleton J value.
    J_comp:
        J value of the composed guardrail.
    objective:
        ``\"minimize\"`` or ``\"maximize\"`` (see above).

    Returns
    -------
    float
        The composability coefficient. May be finite, ``NaN``, or ``+inf``
        depending on edge cases. Non-finite outputs are handled gracefully by
        :func:`classify_cc`.
    """
    J_best = _best_singleton_value(J_baselines, objective)
    J_c = float(J_comp)

    # No usable baseline: CC is undefined numerically.
    # Classification will treat NaN as "Independent".
    if J_best is None:
        return float("nan")

    # Minimize: CC = J_comp / J_best
    if objective == "minimize":
        if J_best == 0.0:
            # Best singleton has perfect privacy (J=0).
            # - If composition also has J=0 → neutral (CC=1).
            # - If composition leaks at all → strictly worse (CC=+inf).
            if J_c == 0.0:
                return 1.0
            return float("inf")
        return J_c / J_best

    # Maximize: CC = J_best / J_comp
    # Guard against non-positive composition scores.
    if J_c <= 0.0:
        # If both best singleton and composition are non-positive, we call this
        # a degenerate tie and set CC=1.0 by convention.
        if J_best <= 0.0:
            return 1.0
        # Otherwise, composition has collapsed while baseline has signal; mark
        # as maximally destructive.
        return float("inf")

    return J_best / J_c


# ---------------------------------------------------------------------------
# CC classification
# ---------------------------------------------------------------------------


def classify_cc(cc: float, tol: float = INDEPENDENT_TOL) -> CCLabel:
    """
    Classify a CC value into ``\"Constructive\"``, ``\"Independent\"``, or
    ``\"Destructive\"``.

    Decision rule
    -------------
    Using a symmetric tolerance band around 1.0:

    - ``cc < 1 - tol``  → ``\"Constructive\"``
    - ``cc > 1 + tol``  → ``\"Destructive\"``
    - otherwise         → ``\"Independent\"``

    Non-finite values
    -----------------
    If ``cc`` is NaN or ±inf, this function returns ``\"Independent\"``. The
    rationale is conservative:

    - We avoid claiming strong benefit or harm when the ratio is undefined
      (e.g. no baselines) or formally infinite (e.g. division by ~0).
    - Downstream analyses can still inspect the raw CC value if needed.

    Parameters
    ----------
    cc:
        Composability coefficient.
    tol:
        Half-width of the neutrality band around 1.0. The default of 0.05
        corresponds to treating CC values in ``[0.95, 1.05]`` as neutral.

    Returns
    -------
    CCLabel
        One of ``\"Constructive\"``, ``\"Independent\"``, ``\"Destructive\"``.
    """
    if not math.isfinite(cc):
        return "Independent"

    if cc < 1.0 - tol:
        return "Constructive"
    if cc > 1.0 + tol:
        return "Destructive"
    return "Independent"
