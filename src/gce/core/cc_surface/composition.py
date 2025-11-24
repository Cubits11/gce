from __future__ import annotations

from typing import Mapping, Literal, Optional
import math

# CC close to 1.0 is treated as "Independent" within this band.
INDEPENDENT_TOL: float = 0.05  # ±5%

Objective = Literal["minimize", "maximize"]
CCLabel = Literal["Constructive", "Independent", "Destructive"]


def _best_singleton_value(
    J_baselines: Mapping[str, float],
    objective: Objective,
) -> Optional[float]:
    """
    Return the best singleton J value according to the objective.

    - Ignores non-finite values (NaN, ±inf).
    - Returns None if no usable singleton exists.
    """
    if not J_baselines:
        return None

    vals = []
    for value in J_baselines.values():
        v = float(value)
        if math.isfinite(v):
            vals.append(v)

    if not vals:
        return None

    if objective == "maximize":
        return max(vals)
    return min(vals)


def compute_cc(
    J_baselines: Mapping[str, float],
    J_comp: float,
    objective: Objective,
) -> float:
    """
    Compute the composability coefficient (CC).

    Convention:
    - objective='minimize': smaller J is better
        CC = J_comp / J_best
    - objective='maximize': larger J is better
        CC = J_best / J_comp

    Interpretation:
    - CC < 1  → composition is better than best singleton (Constructive)
    - CC ≈ 1  → composition is neutral (Independent)
    - CC > 1  → composition is worse than best singleton (Destructive)

    Edge cases:
    - No baselines → neutral CC=1.0
    - No finite baselines → CC=NaN
    - Zero denominators:
        * minimize: J_best == 0
            - J_comp == 0 → CC=1.0 (match)
            - else        → CC=+inf (composition strictly worse)
        * maximize: J_comp <= 0
            - J_best <= 0 → CC=1.0 (degenerate but symmetric)
            - else        → CC=+inf (composition collapsed)
    """
    J_best = _best_singleton_value(J_baselines, objective)
    J_c = float(J_comp)

    # No usable baseline: neutral, but clearly marked.
    if J_best is None:
        return float("nan")

    # Minimize: CC = J_comp / J_best
    if objective == "minimize":
        if J_best == 0.0:
            if J_c == 0.0:
                return 1.0
            return float("inf")
        return J_c / J_best

    # Maximize: CC = J_best / J_comp
    # Guard against non-positive composition scores
    if J_c <= 0.0:
        if J_best <= 0.0:
            return 1.0
        return float("inf")

    return J_best / J_c


def classify_cc(cc: float, tol: float = INDEPENDENT_TOL) -> CCLabel:
    """
    Classify a CC value into Constructive / Independent / Destructive.

    Uses a symmetric tolerance band around 1.0:

        CC < 1 - tol  → Constructive
        CC > 1 + tol  → Destructive
        otherwise     → Independent

    Non-finite CC values are treated as Independent to avoid overclaiming.
    """
    if not math.isfinite(cc):
        return "Independent"

    if cc < 1.0 - tol:
        return "Constructive"
    if cc > 1.0 + tol:
        return "Destructive"
    return "Independent"
