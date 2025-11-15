from __future__ import annotations
from typing import Dict, Literal

INDEPENDENT_TOL: float = 0.05  # ±5% band

Objective = Literal["minimize", "maximize"]

def compute_cc(J_baselines: Dict[str, float], J_comp: float, objective: Objective) -> float:
    if not J_baselines:
        return 1.0  # neutral if nothing to compare to
    vals = [float(x) for x in J_baselines.values()]
    best_singleton = min(vals) if objective == "minimize" else max(vals)
    if best_singleton == 0:
        return float("inf") if objective == "minimize" else (0.0 if J_comp == 0 else 0.0)
    return (float(J_comp) / best_singleton) if objective == "minimize" else (best_singleton / float(J_comp))

def classify_cc(cc: float, tol: float = INDEPENDENT_TOL) -> str:
    if cc < 1.0 - tol:
        return "Constructive"
    if cc > 1.0 + tol:
        return "Destructive"
    return "Independent"
