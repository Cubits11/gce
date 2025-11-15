from __future__ import annotations
from gce.core.cc_surface.api import backend_info, compute_verdict_from_params

def print_backend_info() -> None:
    info = backend_info()
    for k, v in info.items():
        print(f"{k}: {v}")

def cc_quickcheck() -> None:
    # Tiny smoke: two singletons + composed, objective=minimize (smaller J is better)
    verdict = compute_verdict_from_params(
        theta=0.5,
        patterns=["demo"],
        rule="SEQUENTIAL(DFA→RR)",
        J_baselines={"A": 0.30, "B": 0.40},
        J_composed=0.28,
        objective="minimize",
    )
    print(verdict.model_dump_json(indent=2))
