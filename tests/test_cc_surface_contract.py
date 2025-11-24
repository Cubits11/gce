from __future__ import annotations

"""
Contract tests for the cc_surface API.

These tests intentionally exercise the *public* surface:

- RunBundle (pydantic model)
- compute_verdict (fallback backend path when cc-framework is absent)

They pin down:
- CC computation for minimize/maximize objectives
- Label classification (Constructive / Destructive)
- Recommendation narrative text
- Next tests + checklist scaffolding
"""

import sys
from pathlib import Path

import pytest

# Ensure local "src" layout wins over any globally installed gce
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gce.core.cc_surface.api import RunBundle, compute_verdict  # type: ignore[import-untyped]


def test_compute_verdict_minimize_constructive_path() -> None:
    """
    When objective='minimize' and the composed J is strictly better than the best
    singleton, CC < 1 and the label should be Constructive with a positive story.
    """
    bundle = RunBundle(
        theta=0.3,
        patterns=["prior", "denoiser"],
        rule="blend",
        J_baselines={"A": 1.0, "B": 1.2},
        J_composed=0.8,
        objective="minimize",
    )

    verdict = compute_verdict(bundle)

    # CC ratio
    assert verdict.CC == pytest.approx(0.8)

    # Label
    assert verdict.label == "Constructive"

    # Recommendation narrative is part of the contract.
    assert (
        verdict.recommendation
        == "Lean into the synergy. Rule 'blend' at θ=0.30 delivers 0.8 "
        "vs singleton 'A'=1 (CC=0.80, objective=minimize). Patterns in play: prior, denoiser."
    )

    # Next tests should suggest exploring θ, ablations, and re-checking the baseline.
    assert verdict.next_tests == [
        "Expand the θ sweep around 0.30 for rule 'blend' to map the constructive window.",
        "Run leave-one-pattern-out ablations for prior, denoiser to verify their individual lifts.",
        "Re-evaluate singleton 'A' to confirm the minimize reference (1).",
    ]

    # Checklist should enforce objective clarity, dataset parity, θ documentation, and instrumentation.
    assert verdict.checklist == [
        "Confirm objective='minimize' aligns with how J is interpreted.",
        "Ensure 2 singleton baselines use the same dataset and evaluation seed as the composition.",
        "Document how θ=0.30 for rule 'blend' was chosen.",
        "Ensure instrumentation exists for patterns: prior, denoiser.",
    ]


def test_compute_verdict_maximize_destructive_path() -> None:
    """
    When objective='maximize' and the composed J is worse than the best singleton,
    CC > 1 and the label should be Destructive with a rollback story.
    """
    bundle = RunBundle(
        theta=0.5,
        patterns=[],
        rule="gated",
        J_baselines={"alpha": 50.0, "beta": 40.0},
        J_composed=30.0,
        objective="maximize",
    )

    verdict = compute_verdict(bundle)

    # best_singleton = 50; CC = 50 / 30 for maximize
    assert verdict.CC == pytest.approx(50.0 / 30.0)

    # Label
    assert verdict.label == "Destructive"

    # Destructive story: revert to singleton, probe lower θ, audit interactions.
    assert (
        verdict.recommendation
        == "Dial back the composition until diagnostics improve. "
        "Rule 'gated' at θ=0.50 delivers 30 vs singleton 'alpha'=50 (CC=1.67, objective=maximize)."
    )

    assert verdict.next_tests == [
        "Re-run singleton 'alpha' (50) as the fallback while disabling rule 'gated'.",
        "Probe lower θ values than 0.50 to find a safer operating point.",
        "Audit the composed pipeline for unexpected interactions, data leakage, or misconfigured guards.",
    ]

    assert verdict.checklist == [
        "Confirm objective='maximize' aligns with how J is interpreted.",
        "Ensure 2 singleton baselines use the same dataset and evaluation seed as the composition.",
        "Document how θ=0.50 for rule 'gated' was chosen.",
        "Record why no pattern diagnostics were supplied.",
    ]
