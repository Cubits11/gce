from __future__ import annotations

"""
Recommendation + checklist logic for the CC surface.

This module is intentionally:
- Opinionated: clear, human-readable guidance.
- Stable: text is deterministic for a given bundle + CC + label.
- Robust: degrades gracefully when baselines or patterns are missing.
"""

from typing import Iterable, List, Tuple

from .composition import CCLabel
from .validators import RunBundle


def _best_baseline(bundle: RunBundle) -> Tuple[str, float]:
    """
    Return the best-performing singleton according to the objective.

    If no baselines are present, we return ("<none>", NaN) and let the
    caller decide how to phrase recommendations.
    """
    if not bundle.J_baselines:
        return "<none>", float("nan")

    items: Iterable[Tuple[str, float]] = bundle.J_baselines.items()
    if bundle.objective == "maximize":
        return max(items, key=lambda item: float(item[1]))
    return min(items, key=lambda item: float(item[1]))


def _tone_for_label(label: CCLabel) -> str:
    """
    Short, leading phrase that sets the tone of the recommendation.
    """
    return {
        "Constructive": "Lean into the synergy.",
        "Independent": "Hold the line — the blend is neutral.",
        "Destructive": "Dial back the composition until diagnostics improve.",
    }[label]


def make_recommendation(bundle: RunBundle, CC: float, label: CCLabel) -> str:
    """
    Produce a single-sentence recommendation tying the numeric CC to the
    experiment context (rule, theta, baselines, patterns).

    The text is designed to be:
    - Interpretable in isolation.
    - Safe when baselines or patterns are missing.
    """
    tone = _tone_for_label(label)
    has_baselines = bool(bundle.J_baselines)
    best_name, best_val = _best_baseline(bundle)

    if has_baselines:
        # Full comparison against best singleton.
        recommendation = (
            f"{tone} Rule '{bundle.rule}' at θ={bundle.theta:.2f} delivers "
            f"{bundle.J_composed:.3g} vs singleton '{best_name}'={best_val:.3g} "
            f"(CC={CC:.2f}, objective={bundle.objective})."
        )
    else:
        # No reference singleton: explain how to interpret CC.
        recommendation = (
            f"{tone} Rule '{bundle.rule}' at θ={bundle.theta:.2f} delivers "
            f"{bundle.J_composed:.3g} with no singleton baselines; "
            f"treat CC={CC:.2f} as relative to a neutral reference "
            f"(objective={bundle.objective})."
        )

    if bundle.patterns:
        recommendation += f" Patterns in play: {', '.join(bundle.patterns)}."

    return recommendation


def make_next_tests(bundle: RunBundle, CC: float, label: CCLabel) -> List[str]:
    """
    Generate concrete follow-up experiments tailored to the verdict.

    Constructive:
        - Explore theta around current point.
        - Ablations over patterns (if present).
        - Sanity-check best singleton reference (if present).

    Destructive:
        - Fall back to best singleton (if present).
        - Probe safer theta region.
        - Audit pipeline for interactions / leakage.

    Independent:
        - Finer theta sweep to confirm neutrality.
        - Validate measurements.
        - Try orthogonal pattern combos for stronger signal.

    When no baselines exist, tests pivot toward establishing a reference.
    """
    tests: List[str] = []
    has_baselines = bool(bundle.J_baselines)
    best_name, best_val = _best_baseline(bundle)
    pat_str = ", ".join(bundle.patterns)

    if label == "Constructive":
        tests.append(
            f"Expand the θ sweep around {bundle.theta:.2f} for rule '{bundle.rule}' "
            "to map the constructive window."
        )

        if bundle.patterns:
            tests.append(
                f"Run leave-one-pattern-out ablations for {pat_str} to verify their individual lifts."
            )
        else:
            tests.append(
                "Introduce diagnostic ablations for each component before locking the policy."
            )

        if has_baselines:
            tests.append(
                f"Re-evaluate singleton '{best_name}' to confirm the {bundle.objective} "
                f"reference ({best_val:.3g})."
            )
        else:
            tests.append(
                "Establish at least one singleton baseline on the same dataset to quantify the lift."
            )

    elif label == "Destructive":
        if has_baselines:
            tests.append(
                f"Re-run singleton '{best_name}' ({best_val:.3g}) as the fallback while "
                f"disabling rule '{bundle.rule}'."
            )
        else:
            tests.append(
                "Define and measure a reference singleton baseline to serve as a safe fallback."
            )

        tests.append(
            f"Probe lower θ values than {bundle.theta:.2f} to find a safer operating point."
        )
        tests.append(
            "Audit the composed pipeline for unexpected interactions, data leakage, or misconfigured guards."
        )

    else:  # Independent
        tests.append(
            f"Perform a finer θ sweep around {bundle.theta:.2f} to confirm neutral behavior."
        )

        if has_baselines:
            tests.append(
                f"Validate measurements for singleton '{best_name}' ({best_val:.3g}) "
                "to ensure the comparison is trustworthy."
            )
        else:
            tests.append(
                "Measure at least one singleton baseline to anchor the neutrality judgment."
            )

        tests.append(
            "Try orthogonal pattern combinations or alternative rules to search for stronger signals."
        )

    return tests


def make_checklist(bundle: RunBundle) -> List[str]:
    """
    Generate a short checklist of sanity / instrumentation items that should
    be true for the verdict to be trustworthy.
    """
    count = len(bundle.J_baselines)
    checklist: List[str] = [
        f"Confirm objective='{bundle.objective}' aligns with how J is interpreted.",
    ]

    if count > 0:
        checklist.append(
            f"Ensure {count} singleton baselines use the same dataset and evaluation seed as the composition."
        )
    else:
        checklist.append(
            "Record and compute at least one singleton baseline on the same dataset as the composition."
        )

    checklist.append(
        f"Document how θ={bundle.theta:.2f} for rule '{bundle.rule}' was chosen."
    )

    if bundle.patterns:
        checklist.append(
            f"Ensure instrumentation exists for patterns: {', '.join(bundle.patterns)}."
        )
    else:
        checklist.append("Record why no pattern diagnostics were supplied.")

    return checklist
