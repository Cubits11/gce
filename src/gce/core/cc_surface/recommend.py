from __future__ import annotations

from typing import Iterable, List, Tuple

from .validators import RunBundle


def _best_baseline(bundle: RunBundle) -> Tuple[str, float]:
    """Return the best-performing singleton according to the objective."""

    if not bundle.J_baselines:
        return ("<none>", float("nan"))

    items: Iterable[Tuple[str, float]] = bundle.J_baselines.items()
    if bundle.objective == "maximize":
        return max(items, key=lambda item: float(item[1]))
    return min(items, key=lambda item: float(item[1]))


def make_recommendation(bundle: RunBundle, CC: float, label: str) -> str:
    best_name, best_val = _best_baseline(bundle)
    tone = {
        "Constructive": "Lean into the synergy.",
        "Independent": "Hold the line—the blend is neutral.",
        "Destructive": "Dial back the composition until diagnostics improve.",
    }[label]
    recommendation = (
        f"{tone} Rule '{bundle.rule}' at θ={bundle.theta:.2f} delivers {bundle.J_composed:.3g} "
        f"vs singleton '{best_name}'={best_val:.3g} (CC={CC:.2f}, objective={bundle.objective})."
    )
    if bundle.patterns:
        recommendation += f" Patterns in play: {', '.join(bundle.patterns)}."
    return recommendation


def make_next_tests(bundle: RunBundle, CC: float, label: str) -> List[str]:
    best_name, best_val = _best_baseline(bundle)
    pat_str = ", ".join(bundle.patterns)
    tests: List[str] = []
    if label == "Constructive":
        tests.append(
            f"Expand the θ sweep around {bundle.theta:.2f} for rule '{bundle.rule}' to map the constructive window."
        )
        if bundle.patterns:
            tests.append(
                f"Run leave-one-pattern-out ablations for {pat_str} to verify their individual lifts."
            )
        else:
            tests.append("Introduce diagnostic ablations for each component before locking policy.")
        tests.append(
            f"Re-evaluate singleton '{best_name}' to confirm the {bundle.objective} reference ({best_val:.3g})."
        )
    elif label == "Destructive":
        tests.append(
            f"Re-run singleton '{best_name}' ({best_val:.3g}) as the fallback while disabling rule '{bundle.rule}'."
        )
        tests.append(f"Probe lower θ values than {bundle.theta:.2f} to find a safer operating point.")
        tests.append("Audit the composed pipeline for unexpected interactions or data leakage.")
    else:
        tests.append(
            f"Perform a finer θ sweep around {bundle.theta:.2f} to confirm neutral behavior."
        )
        tests.append(
            f"Validate measurements for singleton '{best_name}' ({best_val:.3g}) to ensure the comparison is trustworthy."
        )
        tests.append("Try orthogonal pattern combinations to search for stronger signals.")
    return tests


def make_checklist(bundle: RunBundle) -> List[str]:
    count = len(bundle.J_baselines)
    checklist = [
        f"Confirm objective='{bundle.objective}' aligns with how J is interpreted.",
        f"Ensure {count} singleton baselines use the same dataset and evaluation seed as the composition.",
        f"Document how θ={bundle.theta:.2f} for rule '{bundle.rule}' was chosen.",
    ]
    if bundle.patterns:
        checklist.append(f"Ensure instrumentation exists for patterns: {', '.join(bundle.patterns)}.")
    else:
        checklist.append("Record why no pattern diagnostics were supplied.")
    return checklist
