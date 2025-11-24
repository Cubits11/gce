from __future__ import annotations

"""
Pydantic models for the public CC surface.

These models are deliberately:

- Strict enough to catch bad inputs early.
- Flexible enough to accept JSON / dict / CLI inputs without being brittle.
- Compatible with the CC-Framework backend (same field names / semantics).

They are used in the fallback (no-cc-framework) backend and as schemas for
CLI / JSON I/O.
"""

from typing import Any, Dict, List, Mapping, cast

import math
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .composition import Objective, CCLabel


class RunBundle(BaseModel):
    """
    Minimal bundle describing a compositional experiment.

    Fields
    ------
    theta:
        Composition knob / scenario parameter (typically in [0, 1], but
        we only require it to be finite so callers can repurpose it).
    patterns:
        Human-readable identifiers for the patterns/guardrails involved.
    rule:
        Label of the composition rule (e.g. "SEQUENTIAL(DFA→RR)").
    J_baselines:
        Singleton J metrics indexed by pattern name.
    J_composed:
        J metric of the composed system.
    objective:
        Direction of optimisation, "minimize" or "maximize".
    """

    model_config = ConfigDict(
        extra="ignore",  # tolerate extra fields from other tools / versions
        frozen=False,
    )

    theta: float = Field(..., description="Composition knob / scenario parameter")
    patterns: List[str] = Field(
        default_factory=list,
        description="Pattern identifiers participating in the composition",
    )
    rule: str = Field(..., min_length=1, description="Composition rule label")
    J_baselines: Dict[str, float] = Field(
        ...,
        description="Singleton J values by name",
    )
    J_composed: float = Field(
        ...,
        description="Composed J value for the rule at this theta",
    )
    objective: Objective = Field(
        "minimize",
        description="Direction of optimisation: 'minimize' or 'maximize'",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("theta")
    @classmethod
    def _theta_ok(cls, v: float) -> float:
        v = float(v)
        if not math.isfinite(v):
            raise ValueError("theta must be a finite float")
        # We do NOT clamp to [0, 1]; callers may use other ranges.
        return v

    @field_validator("patterns")
    @classmethod
    def _patterns_ok(cls, v: List[Any]) -> List[str]:
        # Coerce everything to non-empty, stripped strings and drop blanks.
        cleaned: List[str] = []
        for item in v:
            s = str(item).strip()
            if s:
                cleaned.append(s)
        return cleaned

    @field_validator("rule")
    @classmethod
    def _rule_ok(cls, v: str) -> str:
        v = str(v).strip()
        if not v:
            raise ValueError("rule must be a non-empty string")
        return v

    @field_validator("J_baselines")
    @classmethod
    def _j_map_ok(cls, v: Mapping[Any, Any]) -> Dict[str, float]:
        """
        Ensure all keys are strings and all values are finite floats.
        An empty mapping is allowed (CC then defaults to neutral=1.0).
        """
        result: Dict[str, float] = {}
        for k, val in v.items():
            key = str(k)
            try:
                fv = float(val)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"J_baselines[{key!r}] must be numeric, got {val!r}"
                ) from exc
            if not math.isfinite(fv):
                raise ValueError(
                    f"J_baselines[{key!r}] must be finite, got {fv!r}"
                )
            result[key] = fv
        return result

    @field_validator("J_composed")
    @classmethod
    def _j_ok(cls, v: Any) -> float:
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"J_composed must be numeric, got {v!r}") from exc
        if not math.isfinite(fv):
            raise ValueError(f"J_composed must be finite, got {fv!r}")
        return fv

    @field_validator("objective")
    @classmethod
    def _objective_ok(cls, v: Any) -> Objective:
        """
        Normalise and validate the objective string.

        Accepts case-insensitive inputs like "Minimize", "MAXIMIZE", etc.
        """
        if isinstance(v, str):
            norm = v.strip().lower()
        else:
            norm = str(v).strip().lower()

        if norm not in ("minimize", "maximize"):
            raise ValueError(
                f"objective must be 'minimize' or 'maximize', got {v!r}"
            )
        return cast(Objective, norm)


class Verdict(BaseModel):
    """
    Result of a composability analysis.

    Fields
    ------
    CC:
        Primary composability ratio (>= 0).
    label:
        Qualitative classification: "Constructive", "Independent", "Destructive".
    recommendation:
        Narrative guidance based on the CC and bundle.
    next_tests:
        Suggested follow-up experiments.
    checklist:
        Sanity checks / instrumentation items to verify.
    """

    model_config = ConfigDict(
        extra="ignore",  # tolerate additional fields from upstream backends
        frozen=False,
    )

    CC: float = Field(..., ge=0.0, description="Primary composability ratio (>= 0)")
    label: CCLabel
    recommendation: str = Field(
        "",
        description="Narrative recommendation for how to use this composition",
    )
    next_tests: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up experiments",
    )
    checklist: List[str] = Field(
        default_factory=list,
        description="Checklist of instrumentation / sanity checks",
    )

    @field_validator("CC")
    @classmethod
    def _cc_ok(cls, v: Any) -> float:
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CC must be numeric, got {v!r}") from exc
        if fv < 0 or not math.isfinite(fv):
            raise ValueError(f"CC must be a finite, non-negative float, got {fv!r}")
        return fv

    @field_validator("next_tests", "checklist")
    @classmethod
    def _string_list(cls, v: List[Any]) -> List[str]:
        """
        Normalise next_tests/checklist to clean string lists.
        """
        cleaned: List[str] = []
        for item in v:
            s = str(item).strip()
            if s:
                cleaned.append(s)
        return cleaned
