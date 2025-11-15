from __future__ import annotations
from typing import Dict, List, Literal
from pydantic import BaseModel, Field, field_validator

Objective = Literal["minimize", "maximize"]

class RunBundle(BaseModel):
    theta: float = Field(..., description="composition knob / scenario parameter")
    patterns: List[str] = Field(default_factory=list)
    rule: str = Field(..., description="composition rule label")
    J_baselines: Dict[str, float] = Field(..., description="singleton J values by name")
    J_composed: float = Field(..., description="composed J")
    objective: Objective = "minimize"

    @field_validator("theta")
    @classmethod
    def _theta_ok(cls, v: float) -> float:
        return float(v)

    @field_validator("J_baselines")
    @classmethod
    def _j_map_ok(cls, v: Dict[str, float]) -> Dict[str, float]:
        return {str(k): float(val) for k, val in v.items()}

    @field_validator("J_composed")
    @classmethod
    def _j_ok(cls, v: float) -> float:
        return float(v)

class Verdict(BaseModel):
    CC: float = Field(..., description="primary composability ratio")
    label: Literal["Constructive", "Independent", "Destructive"]
    recommendation: str
    next_tests: List[str]
    checklist: List[str]
