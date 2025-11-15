from __future__ import annotations
from typing import Sequence
import numpy as np

def youden_j(tpr: float | Sequence[float], fpr: float | Sequence[float]):
    t = np.asarray(tpr, dtype=float)
    f = np.asarray(fpr, dtype=float)
    j = np.clip(t - f, -1.0, 1.0)
    if np.isscalar(tpr) and np.isscalar(fpr):
        return float(j)
    return j
