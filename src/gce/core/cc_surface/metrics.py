from __future__ import annotations

from typing import Sequence, Union

import numpy as np

ArrayLike = Union[float, Sequence[float], np.ndarray]

__all__ = ["youden_j"]


def youden_j(tpr: ArrayLike, fpr: ArrayLike) -> Union[float, np.ndarray]:
    """
    Compute Youden's J statistic: J = TPR − FPR, clipped to [-1, 1].

    Design goals
    ------------
    - Support both scalar and vector inputs:
        * If both inputs are Python scalars ⇒ return a Python float.
        * Otherwise ⇒ return a numpy.ndarray.
    - Allow lists, tuples, numpy arrays (or mixtures).
    - Use numpy broadcasting where possible, but fail fast with a clear error
      if shapes cannot be broadcast.
    - Preserve NaNs and infs (no silent sanitisation), but clip finite values
      into the mathematically valid interval [-1, 1].

    Parameters
    ----------
    tpr:
        True positive rate(s). Typically in [0, 1], but not enforced.
    fpr:
        False positive rate(s). Typically in [0, 1], but not enforced.

    Returns
    -------
    float or numpy.ndarray
        - float if both inputs were Python scalars.
        - numpy.ndarray otherwise, matching the broadcast shape of `tpr` and `fpr`.

    Raises
    ------
    ValueError
        If `tpr` and `fpr` cannot be broadcast to a common shape.
    """
    # Convert to float arrays; this will raise if input is completely non-numeric.
    t = np.asarray(tpr, dtype=float)
    f = np.asarray(fpr, dtype=float)

    # Attempt broadcasting explicitly so we can give a nicer error message
    try:
        t_b, f_b = np.broadcast_arrays(t, f)
    except ValueError as exc:
        raise ValueError(
            f"tpr and fpr shapes are not broadcastable: {t.shape} vs {f.shape}"
        ) from exc

    # Core definition: J = TPR − FPR, clipped to the mathematically valid range [-1, 1].
    # NaNs / infs are preserved by np.clip.
    j = np.clip(t_b - f_b, -1.0, 1.0)

    # If the user gave us two Python scalars, return a plain float for ergonomics.
    if np.isscalar(tpr) and np.isscalar(fpr) and j.shape == ():
        return float(j)

    return j
