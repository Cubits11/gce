from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence, Tuple, Union

import numpy as np

# Public numeric input type: scalars, Python sequences, or numpy arrays.
ArrayLike = Union[float, Sequence[float], np.ndarray]
# Public return type: either a scalar float or an ndarray, depending on input.
ReturnType = Union[float, np.ndarray]

__all__ = [
    "ArrayLike",
    "ReturnType",
    "youden_j",
    "YoudenCurve",
    "compute_youden_curve",
    "optimal_youden_threshold",
    "compute_cc_max",
]


# ---------------------------------------------------------------------------
# Low-level helper: robust conversion to float arrays
# ---------------------------------------------------------------------------


def _to_float_array(x: ArrayLike, name: str) -> np.ndarray:
    """
    Convert input to a NumPy float array with a clear error message on failure.

    Parameters
    ----------
    x:
        Input value(s) to convert.
    name:
        Logical name of the argument (e.g. "tpr", "fpr", "scores_w0") for
        error messages.

    Returns
    -------
    numpy.ndarray
        Array of dtype float64.

    Raises
    ------
    TypeError
        If the input cannot be converted to a float array.
    """
    try:
        return np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be convertible to a float array of numbers; "
            f"received object of type {type(x)!r}."
        ) from exc


# ---------------------------------------------------------------------------
# Youden's J statistic (scalar / vector)
# ---------------------------------------------------------------------------


def youden_j(tpr: ArrayLike, fpr: ArrayLike) -> ReturnType:
    r"""
    Compute Youden's J statistic, :math:`J = \mathrm{TPR} - \mathrm{FPR}`.

    This is the basic scalar/vector form of the Youden index used throughout
    the GCE metrics stack. It supports broadcasting and mixed input types,
    and enforces the mathematically valid range for *finite* values.

    Design goals
    ------------
    - **Flexible inputs**:
        - Accept Python scalars, lists/tuples, or numpy arrays (or mixtures).
        - Use NumPy broadcasting so shapes like ``(n,)`` and ``(1,)`` work
          naturally.
    - **Ergonomic return types**:
        - If *both* inputs are Python scalars ⇒ return a Python ``float``.
        - Otherwise ⇒ return a ``numpy.ndarray`` with the broadcast shape.
    - **Numerical sanity**:
        - Finite values of :math:`J` are clipped into the interval ``[-1, 1]``.
        - Non-finite values (``NaN``, ``+/-inf``) are preserved and *not*
          silently clipped or sanitized.
    - **Clear failure modes**:
        - If shapes are not broadcastable, raise a ``ValueError`` describing
          the mismatch.
        - If inputs are not numeric, raise a ``TypeError`` describing which
          argument failed.

    Parameters
    ----------
    tpr : ArrayLike
        True positive rate(s). Semantically in ``[0, 1]``, but this function
        does **not** enforce that range; callers are responsible for upstream
        validation.
    fpr : ArrayLike
        False positive rate(s). Semantically in ``[0, 1]``, likewise not
        enforced here.

    Returns
    -------
    float or numpy.ndarray
        - ``float`` if **both** inputs were Python scalars.
        - ``numpy.ndarray`` otherwise, with the broadcast shape of
          ``tpr`` and ``fpr``.

    Raises
    ------
    TypeError
        If either of ``tpr`` or ``fpr`` cannot be converted to a numeric
        float array.
    ValueError
        If ``tpr`` and ``fpr`` cannot be broadcast to a common shape.

    Notes
    -----
    - For inputs that are already in the valid probability range
      (``tpr, fpr ∈ [0, 1]``), the raw difference ``tpr - fpr`` is
      automatically in ``[-1, 1]``, so clipping only protects against
      numerical noise or upstream misuse.
    - Non-finite values are preserved. For example, if
      ``tpr - fpr == np.inf``, the result will be ``np.inf``, not ``1.0``.

    Examples
    --------
    Scalar inputs:

    >>> youden_j(0.9, 0.1)
    0.8

    Vector inputs:

    >>> youden_j([0.8, 0.9], [0.2, 0.1])
    array([0.6, 0.8])

    Broadcasting:

    >>> youden_j([0.8, 0.9], 0.2)
    array([0.6, 0.7])

    Handling NaNs and infs:

    >>> import numpy as np
    >>> youden_j(np.array([0.9, np.nan, np.inf]), np.array([0.1, 0.2, 0.3]))
    array([0.8,  nan,  inf])
    """
    # Convert inputs to float arrays with consistent dtype and good error messages.
    t = _to_float_array(tpr, "tpr")
    f = _to_float_array(fpr, "fpr")

    # Attempt broadcasting explicitly so we can provide a clear error message.
    try:
        t_b, f_b = np.broadcast_arrays(t, f)
    except ValueError as exc:
        raise ValueError(
            f"tpr and fpr shapes are not broadcastable: {t.shape} vs {f.shape}"
        ) from exc

    # Core definition: raw J = TPR − FPR.
    raw = t_b - f_b

    # For *finite* values, clip to [-1, 1].
    # For NaN / +/-inf, preserve the original value.
    finite_mask = np.isfinite(raw)
    clipped = np.clip(raw, -1.0, 1.0)
    j = np.where(finite_mask, clipped, raw)

    # If the user gave two Python scalars, return a plain float for ergonomics.
    # Note: j is a 0-d array in this case; float(j) extracts the scalar.
    if np.isscalar(tpr) and np.isscalar(fpr) and j.shape == ():
        return float(j)

    return j


# ---------------------------------------------------------------------------
# Threshold scanning: Youden curve from W0 / W1 scores
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YoudenCurve:
    """
    Summary of Youden's J over a threshold scan.

    Attributes
    ----------
    thresholds : numpy.ndarray
        The thresholds evaluated (monotonically increasing).
    tpr : numpy.ndarray
        True positive rate for each threshold: P(decide=1 | W1).
    fpr : numpy.ndarray
        False positive rate for each threshold: P(decide=1 | W0).
    j : numpy.ndarray
        Youden's J = TPR - FPR at each threshold.
    """

    thresholds: np.ndarray
    tpr: np.ndarray
    fpr: np.ndarray
    j: np.ndarray


def _validate_scores(scores_w0: ArrayLike, scores_w1: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal helper to coerce and validate W0/W1 score arrays.

    - Converts to 1-D float arrays.
    - Requires at least one sample in each world.
    - Requires all scores to be finite (no NaNs or infs); those would make the
      ROC / J computations ill-defined and should be handled upstream.
    """
    s0 = _to_float_array(scores_w0, "scores_w0").ravel()
    s1 = _to_float_array(scores_w1, "scores_w1").ravel()

    if s0.size == 0 or s1.size == 0:
        raise ValueError(
            f"scores_w0 and scores_w1 must both be non-empty; "
            f"got shapes {s0.shape} and {s1.shape}."
        )

    if not np.all(np.isfinite(s0)):
        raise ValueError("scores_w0 contains non-finite values; please clean or filter upstream.")
    if not np.all(np.isfinite(s1)):
        raise ValueError("scores_w1 contains non-finite values; please clean or filter upstream.")

    return s0, s1


def compute_youden_curve(
    scores_w0: ArrayLike,
    scores_w1: ArrayLike,
    *,
    thresholds: ArrayLike | None = None,
    higher_scores_leakier: bool = True,
) -> YoudenCurve:
    """
    Compute TPR, FPR, and J across a grid of thresholds from W0/W1 scores.

    Parameters
    ----------
    scores_w0 : ArrayLike
        Model or guardrail scores for world W0 (secret absent / baseline).
    scores_w1 : ArrayLike
        Model or guardrail scores for world W1 (secret present / attack).
    thresholds : ArrayLike, optional
        Explicit thresholds at which to evaluate TPR/FPR/J. If omitted,
        the thresholds are chosen as the sorted unique union of all scores
        from W0 and W1.
    higher_scores_leakier : bool, default True
        If True, the decision rule is ``decide_leak = (score >= threshold)``.
        If False, use ``decide_leak = (score <= threshold)`` instead. This
        lets you adapt to models where *lower* scores correspond to more
        suspicious / leaky behavior.

    Returns
    -------
    YoudenCurve
        Dataclass containing thresholds, TPR, FPR, and J arrays (all 1-D).

    Raises
    ------
    ValueError
        If either score array is empty or contains non-finite values.

    Notes
    -----
    - This function makes no claim about *optimality*; it just evaluates
      Youden's J on a grid of thresholds. See ``optimal_youden_threshold``
      for a convenience wrapper that finds the best threshold.
    - Using the full unique score set as thresholds is reasonable for many
      experiments, but for very large logs you may wish to downsample or
      provide a custom threshold grid for performance.
    """
    s0, s1 = _validate_scores(scores_w0, scores_w1)

    if thresholds is None:
        thresholds_arr = np.unique(np.concatenate([s0, s1]))
    else:
        thresholds_arr = _to_float_array(thresholds, "thresholds").ravel()
        thresholds_arr = np.unique(thresholds_arr)

    if thresholds_arr.size == 0:
        raise ValueError("No thresholds to evaluate: 'thresholds' resolved to an empty array.")

    # Shape: (n_thresholds, n_samples)
    thr = thresholds_arr[:, None]

    if higher_scores_leakier:
        # Decide "leak" when score >= threshold
        decisions_w0 = s0[None, :] >= thr
        decisions_w1 = s1[None, :] >= thr
    else:
        # Decide "leak" when score <= threshold
        decisions_w0 = s0[None, :] <= thr
        decisions_w1 = s1[None, :] <= thr

    # TPR = P(decide=1 | W1), FPR = P(decide=1 | W0)
    tpr = decisions_w1.mean(axis=1)
    fpr = decisions_w0.mean(axis=1)

    j = youden_j(tpr, fpr)  # vectorised call; returns ndarray

    return YoudenCurve(
        thresholds=thresholds_arr,
        tpr=tpr,
        fpr=fpr,
        j=j,
    )


def optimal_youden_threshold(
    scores_w0: ArrayLike,
    scores_w1: ArrayLike,
    *,
    thresholds: ArrayLike | None = None,
    higher_scores_leakier: bool = True,
) -> Tuple[float, float, YoudenCurve]:
    """
    Find the threshold that maximizes Youden's J, along with the full curve.

    Parameters
    ----------
    scores_w0 : ArrayLike
        Scores for W0 (secret absent).
    scores_w1 : ArrayLike
        Scores for W1 (secret present).
    thresholds : ArrayLike, optional
        Threshold grid to scan. If None, the union of unique scores is used.
    higher_scores_leakier : bool, default True
        See ``compute_youden_curve``.

    Returns
    -------
    best_j : float
        Maximum Youden's J (clipped to [-1, 1]) over the threshold grid.
    best_threshold : float
        Threshold at which J is maximized. If multiple thresholds share the
        same maximum J, the smallest such threshold is returned.
    curve : YoudenCurve
        The full curve object (thresholds, TPR, FPR, J).

    Raises
    ------
    ValueError
        If score validation fails (empty arrays, non-finite values, etc.).

    Notes
    -----
    - This function does a simple argmax over the scanned thresholds; it does
      not perform any smoothing or regularization. For most experiment sizes,
      this is more than adequate.
    """
    curve = compute_youden_curve(
        scores_w0,
        scores_w1,
        thresholds=thresholds,
        higher_scores_leakier=higher_scores_leakier,
    )

    # np.nanargmax will raise if all entries are NaN; that would indicate a
    # deeper data issue (e.g. J wasn't computed correctly). We let that error
    # surface rather than silently guessing.
    idx = int(np.nanargmax(curve.j))
    best_j = float(curve.j[idx])
    best_threshold = float(curve.thresholds[idx])

    return best_j, best_threshold, curve


# ---------------------------------------------------------------------------
# Composability Coefficient (CC_max) with safe edge-case handling
# ---------------------------------------------------------------------------


def compute_cc_max(
    j_observed: float,
    j_dfa: float,
    j_dp: float,
    *,
    eps: float = 1e-12,
    zero_denom_policy: str = "independent",
) -> float:
    """
    Compute the Composability Coefficient

        CC_max = J_observed / max(J_DFA, J_DP)

    with explicit handling for the edge case ``max(J_DFA, J_DP) ≈ 0``.

    Parameters
    ----------
    j_observed : float
        Observed Youden's J for the *composed* guardrail system.
    j_dfa : float
        Youden's J for the formal / DFA-based guardrail alone (same traffic,
        same attacker model).
    j_dp : float
        Youden's J for the DP-based guardrail alone.
    eps : float, default 1e-12
        Threshold below which the denominator is considered effectively zero.
        If ``max(j_dfa, j_dp) <= eps``, the behavior is controlled by
        ``zero_denom_policy``.
    zero_denom_policy : {"independent", "nan", "zero"}, default "independent"
        Policy for handling the ``0 / 0`` or near-zero denominator case:

        - "independent"  ⇒ return 1.0, meaning "no additional composition
          penalty" / effectively independent.
        - "nan"          ⇒ return ``float("nan")``.
        - "zero"         ⇒ return 0.0.

    Returns
    -------
    float
        The composability coefficient CC_max.

    Raises
    ------
    ValueError
        If ``zero_denom_policy`` is not one of the supported options.

    Notes
    -----
    - Under the base spec, if both individual guardrails leak essentially
      nothing (J≈0) but the composed system has some non-zero leak, the notion
      of "relative blow-up" is ill-defined: any positive J_observed divided by
      ~0 would be huge. By *convention*, this function treats such cases as
      CC_max = 1.0 (independent) under the default "independent" policy.
      This is conservative from a safety-engineering perspective.
    - If you prefer to explicitly surface this ambiguity, you can set
      ``zero_denom_policy="nan"`` in the caller and handle NaNs downstream.
    """
    denom = max(float(j_dfa), float(j_dp))

    if denom <= eps:
        policy = zero_denom_policy.lower()
        if policy == "independent":
            return 1.0
        if policy == "nan":
            return float("nan")
        if policy == "zero":
            return 0.0
        raise ValueError(
            "Unsupported zero_denom_policy={!r}. Expected one of "
            "{'independent', 'nan', 'zero'}.".format(zero_denom_policy)
        )

    return float(j_observed) / denom
