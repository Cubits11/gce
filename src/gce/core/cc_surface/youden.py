from __future__ import annotations

from typing import Sequence, Union

import numpy as np

# Public numeric input type: scalars, Python sequences, or numpy arrays.
ArrayLike = Union[float, Sequence[float], np.ndarray]
# Public return type: either a scalar float or an ndarray, depending on input.
ReturnType = Union[float, np.ndarray]

__all__ = ["ArrayLike", "youden_j"]


def _to_float_array(x: ArrayLike, name: str) -> np.ndarray:
    """
    Convert input to a NumPy float array with a clear error message on failure.

    Parameters
    ----------
    x:
        Input value(s) to convert.
    name:
        Logical name of the argument (e.g. "tpr", "fpr") for error messages.

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
