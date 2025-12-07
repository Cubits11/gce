"""
Tests for gce.core.cc_surface.youden.youden_j

The goal is to lock in *semantics*, not just implementation details:

- J = TPR - FPR (up to clipping)
- Finite values are clipped to [-1, 1]
- Non-finite values (NaN, +/-inf) are preserved
- Scalars → Python float, otherwise → numpy.ndarray
- Broadcasting follows NumPy rules, with clear errors when shapes mismatch
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gce.core.cc_surface.youden import youden_j


# ---------------------------------------------------------------------------
# Scalar behaviour and return type
# ---------------------------------------------------------------------------


def test_scalar_inputs_return_python_float():
    """Two Python scalars should yield a Python float, not a 0-d ndarray."""
    result = youden_j(0.9, 0.1)
    assert isinstance(result, float)
    assert result == pytest.approx(0.8)


def test_scalar_int_inputs_are_accepted_and_return_float():
    """Integer scalars should be accepted and behave like floats."""
    result = youden_j(1, 0)  # raw = 1, already in-range
    assert isinstance(result, float)
    assert result == pytest.approx(1.0)


def test_scalar_clipping_above_one():
    """
    If tpr - fpr > 1 for finite inputs, the result must be clipped to 1.0.
    """
    # Raw diff = 2.5, but finite values are clipped to 1.0
    result = youden_j(2.0, -0.5)
    assert isinstance(result, float)
    assert result == pytest.approx(1.0)


def test_scalar_clipping_below_minus_one():
    """
    If tpr - fpr < -1 for finite inputs, the result must be clipped to -1.0.
    """
    # Raw diff = -2.0, but finite values are clipped to -1.0
    result = youden_j(-0.5, 1.5)
    assert isinstance(result, float)
    assert result == pytest.approx(-1.0)


def test_scalar_vs_numpy_scalar_returns_ndarray():
    """
    If *either* argument is not a pure Python scalar (e.g. np.float64),
    the function should return an ndarray, not a Python float.
    """
    tpr = np.float64(0.9)
    fpr = 0.1  # pure Python float

    result = youden_j(tpr, fpr)

    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result.item() == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Array behaviour: shape, broadcasting, and dtypes
# ---------------------------------------------------------------------------


def test_array_same_shape_returns_ndarray():
    """Arrays of the same shape should produce an ndarray of that shape."""
    tpr = np.array([0.8, 0.9, 0.95])
    fpr = np.array([0.3, 0.1, 0.05])

    result = youden_j(tpr, fpr)

    assert isinstance(result, np.ndarray)
    assert result.shape == tpr.shape
    expected = tpr - fpr
    expected = np.clip(expected, -1.0, 1.0)
    np.testing.assert_allclose(result, expected)


def test_broadcast_vector_and_scalar():
    """
    When one argument is a 1-D array and the other is a scalar, broadcasting
    should align with NumPy semantics.
    """
    tpr = np.array([0.2, 0.5, 0.9])
    fpr = 0.1

    result = youden_j(tpr, fpr)

    assert isinstance(result, np.ndarray)
    assert result.shape == tpr.shape
    expected = np.clip(tpr - fpr, -1.0, 1.0)
    np.testing.assert_allclose(result, expected)


def test_broadcast_scalar_and_vector():
    """Symmetric case: scalar TPR, vector FPR."""
    tpr = 0.9
    fpr = np.array([0.1, 0.2, 0.3])

    result = youden_j(tpr, fpr)

    assert isinstance(result, np.ndarray)
    assert result.shape == fpr.shape
    expected = np.clip(tpr - fpr, -1.0, 1.0)
    np.testing.assert_allclose(result, expected)


def test_broadcast_2d_and_1d():
    """
    A 2-D array and a 1-D array that broadcast under NumPy rules should be
    handled correctly (e.g., (2, 3) with (3,)).
    """
    tpr = np.array([[0.8, 0.9, 1.1], [0.1, 0.2, 0.3]])
    fpr = np.array([0.2, 0.3, 0.4])

    result = youden_j(tpr, fpr)

    assert isinstance(result, np.ndarray)
    assert result.shape == tpr.shape
    expected = np.clip(tpr - fpr, -1.0, 1.0)
    np.testing.assert_allclose(result, expected)


def test_list_inputs_return_ndarray():
    """
    Using Python lists/tuples should still yield a numpy.ndarray when at
    least one argument is non-scalar.
    """
    tpr = [0.8, 0.9]
    fpr = [0.1, 0.2]

    result = youden_j(tpr, fpr)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    expected = np.clip(np.array(tpr) - np.array(fpr), -1.0, 1.0)
    np.testing.assert_allclose(result, expected)


def test_mixed_list_and_array_inputs():
    """
    Mixed container types (list + ndarray, tuple + ndarray) should work and
    broadcast as expected.
    """
    tpr = [0.1, 0.4, 0.7]
    fpr = np.array([0.2, 0.2, 0.2])

    result = youden_j(tpr, fpr)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    expected = np.clip(np.array(tpr, dtype=float) - fpr, -1.0, 1.0)
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# NaN and inf semantics
# ---------------------------------------------------------------------------


def test_nan_and_inf_are_preserved():
    """
    Non-finite values (NaN, +inf, -inf) should be preserved, not clipped.
    Finite entries are still clipped.
    """
    tpr = np.array([0.9, np.nan, np.inf, -np.inf])
    fpr = np.array([0.1, 0.2, 0.3, 0.4])

    result = youden_j(tpr, fpr)

    # Expected raw: [0.8, nan, inf, -inf]
    assert result.shape == (4,)

    # Index 0: finite → clipped (but still 0.8 here)
    assert result[0] == pytest.approx(0.8)

    # Index 1: NaN preserved
    assert math.isnan(result[1])

    # Index 2: +inf preserved
    assert math.isinf(result[2]) and result[2] > 0

    # Index 3: -inf preserved
    assert math.isinf(result[3]) and result[3] < 0


def test_all_finite_outputs_within_closed_interval():
    """
    For a grid of finite (tpr, fpr) values, the result must always lie in [-1, 1],
    and the clipping should actually hit both boundaries somewhere.
    """
    t_values = np.linspace(-2.0, 2.0, num=21)
    f_values = np.linspace(-2.0, 2.0, num=21)

    t_grid, f_grid = np.meshgrid(t_values, f_values, indexing="ij")
    result = youden_j(t_grid, f_grid)

    assert isinstance(result, np.ndarray)
    assert result.shape == t_grid.shape

    # All finite outputs must lie in [-1, 1] up to tiny numerical slack.
    assert np.all(result <= 1.0 + 1e-12)
    assert np.all(result >= -1.0 - 1e-12)

    # We also expect the clipping to actually *reach* both boundaries.
    max_val = float(result.max())
    min_val = float(result.min())

    assert max_val <= 1.0 + 1e-12
    assert min_val >= -1.0 - 1e-12
    assert max_val >= 1.0 - 1e-6
    assert min_val <= -1.0 + 1e-6


# ---------------------------------------------------------------------------
# Failure modes: TypeError and ValueError
# ---------------------------------------------------------------------------


def test_non_numeric_tpr_raises_typeerror():
    """
    A completely non-numeric TPR should fail during conversion with TypeError.
    """
    with pytest.raises(TypeError) as excinfo:
        youden_j("not numeric", 0.1)  # type: ignore[arg-type]

    msg = str(excinfo.value)
    assert "tpr must be convertible to a float array" in msg


def test_non_numeric_fpr_raises_typeerror():
    """
    A completely non-numeric FPR should fail during conversion with TypeError.
    """
    with pytest.raises(TypeError) as excinfo:
        youden_j(0.9, {"fpr": 0.1})  # type: ignore[arg-type]

    msg = str(excinfo.value)
    assert "fpr must be convertible to a float array" in msg


def test_non_broadcastable_shapes_raise_valueerror():
    """
    Shapes that cannot be broadcast under NumPy rules should raise ValueError
    with a clear message.
    """
    tpr = np.zeros((2,))
    fpr = np.zeros((3,))

    with pytest.raises(ValueError) as excinfo:
        youden_j(tpr, fpr)

    msg = str(excinfo.value)
    assert "not broadcastable" in msg
    # crude but ensures shapes are mentioned
    assert "2" in msg and "3" in msg
