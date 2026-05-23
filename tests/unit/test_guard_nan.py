"""Tests for NaN/Inf detection."""

import math

import numpy as np
import pytest

from flowforge.guard import nan_detector


def test_is_finite_scalar():
    assert nan_detector.is_finite_scalar(1.0)
    assert not nan_detector.is_finite_scalar(float("nan"))
    assert not nan_detector.is_finite_scalar(float("inf"))
    assert not nan_detector.is_finite_scalar("not-a-number")


def test_find_nonfinite_in_dict():
    bad = nan_detector.find_nonfinite({"a": 1.0, "b": float("nan"), "c": [1, 2, float("inf")]})
    assert any("b" in p for p in bad)
    assert any("c" in p for p in bad)


def test_assert_finite_passes():
    nan_detector.assert_finite({"x": 1, "y": [1, 2, 3]})


def test_assert_finite_raises():
    with pytest.raises(ValueError):
        nan_detector.assert_finite([1.0, math.nan, 3.0], label="reward")


def test_find_nonfinite_ndarray():
    arr = np.array([1.0, 2.0, np.inf, 4.0])
    bad = nan_detector.find_nonfinite(arr)
    assert len(bad) == 1
    assert "1 non-finite" in bad[0]


def test_find_nonfinite_empty_ok():
    assert nan_detector.find_nonfinite(np.array([])) == []
