"""Tests for bootstrap CI helpers."""

import numpy as np
import pytest

from flowforge.guard import stats


def test_bootstrap_ci_basic():
    data = [0, 0, 1, 1, 1]
    p, lo, hi = stats.bootstrap_ci(data, rng_seed=0)
    assert 0.0 <= lo <= p <= hi <= 1.0


def test_bootstrap_ci_singleton_collapses():
    p, lo, hi = stats.bootstrap_ci([0.7], rng_seed=0)
    assert (p, lo, hi) == (0.7, 0.7, 0.7)


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        stats.bootstrap_ci([])


def test_invalid_confidence_raises():
    with pytest.raises(ValueError):
        stats.bootstrap_ci([1, 2, 3], confidence=1.5)


def test_success_rate_ci():
    p, lo, hi = stats.success_rate_ci([True, False, True, True], rng_seed=0)
    assert 0.0 <= lo <= p <= hi <= 1.0


def test_ci_overlaps_true():
    a = (0.5, 0.4, 0.6)
    b = (0.5, 0.45, 0.55)
    assert stats.ci_overlaps(a, b)


def test_ci_overlaps_false():
    a = (0.8, 0.7, 0.9)
    b = (0.3, 0.2, 0.4)
    assert not stats.ci_overlaps(a, b)


def test_ci_dominates():
    a = (0.8, 0.7, 0.9)
    b = (0.4, 0.3, 0.5)
    assert stats.ci_dominates(a, b)
    assert not stats.ci_dominates(b, a)


def test_bootstrap_ci_deterministic_with_seed():
    data = list(np.random.RandomState(0).random(50))
    a = stats.bootstrap_ci(data, n_resamples=500, rng_seed=99)
    b = stats.bootstrap_ci(data, n_resamples=500, rng_seed=99)
    assert a == b
