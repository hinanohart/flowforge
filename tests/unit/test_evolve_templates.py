"""Tests for fixed templates."""

import pytest

from flowforge.evolve.templates import reward_shaping, sampling_schedule


def test_polynomial_defaults_const_one():
    f = sampling_schedule.polynomial({})
    assert f(0.0) == pytest.approx(1.0)
    assert f(1.0) == pytest.approx(1.0)


def test_polynomial_linear():
    f = sampling_schedule.polynomial({"a0": 0.0, "a1": 2.0})
    assert f(0.0) == pytest.approx(0.0)
    assert f(0.5) == pytest.approx(1.0)
    assert f(1.0) == pytest.approx(2.0)


def test_polynomial_clipped_high():
    f = sampling_schedule.polynomial({"a0": 1e6})
    assert f(0.0) == sampling_schedule.OUTPUT_HIGH


def test_piecewise_monotone():
    f = sampling_schedule.piecewise({"v0": 0.0, "v1": 1.0, "v2": 2.0, "v3": 3.0, "v4": 4.0})
    vs = [f(t) for t in [0.0, 0.25, 0.5, 0.75, 1.0]]
    assert vs == sorted(vs)


def test_cosine_period():
    f = sampling_schedule.cosine({"omega": 1.0, "phi": 0.0, "amp": 1.0, "dc": 1.0})
    a = f(0.0)
    b = f(1.0)
    assert a == pytest.approx(b)


def test_schedule_registry_unknown():
    with pytest.raises(ValueError):
        sampling_schedule.build("nonexistent", {})


def test_potential_zero_without_prev_state():
    f = reward_shaping.potential({"gamma": 0.99, "scale": 0.1})
    assert f([1.0], {}) == 0.0


def test_potential_with_prev_state():
    f = reward_shaping.potential({"gamma": 0.99, "scale": 1.0})
    val = f([1.0, 0.0], {"prev_state": [0.0, 0.0], "ref": [1.0, 0.0]})
    # φ(s_next)=0, φ(s_prev)=-1 → 0.99*0 - (-1) = 1.0 clipped to 1.0
    assert val == pytest.approx(1.0)


def test_dense_peak_at_target():
    f = reward_shaping.dense({"sigma": 0.5, "amp": 0.5})
    peak = f([1.0, 2.0], {"target": [1.0, 2.0]})
    off = f([5.0, 5.0], {"target": [1.0, 2.0]})
    assert peak > off


def test_sparse_threshold():
    f = reward_shaping.sparse({"threshold": 0.5, "amp": 1.0})
    assert f([0.0], {"target": [0.1]}) == 1.0
    assert f([0.0], {"target": [5.0]}) == 0.0


def test_reward_registry_unknown():
    with pytest.raises(ValueError):
        reward_shaping.build("nonexistent", {})


def test_reward_handles_shape_mismatch():
    f = reward_shaping.dense({"sigma": 0.5, "amp": 1.0})
    assert f([1.0, 2.0], {"target": [1.0]}) == 0.0
