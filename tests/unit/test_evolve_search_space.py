"""Tests for genome bounds / clamping / random sampling."""

import numpy as np

from flowforge.evolve.search_space import clamp_genome, default_genome, random_genome


def test_default_genome_structure():
    g = default_genome()
    assert "sched_template" in g
    assert "sched_coefs" in g
    assert "reward_template" in g
    assert "reward_coefs" in g


def test_default_genome_passes_clamp_without_warnings():
    cleaned, warnings = clamp_genome(default_genome())
    assert warnings == []
    assert cleaned == default_genome()


def test_clamp_unknown_template_falls_back():
    g = {
        "sched_template": "nope",
        "sched_coefs": {},
        "reward_template": "potential",
        "reward_coefs": {},
    }
    cleaned, warnings = clamp_genome(g)
    assert cleaned["sched_template"] == "polynomial"
    assert any("sched_template" in w for w in warnings)


def test_clamp_out_of_range_value():
    g = default_genome()
    g["sched_coefs"]["a0"] = 1e6
    cleaned, warnings = clamp_genome(g)
    assert cleaned["sched_coefs"]["a0"] <= 10.0
    assert any("a0" in w for w in warnings)


def test_clamp_missing_keys_filled_with_defaults():
    g = {
        "sched_template": "polynomial",
        "sched_coefs": {},
        "reward_template": "potential",
        "reward_coefs": {},
    }
    cleaned, _ = clamp_genome(g)
    assert cleaned["sched_coefs"]["a0"] == 1.0


def test_random_genome_within_bounds():
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = random_genome(rng)
        cleaned, warnings = clamp_genome(g)
        # Random samples come from the same bounds the clamper enforces, so no
        # values should be modified.
        assert warnings == []
        assert cleaned == g
