"""Tests for fitness aggregation."""

from flowforge.bench import fitness
from flowforge.evolve.search_space import default_genome


def test_evaluate_genome_stub_returns_report():
    rep = fitness.evaluate_genome_stub(default_genome(), tasks=["libero_spatial"], seeds=[0, 1, 2])
    assert rep.n_episodes == 3
    assert 0.0 <= rep.success_rate <= 1.0
    assert rep.success_rate_ci95_low <= rep.success_rate <= rep.success_rate_ci95_high


def test_make_eval_fn():
    ef = fitness.make_eval_fn(["libero_spatial"], [0])
    rep = ef(default_genome())
    assert rep.n_episodes == 1


def test_build_callables():
    sched, rew = fitness.build_callables(default_genome())
    assert callable(sched)
    assert callable(rew)
    assert isinstance(sched(0.5), float)


def test_evaluate_handles_invalid_template():
    g = default_genome()
    g["sched_template"] = "not_a_real_template"
    rep = fitness.evaluate_genome_stub(g, tasks=["libero_spatial"], seeds=[0])
    assert rep.failed
