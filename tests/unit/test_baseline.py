"""Tests for parametric baselines."""

from flowforge._types import FitnessReport
from flowforge.baseline import RandomGridConfig, run_random_grid
from flowforge.baseline.optuna_baseline import OptunaConfig, run_optuna


def stub_eval(genome):
    a0 = float(genome.get("sched_coefs", {}).get("a0", 0.0))
    return FitnessReport(
        candidate_id="b",
        n_episodes=1,
        success_rate=a0 / 10.0,
        success_rate_ci95_low=max(0.0, a0 / 10.0 - 0.05),
        success_rate_ci95_high=min(1.0, a0 / 10.0 + 0.05),
        mean_reward=0.0,
        wall_time_s=0.0,
    )


def test_random_grid_basic():
    out = run_random_grid(stub_eval, RandomGridConfig(n_samples=12, seed=0))
    assert out["best_score"] >= 0.0
    assert out["best_genome"] is not None
    assert len(out["history"]) == 12


def test_random_grid_zero_samples():
    out = run_random_grid(stub_eval, RandomGridConfig(n_samples=0, seed=0))
    assert out["best_genome"] is None
    assert out["best_score"] == float("-inf")


def test_optuna_when_available():
    try:
        import optuna  # noqa: F401
    except ImportError:
        return  # not installed in this env; skip silently
    out = run_optuna(stub_eval, "polynomial", "potential", OptunaConfig(n_trials=8, sampler_seed=0))
    assert out["best_genome"] is not None
    assert 0.0 <= out["best_score"] <= 1.0
