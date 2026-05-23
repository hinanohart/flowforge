"""Tests for the built-in EvolveLoop."""

from flowforge._types import FitnessReport
from flowforge.evolve import EvolveConfig, EvolveLoop


def constant_eval(genome):
    return FitnessReport(
        candidate_id="x",
        n_episodes=1,
        success_rate=0.5,
        success_rate_ci95_low=0.4,
        success_rate_ci95_high=0.6,
        mean_reward=0.0,
        wall_time_s=0.0,
    )


def gradient_eval(genome):
    """Higher a0 → higher fitness."""
    a0 = float(genome.get("sched_coefs", {}).get("a0", 0.0))
    sr = max(0.0, min(1.0, a0 / 10.0))
    return FitnessReport(
        candidate_id="x",
        n_episodes=1,
        success_rate=sr,
        success_rate_ci95_low=max(0.0, sr - 0.05),
        success_rate_ci95_high=min(1.0, sr + 0.05),
        mean_reward=0.0,
        wall_time_s=0.0,
    )


def test_evolve_loop_runs_minimal():
    loop = EvolveLoop(EvolveConfig(n_generations=2, population_size=4, seed=0), constant_eval)
    result = loop.run()
    assert "best" in result
    assert len(result["history"]) == 2


def test_evolve_loop_population_correct_size():
    loop = EvolveLoop(EvolveConfig(n_generations=1, population_size=6, seed=0), constant_eval)
    loop.seed_population()
    assert len(loop.population) == 6


def test_evolve_loop_makes_progress_with_gradient_signal():
    loop = EvolveLoop(EvolveConfig(n_generations=4, population_size=6, seed=1), gradient_eval)
    result = loop.run()
    scores = [h["best_score"] for h in result["history"]]
    # Monotone non-decreasing for best-so-far is not enforced here, but
    # final ≥ first should hold given elitism.
    assert scores[-1] >= scores[0]


def test_evolve_loop_with_mutator():
    def identity_mutator(g):
        return g

    loop = EvolveLoop(EvolveConfig(n_generations=2, population_size=4, seed=0), constant_eval)
    result = loop.run(mutator=identity_mutator)
    assert result["history"][-1]["best_score"] == 0.5
