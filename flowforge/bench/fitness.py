"""Fitness aggregation across seeds + tasks with bootstrap CI 95%.

This is the canonical FitnessReport producer. Architecture R5 / F2 compliance:
- Every metric carries (point, ci_low, ci_high)
- Empty / NaN results -> failed=True, fitness=0.0, with reason recorded
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from flowforge._types import EpisodeResult, FitnessReport
from flowforge.bench import libero_adapter
from flowforge.bench.episode_runner import RunnerConfig, run_episode
from flowforge.evolve.templates import reward_shaping, sampling_schedule
from flowforge.guard import stats
from flowforge.policy.stub_policy import StubPolicy

log = logging.getLogger(__name__)


def build_callables(genome: dict[str, Any]):
    sched = sampling_schedule.build(genome["sched_template"], genome["sched_coefs"])
    rew = reward_shaping.build(genome["reward_template"], genome["reward_coefs"])
    return sched, rew


def evaluate_genome_stub(
    genome: dict[str, Any],
    *,
    tasks: list[str],
    seeds: list[int],
    candidate_id: str = "anon",
) -> FitnessReport:
    """Evaluate a genome with the stub env + stub policy.

    Used for CI smoke tests and as a fallback on CPU-only hosts.
    """
    outcomes: list[EpisodeResult] = []
    runner_cfg = RunnerConfig()
    try:
        sched, rew = build_callables(genome)
        for task in tasks:
            for seed in seeds:
                env = libero_adapter.make_env(libero_adapter.LiberoConfig(task_suite=task))
                policy = StubPolicy(sched, rew, seed=seed)
                # Bias env's success_prob using the stub policy's signal.
                env.success_prob = policy.predict_success_prob()  # type: ignore[attr-defined]
                outcomes.append(run_episode(env, policy, seed=seed, task=task, cfg=runner_cfg))
    except Exception as e:  # noqa: BLE001 — turn into a failed report
        log.warning("evaluate_genome_stub failed: %s", e)
        return FitnessReport(
            candidate_id=candidate_id,
            n_episodes=0,
            success_rate=0.0,
            success_rate_ci95_low=0.0,
            success_rate_ci95_high=0.0,
            mean_reward=0.0,
            wall_time_s=0.0,
            failed=True,
            failure_reason=str(e),
        )

    successes = [e.success for e in outcomes]
    rewards = [e.reward for e in outcomes]
    if not successes:
        return FitnessReport(
            candidate_id=candidate_id,
            n_episodes=0,
            success_rate=0.0,
            success_rate_ci95_low=0.0,
            success_rate_ci95_high=0.0,
            mean_reward=0.0,
            wall_time_s=0.0,
            failed=True,
            failure_reason="no episodes",
        )

    p, lo, hi = stats.success_rate_ci(successes, n_resamples=2000, rng_seed=0)
    return FitnessReport(
        candidate_id=candidate_id,
        n_episodes=len(outcomes),
        success_rate=p,
        success_rate_ci95_low=lo,
        success_rate_ci95_high=hi,
        mean_reward=float(sum(rewards) / len(rewards)),
        wall_time_s=float(sum(e.wall_time_s for e in outcomes)),
    )


def make_eval_fn(tasks: list[str], seeds: list[int]) -> Callable[[dict[str, Any]], FitnessReport]:
    """Factory: returns a closure with fixed tasks/seeds for the EvolveLoop."""

    def _eval(genome: dict[str, Any]) -> FitnessReport:
        return evaluate_genome_stub(genome, tasks=tasks, seeds=seeds)

    return _eval
