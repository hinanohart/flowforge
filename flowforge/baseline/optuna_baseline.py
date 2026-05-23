"""Optuna baseline (S2_baseline_parametric).

Runs parametric search over the same coefficient bounds as ShinkaEvolve, so
that the final report can answer "did evolution actually beat tuned random
search?" honestly. If Optuna is unavailable, returns an empty report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from flowforge.evolve.search_space import REWARD_BOUNDS, SCHED_BOUNDS

log = logging.getLogger(__name__)


@dataclass
class OptunaConfig:
    n_trials: int = 100
    sampler_seed: int = 0
    direction: str = "maximize"
    timeout_s: float | None = None


def run_optuna(
    eval_fn: Callable[[dict[str, Any]], Any],
    sched_template: str,
    reward_template: str,
    config: OptunaConfig | None = None,
) -> dict[str, Any]:
    """Returns a dict with `best_genome`, `best_score`, `trials`."""
    cfg = config or OptunaConfig()
    try:
        import optuna
    except ImportError:
        log.warning("optuna not installed; skipping parametric baseline")
        return {"best_genome": None, "best_score": float("-inf"), "trials": []}

    sched_bounds = SCHED_BOUNDS[sched_template]
    reward_bounds = REWARD_BOUNDS[reward_template]

    def objective(trial: "optuna.Trial") -> float:
        genome = {
            "sched_template": sched_template,
            "sched_coefs": {
                b.name: trial.suggest_float(f"s_{b.name}", b.low, b.high) for b in sched_bounds
            },
            "reward_template": reward_template,
            "reward_coefs": {
                b.name: trial.suggest_float(f"r_{b.name}", b.low, b.high) for b in reward_bounds
            },
        }
        report = eval_fn(genome)
        return float(getattr(report, "success_rate", 0.0))

    sampler = optuna.samplers.TPESampler(seed=cfg.sampler_seed)
    study = optuna.create_study(direction=cfg.direction, sampler=sampler)
    study.optimize(objective, n_trials=cfg.n_trials, timeout=cfg.timeout_s, show_progress_bar=False)

    best = study.best_trial
    best_genome = {
        "sched_template": sched_template,
        "sched_coefs": {b.name: best.params[f"s_{b.name}"] for b in sched_bounds},
        "reward_template": reward_template,
        "reward_coefs": {b.name: best.params[f"r_{b.name}"] for b in reward_bounds},
    }
    return {
        "best_genome": best_genome,
        "best_score": float(best.value),
        "trials": [{"params": t.params, "value": t.value} for t in study.trials],
    }
