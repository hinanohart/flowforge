"""Random-grid baseline — uniformly sample N genomes; no learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from flowforge.evolve.search_space import random_genome


@dataclass
class RandomGridConfig:
    n_samples: int = 100
    seed: int = 0


def run_random_grid(
    eval_fn: Callable[[dict[str, Any]], Any],
    config: RandomGridConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RandomGridConfig()
    rng = np.random.default_rng(cfg.seed)
    best_score = float("-inf")
    best_genome: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for i in range(cfg.n_samples):
        g = random_genome(rng)
        report = eval_fn(g)
        score = float(getattr(report, "success_rate", 0.0))
        history.append({"i": i, "score": score, "genome": g})
        if score > best_score:
            best_score = score
            best_genome = g
    return {"best_genome": best_genome, "best_score": best_score, "history": history}
