"""Coefficient-only evolution loop. NOTE: v0.1.0 ships the built-in EvolveLoop
below; the full SakanaAI/ShinkaEvolve `ShinkaEvolveRunner` integration is
NOT invoked in v0.1.0 — `try_shinka_runner` returns None until v0.2.

Why: ShinkaEvolve's full code-rewriting (diff/full/cross patches) is more
power than v0.1.0's frozen scope needs. We restrict the search space to
fixed templates × coefficients, so a built-in tournament+elitism loop with
clamp + random/LLM mutation is sufficient and easier to reason about.

v0.2 will wire ShinkaEvolve via the `cross` patch type behind an
EVOLVE-BLOCK that holds only a JSON literal of coefficients.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from flowforge.evolve.search_space import clamp_genome, default_genome, random_genome

log = logging.getLogger(__name__)


@dataclass
class EvolveConfig:
    n_generations: int = 30
    population_size: int = 8
    elitism: int = 2
    mutation_rate: float = 0.5
    seed: int = 42
    n_islands: int = 1


class EvolveLoop:
    """Coefficient-only evolutionary loop.

    The `eval_fn(genome) -> FitnessReport`-style callable is supplied by the
    orchestrator (which wires it to LIBERO + bootstrap CI).
    """

    def __init__(self, config: EvolveConfig, eval_fn: Callable[[dict[str, Any]], Any]):
        self.config = config
        self.eval_fn = eval_fn
        self.rng = np.random.default_rng(config.seed)
        self.population: list[dict[str, Any]] = []
        self.scores: list[float] = []
        self.history: list[dict[str, Any]] = []

    def seed_population(self) -> None:
        self.population = [default_genome()]
        while len(self.population) < self.config.population_size:
            self.population.append(random_genome(self.rng))
        self.scores = [float("-inf")] * len(self.population)

    def evaluate(self) -> None:
        """Run eval_fn on every candidate; report success_rate as scalar score."""
        for i, g in enumerate(self.population):
            try:
                report = self.eval_fn(g)
                score = getattr(report, "success_rate", None)
                if score is None and isinstance(report, dict):
                    score = report.get("success_rate", 0.0)
                self.scores[i] = float(score) if score is not None else 0.0
            except Exception as e:  # noqa: BLE001 — pop-level safety
                log.warning("eval failed for candidate %d: %s", i, e)
                self.scores[i] = 0.0

    def select_and_mutate(self, mutator: Callable[[dict[str, Any]], dict[str, Any]] | None) -> None:
        """Tournament selection + LLM/random mutation."""
        order = np.argsort(self.scores)[::-1]
        elites = [self.population[i] for i in order[: self.config.elitism]]
        new_pop = list(elites)
        while len(new_pop) < self.config.population_size:
            i, j = self.rng.integers(0, len(self.population), size=2)
            winner = self.population[i] if self.scores[i] >= self.scores[j] else self.population[j]
            if mutator is not None and self.rng.random() < self.config.mutation_rate:
                child = mutator(dict(winner))
            else:
                child = random_genome(self.rng)
            child, _ = clamp_genome(child)
            new_pop.append(child)
        self.population = new_pop
        self.scores = [float("-inf")] * len(new_pop)

    def run(
        self, mutator: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        self.seed_population()
        for gen in range(self.config.n_generations):
            self.evaluate()
            best_idx = int(np.argmax(self.scores))
            self.history.append(
                {
                    "generation": gen,
                    "best_score": float(self.scores[best_idx]),
                    "mean_score": float(np.mean(self.scores)),
                    "best_genome": self.population[best_idx],
                }
            )
            log.info("gen=%d best=%.4f mean=%.4f", gen, self.scores[best_idx], np.mean(self.scores))
            if gen < self.config.n_generations - 1:
                self.select_and_mutate(mutator)
        return {"history": self.history, "best": self.history[-1]["best_genome"]}


def try_shinka_runner(*_args, **_kwargs) -> None:
    """Optional ShinkaEvolveRunner factory.

    Returns None if shinka-evolve is not installed. v0.1.0 uses our built-in
    EvolveLoop; we keep this stub to make v0.2 integration explicit.
    """
    try:
        import shinka  # noqa: F401
    except ImportError:
        return None
    return None  # explicit: v0.1.0 doesn't wire the full runner yet


def evolve_payload_as_json(genome: dict[str, Any]) -> str:
    """Serialise a genome to a JSON string that fits inside an EVOLVE-BLOCK."""
    return json.dumps(genome, indent=2, sort_keys=True)
