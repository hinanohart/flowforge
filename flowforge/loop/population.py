"""Population helpers thin enough to live next to the orchestrator."""

from __future__ import annotations

from typing import Any

from flowforge.evolve.search_space import clamp_genome, default_genome, random_genome


def seed_population(rng, size: int) -> list[dict[str, Any]]:
    pop = [default_genome()]
    while len(pop) < size:
        pop.append(random_genome(rng))
    return [clamp_genome(g)[0] for g in pop]


def best_of(population: list[dict[str, Any]], scores: list[float]) -> tuple[dict[str, Any], float]:
    idx = max(range(len(scores)), key=lambda i: scores[i])
    return population[idx], float(scores[idx])
