"""Coefficient bounds and validation for the fixed-template search space.

The LLM mutation router is constrained to emit JSON of the form:
    {"sched_template": "<id>", "sched_coefs": {...},
     "reward_template": "<id>", "reward_coefs": {...}}
This module clamps coefs to legal bounds and reports out-of-range proposals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flowforge.evolve.templates import reward_shaping, sampling_schedule


@dataclass(frozen=True)
class Bound:
    name: str
    low: float
    high: float
    default: float

    def clamp(self, x: float) -> float:
        return max(self.low, min(self.high, float(x)))


SCHED_BOUNDS: dict[str, list[Bound]] = {
    "polynomial": [
        Bound("a0", 0.0, 10.0, 1.0),
        Bound("a1", -10.0, 10.0, 0.0),
        Bound("a2", -10.0, 10.0, 0.0),
        Bound("a3", -10.0, 10.0, 0.0),
    ],
    "piecewise": [
        Bound("b1", 0.05, 0.95, 0.25),
        Bound("b2", 0.05, 0.95, 0.50),
        Bound("b3", 0.05, 0.95, 0.75),
        Bound("v0", 0.0, 10.0, 1.0),
        Bound("v1", 0.0, 10.0, 1.0),
        Bound("v2", 0.0, 10.0, 1.0),
        Bound("v3", 0.0, 10.0, 1.0),
        Bound("v4", 0.0, 10.0, 1.0),
    ],
    "cosine": [
        Bound("omega", 0.1, 5.0, 1.0),
        Bound("phi", -3.1416, 3.1416, 0.0),
        Bound("amp", 0.0, 5.0, 0.5),
        Bound("dc", 0.0, 10.0, 1.0),
    ],
}

REWARD_BOUNDS: dict[str, list[Bound]] = {
    "potential": [
        Bound("gamma", 0.5, 0.999, 0.99),
        Bound("scale", 0.0, 1.0, 0.1),
    ],
    "dense": [
        Bound("sigma", 0.05, 5.0, 0.5),
        Bound("amp", 0.0, 1.0, 0.5),
    ],
    "sparse": [
        Bound("threshold", 0.0, 5.0, 0.1),
        Bound("amp", 0.0, 1.0, 0.5),
    ],
}


def default_genome() -> dict[str, Any]:
    """A safe starting genome (used for population seeding)."""
    return {
        "sched_template": "polynomial",
        "sched_coefs": {b.name: b.default for b in SCHED_BOUNDS["polynomial"]},
        "reward_template": "potential",
        "reward_coefs": {b.name: b.default for b in REWARD_BOUNDS["potential"]},
    }


def clamp_genome(genome: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Clamp a possibly-out-of-bounds genome; return (cleaned, warnings)."""
    warnings: list[str] = []
    sched_id = genome.get("sched_template", "polynomial")
    reward_id = genome.get("reward_template", "potential")
    if sched_id not in sampling_schedule.REGISTRY:
        warnings.append(f"unknown sched_template={sched_id!r}, defaulting to polynomial")
        sched_id = "polynomial"
    if reward_id not in reward_shaping.REGISTRY:
        warnings.append(f"unknown reward_template={reward_id!r}, defaulting to potential")
        reward_id = "potential"

    raw_sched = dict(genome.get("sched_coefs", {}))
    raw_reward = dict(genome.get("reward_coefs", {}))
    cleaned_sched: dict[str, float] = {}
    for b in SCHED_BOUNDS[sched_id]:
        if b.name in raw_sched:
            v = b.clamp(raw_sched[b.name])
            if v != float(raw_sched[b.name]):
                warnings.append(f"sched.{b.name} clamped to {v}")
            cleaned_sched[b.name] = v
        else:
            cleaned_sched[b.name] = b.default
    cleaned_reward: dict[str, float] = {}
    for b in REWARD_BOUNDS[reward_id]:
        if b.name in raw_reward:
            v = b.clamp(raw_reward[b.name])
            if v != float(raw_reward[b.name]):
                warnings.append(f"reward.{b.name} clamped to {v}")
            cleaned_reward[b.name] = v
        else:
            cleaned_reward[b.name] = b.default

    return (
        {
            "sched_template": sched_id,
            "sched_coefs": cleaned_sched,
            "reward_template": reward_id,
            "reward_coefs": cleaned_reward,
        },
        warnings,
    )


def random_genome(rng) -> dict[str, Any]:
    """Sample a uniformly-random genome (for baseline/random_grid)."""
    sched_id = rng.choice(list(SCHED_BOUNDS.keys()))
    reward_id = rng.choice(list(REWARD_BOUNDS.keys()))
    return {
        "sched_template": sched_id,
        "sched_coefs": {b.name: float(rng.uniform(b.low, b.high)) for b in SCHED_BOUNDS[sched_id]},
        "reward_template": reward_id,
        "reward_coefs": {
            b.name: float(rng.uniform(b.low, b.high)) for b in REWARD_BOUNDS[reward_id]
        },
    }
