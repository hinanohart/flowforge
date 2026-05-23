"""Deterministic stub policy for CI smoke tests on CPU-only hosts.

The stub produces synthetic actions and a synthetic success signal that is
*sensitive* to the evolve genome's coefficients, so unit tests can exercise
the full pipeline (orchestrator -> evaluate -> CI -> next gen) without
requiring openpi, LIBERO, or a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StubObservation:
    state: np.ndarray
    step: int
    task_id: str


class StubPolicy:
    """A toy policy whose success rate depends on the schedule's mean value.

    Returns success with probability `clip(schedule_mean / 5, 0, 1)`, so
    a "good" coefficient set is one that makes the schedule average ~5.0.
    This is a *deliberately fake* signal — it exists only to verify wiring.
    """

    def __init__(self, schedule_fn: Any, reward_fn: Any, seed: int = 0):
        self.schedule_fn = schedule_fn
        self.reward_fn = reward_fn
        self.rng = np.random.default_rng(seed)
        self._mean_schedule = self._estimate_schedule_mean()

    def _estimate_schedule_mean(self) -> float:
        ts = np.linspace(0.0, 1.0, 32)
        vals = [float(self.schedule_fn(float(t))) for t in ts]
        return float(np.mean(vals))

    def infer(self, obs: StubObservation) -> dict[str, Any]:
        action = self.rng.normal(0.0, 0.1, size=4)
        return {"actions": action, "schedule_mean": self._mean_schedule}

    def predict_success_prob(self) -> float:
        # peak around schedule_mean ≈ 5.0
        return float(math.exp(-((self._mean_schedule - 5.0) ** 2) / 4.0))
