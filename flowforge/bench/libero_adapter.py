"""LIBERO environment adapter.

When LIBERO is not installed (e.g., CI on CPU-only hosts), a deterministic
in-memory `StubEnv` is returned so the rest of the pipeline still runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class LiberoConfig:
    task_suite: str = "libero_spatial"  # spatial / object / goal / 10
    task_id: int = 0
    obs_dim: int = 16


class StubEnv:
    """CPU-only deterministic stand-in for LIBERO.

    Reward is a function of action norm; episode succeeds when cumulative
    reward exceeds a threshold. This lets stub policies "succeed" with
    probability tunable via the schedule.
    """

    def __init__(self, config: LiberoConfig, success_prob: float = 0.3):
        self.config = config
        self.success_prob = float(np.clip(success_prob, 0.0, 1.0))
        self.rng = np.random.default_rng(0)
        self._step = 0

    def reset(self, *, seed: int) -> Any:
        self.rng = np.random.default_rng(seed)
        self._step = 0
        return self.rng.normal(0.0, 0.1, size=self.config.obs_dim)

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        self._step += 1
        a = np.asarray(action, dtype=float).reshape(-1)
        reward = -float(np.linalg.norm(a) * 0.01)
        done = self._step >= 32
        # success draw at episode end, weighted by success_prob
        success = False
        if done and self.rng.random() < self.success_prob:
            success = True
            reward += 1.0
        obs = self.rng.normal(0.0, 0.1, size=self.config.obs_dim)
        return obs, reward, done, {"success": success}


def make_env(config: LiberoConfig, *, success_prob_hint: float | None = None):
    """Return a real LIBERO env if importable, else a StubEnv."""
    sp = 0.3 if success_prob_hint is None else float(success_prob_hint)
    try:
        import libero  # noqa: F401  # type: ignore
    except ImportError:
        log.info("LIBERO not installed; using StubEnv (success_prob=%.3f)", sp)
        return StubEnv(config, success_prob=sp)
    # When LIBERO is installed we leave construction to the user-supplied helper;
    # v0.1.0 does not ship a tested LIBERO loader (Phase 0 bootstrap requirement).
    raise NotImplementedError(
        "real LIBERO loader not wired in v0.1.0; use StubEnv or extend libero_adapter.make_env"
    )
