"""Episode runner: pumps observations through a policy and records outcomes.

Decoupled from LIBERO/openpi so it can be reused on stubs or other simulators
(this is honest-claim #2 in the README: harness is reusable).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from flowforge._types import EpisodeResult
from flowforge.guard import nan_detector

log = logging.getLogger(__name__)


class Policy(Protocol):
    def infer(self, obs: Any) -> dict[str, Any]: ...
    def reset(self) -> None: ...


class Env(Protocol):
    def reset(self, *, seed: int) -> Any: ...
    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]: ...


@dataclass
class RunnerConfig:
    max_steps: int = 250
    deterministic: bool = True
    success_key: str = "success"


def run_episode(
    env: Env, policy: Policy, *, seed: int, task: str, cfg: RunnerConfig
) -> EpisodeResult:
    start = time.monotonic()
    obs = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    total_reward = 0.0
    success = False
    n_steps = 0
    for n_steps in range(1, cfg.max_steps + 1):
        out = policy.infer(obs)
        action = out["actions"]
        nan_detector.assert_finite(action, label=f"action@step{n_steps}")
        # If policy returned an action chunk, take the first.
        action_arr = np.asarray(action, dtype=float)
        if action_arr.ndim > 1:
            action_arr = action_arr[0]
        obs, reward, done, info = env.step(action_arr)
        total_reward += float(reward)
        if isinstance(info, dict) and info.get(cfg.success_key):
            success = True
            break
        if done:
            break
    return EpisodeResult(
        task=task,
        seed=seed,
        success=success,
        reward=total_reward,
        n_steps=n_steps,
        wall_time_s=time.monotonic() - start,
    )
