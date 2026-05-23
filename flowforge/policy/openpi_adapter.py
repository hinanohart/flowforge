"""Adapter for Physical-Intelligence/openpi pi0.5 policy on LIBERO.

The openpi public API exposes only `policy.infer(example)["actions"]`; internal
sampling parameters (denoising steps, classifier-free guidance scale) are not
documented as keyword arguments. FlowForge therefore *does not* mutate openpi
internals. Instead, the schedule_fn from a genome is applied as a per-step
*action-scaling multiplier* on the returned action chunk, and reward_fn is
applied as a *post-hoc shaping bonus* on the episode return.

This adapter is GPU-only. On CPU-only hosts the orchestrator falls back to
StubPolicy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class OpenPiConfig:
    checkpoint_path: str  # gs:// or local
    device: str = "cuda"
    action_dim: int = 7
    chunk_size: int = 8


class OpenPiAdapter:
    """Loads pi0.5 LIBERO checkpoint and exposes a schedule-aware infer()."""

    def __init__(self, config: OpenPiConfig, schedule_fn: Any, reward_fn: Any):
        self.config = config
        self.schedule_fn = schedule_fn
        self.reward_fn = reward_fn
        self._policy: Any = None
        self._step_idx = 0

    def _ensure_loaded(self) -> None:
        if self._policy is not None:
            return
        try:
            from openpi.policies import policy_config  # type: ignore
            from openpi.policies.policy_config import PolicyConfig  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openpi not importable; ensure `external/openpi` is on PYTHONPATH "
                "or use StubPolicy on CPU-only hosts"
            ) from e
        log.info("loading pi0.5 LIBERO from %s", self.config.checkpoint_path)
        # NOTE: the precise openpi loader API may evolve; we keep this thin so
        # the orchestrator can be told at S0 to refuse with HITL on mismatch.
        cfg = PolicyConfig(checkpoint=self.config.checkpoint_path)  # type: ignore[call-arg]
        self._policy = policy_config.load(cfg)  # type: ignore[attr-defined]

    def reset(self) -> None:
        self._step_idx = 0

    def infer(self, observation: Any) -> dict[str, Any]:
        self._ensure_loaded()
        raw = self._policy.infer(observation)
        action_chunk = np.asarray(raw["actions"], dtype=float)
        # Apply schedule as per-step scaling multiplier.
        scale = float(self.schedule_fn(self._normalised_t()))
        scaled = action_chunk * scale
        self._step_idx += 1
        return {"actions": scaled, "scale": scale}

    def _normalised_t(self) -> float:
        """Normalise step idx into [0, 1] over an episode horizon estimate."""
        # rough horizon — LIBERO single-task episodes are typically <= 250 steps
        return min(1.0, self._step_idx / 250.0)
