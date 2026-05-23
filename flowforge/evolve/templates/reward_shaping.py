"""Fixed reward-shaping templates.

A shaped reward augments the task reward with a learned/heuristic bonus.
Templates are pure functions; only the coefficients are mutated.

Each template:
    factory(coefs: dict[str, float]) -> Callable[[state, info], float]

`state` is a numpy array (task-specific); `info` is a dict (must include
`target` for `dense` and `gamma` for `potential`).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

BONUS_LOW = -1.0
BONUS_HIGH = 1.0


def _clip(v: float) -> float:
    if not math.isfinite(v):
        return 0.0
    if v < BONUS_LOW:
        return BONUS_LOW
    if v > BONUS_HIGH:
        return BONUS_HIGH
    return v


def _to_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def potential(coefs: dict[str, float]) -> Callable[[Any, Any], float]:
    """F(s, s') = γ * φ(s') − φ(s), with φ(s) = -scale * ||s − ref||.

    Ng-style potential shaping (Ng, Harada, Russell 1999). Always
    policy-invariant under standard MDP assumptions.
    """
    gamma = float(coefs.get("gamma", 0.99))
    scale = float(coefs.get("scale", 0.1))

    def reward(state: Any, info: Any) -> float:
        prev_state = info.get("prev_state") if isinstance(info, dict) else None
        ref = info.get("ref", None) if isinstance(info, dict) else None
        if prev_state is None or ref is None:
            return 0.0
        s = _to_array(state)
        ps = _to_array(prev_state)
        r = _to_array(ref)
        if s.shape != ps.shape or s.shape != r.shape:
            return 0.0
        phi_next = -scale * float(np.linalg.norm(s - r))
        phi_prev = -scale * float(np.linalg.norm(ps - r))
        return _clip(gamma * phi_next - phi_prev)

    reward.template_id = "potential"  # type: ignore[attr-defined]
    reward.coefs = {"gamma": gamma, "scale": scale}  # type: ignore[attr-defined]
    return reward


def dense(coefs: dict[str, float]) -> Callable[[Any, Any], float]:
    """Gaussian bonus around `target`: r = amp * exp(-||s − target||^2 / 2σ^2)."""
    sigma = max(1e-3, float(coefs.get("sigma", 0.5)))
    amp = float(coefs.get("amp", 0.5))

    def reward(state: Any, info: Any) -> float:
        target = info.get("target") if isinstance(info, dict) else None
        if target is None:
            return 0.0
        s = _to_array(state)
        t = _to_array(target)
        if s.shape != t.shape:
            return 0.0
        d2 = float(np.sum((s - t) ** 2))
        return _clip(amp * math.exp(-d2 / (2.0 * sigma * sigma)))

    reward.template_id = "dense"  # type: ignore[attr-defined]
    reward.coefs = {"sigma": sigma, "amp": amp}  # type: ignore[attr-defined]
    return reward


def sparse(coefs: dict[str, float]) -> Callable[[Any, Any], float]:
    """Binary bonus: amp if ||s − target|| < threshold, else 0."""
    threshold = max(0.0, float(coefs.get("threshold", 0.1)))
    amp = float(coefs.get("amp", 0.5))

    def reward(state: Any, info: Any) -> float:
        target = info.get("target") if isinstance(info, dict) else None
        if target is None:
            return 0.0
        s = _to_array(state)
        t = _to_array(target)
        if s.shape != t.shape:
            return 0.0
        d = float(np.linalg.norm(s - t))
        return _clip(amp if d < threshold else 0.0)

    reward.template_id = "sparse"  # type: ignore[attr-defined]
    reward.coefs = {"threshold": threshold, "amp": amp}  # type: ignore[attr-defined]
    return reward


REGISTRY: dict[str, Callable[[dict[str, float]], Callable[[Any, Any], float]]] = {
    "potential": potential,
    "dense": dense,
    "sparse": sparse,
}


def build(template_id: str, coefs: dict[str, float]) -> Callable[[Any, Any], float]:
    if template_id not in REGISTRY:
        raise ValueError(f"unknown reward template_id: {template_id!r}; allowed: {list(REGISTRY)}")
    return REGISTRY[template_id](coefs)
