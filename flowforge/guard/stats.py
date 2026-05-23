"""Bootstrap CI 95% — non-parametric confidence interval via resampling.

Architecture R5 compliance: every reported fitness metric carries a 95% CI.
Failure mode F2: if CI is unrealistically wide (low ≤ 0 and high ≥ 1 for a
success-rate), the caller must report "Δ not detected" rather than a point
estimate.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    rng_seed: int | None = 0,
) -> tuple[float, float, float]:
    """Return (point, lo, hi) for a bootstrap CI of `statistic` on `data`.

    Args:
        data: 1-D sample.
        statistic: function np.ndarray -> float.
        n_resamples: number of bootstrap resamples.
        confidence: e.g., 0.95 for a 95% CI.
        rng_seed: deterministic seed (None for nondeterministic).

    Raises:
        ValueError: if data is empty.
    """
    arr = np.asarray(list(data), dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_ci: data is empty")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0,1); got {confidence}")

    rng = np.random.default_rng(rng_seed)
    n = arr.size
    point = float(statistic(arr))

    # Single-sample edge case: CI collapses to the point.
    if n == 1:
        return point, point, point

    resamples = rng.integers(0, n, size=(n_resamples, n))
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        stats[i] = statistic(arr[resamples[i]])
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1.0 - alpha))
    return point, lo, hi


def success_rate_ci(successes: Sequence[bool], **kwargs) -> tuple[float, float, float]:
    """Wrapper for binary outcomes."""
    return bootstrap_ci([1.0 if s else 0.0 for s in successes], statistic=np.mean, **kwargs)


def ci_overlaps(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """True iff intervals [a.lo, a.hi] and [b.lo, b.hi] overlap.

    Used by S5 to decide whether ShinkaEvolve beat the baseline at 95% CI.
    """
    _, a_lo, a_hi = a
    _, b_lo, b_hi = b
    return not (a_hi < b_lo or b_hi < a_lo)


def ci_dominates(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """True iff a's CI lies strictly above b's (a.lo > b.hi)."""
    _, a_lo, _ = a
    _, _, b_hi = b
    return a_lo > b_hi
