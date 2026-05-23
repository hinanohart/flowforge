"""Wall-clock and call-count budget guards.

Used to wrap evaluation of a candidate so that runaway loops or infinite
recursion in LLM-generated code cannot exceed a configured envelope.
"""

from __future__ import annotations

import contextlib
import signal
import time


class BudgetExceeded(Exception):
    """Raised when a wall-clock budget is exhausted."""


@contextlib.contextmanager
def time_budget(seconds: float):
    """Context manager that raises BudgetExceeded if `seconds` elapses.

    Uses SIGALRM (POSIX). Nested usage is not safe; do not nest.
    """
    if seconds <= 0:
        raise ValueError("seconds must be > 0")

    def _handler(signum, frame):  # noqa: ARG001 — signal handler signature
        raise BudgetExceeded(f"time_budget exceeded after {seconds}s")

    prev = signal.signal(signal.SIGALRM, _handler)
    # signal.setitimer supports sub-second precision; preferred over signal.alarm
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


class CallCounter:
    """Track call counts and raise if a threshold is exceeded.

    Useful for bounding the number of policy.infer() calls per candidate eval.
    """

    __slots__ = ("max_calls", "n_calls", "_label")

    def __init__(self, max_calls: int, label: str = "calls"):
        if max_calls <= 0:
            raise ValueError("max_calls must be > 0")
        self.max_calls = max_calls
        self.n_calls = 0
        self._label = label

    def bump(self, n: int = 1) -> None:
        self.n_calls += n
        if self.n_calls > self.max_calls:
            raise BudgetExceeded(
                f"{self._label} budget exceeded ({self.n_calls} > {self.max_calls})"
            )

    def __repr__(self) -> str:
        return f"CallCounter({self._label}: {self.n_calls}/{self.max_calls})"


class WallclockTracker:
    """Cumulative wall-clock tracker for multi-step runs (e.g., per-gen)."""

    __slots__ = ("started_at", "max_seconds", "_label")

    def __init__(self, max_seconds: float, label: str = "wallclock"):
        self.started_at = time.monotonic()
        self.max_seconds = max_seconds
        self._label = label

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def check(self) -> None:
        if self.elapsed() > self.max_seconds:
            raise BudgetExceeded(
                f"{self._label} budget exceeded ({self.elapsed():.1f}s > {self.max_seconds}s)"
            )

    def remaining(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed())
