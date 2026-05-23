"""NaN / Inf detector for fitness pipelines.

Any NaN or Inf in a fitness signal forces the candidate to be marked failed
and quarantined under `_wip/nan_<candidate_id>/` (architecture R8 compliance).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def is_finite_scalar(x: Any) -> bool:
    """True iff `x` is a finite float/int (not NaN, not ±Inf)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def find_nonfinite(obj: Any, path: str = "$") -> list[str]:
    """Walk nested dict/list/array; return paths where NaN/Inf live."""
    bad: list[str] = []
    if isinstance(obj, (int, float)):
        if not is_finite_scalar(obj):
            bad.append(f"{path}={obj!r}")
    elif isinstance(obj, np.ndarray):
        if obj.size == 0:
            return bad
        if np.issubdtype(obj.dtype, np.number) and not np.isfinite(obj).all():
            n_bad = int(np.sum(~np.isfinite(obj)))
            bad.append(f"{path}<ndarray>={n_bad} non-finite of {obj.size}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad.extend(find_nonfinite(v, f"{path}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad.extend(find_nonfinite(v, f"{path}[{i}]"))
    return bad


def assert_finite(obj: Any, label: str = "") -> None:
    """Raise ValueError if any leaf of `obj` is non-finite."""
    bad = find_nonfinite(obj)
    if bad:
        prefix = f"{label}: " if label else ""
        raise ValueError(f"{prefix}non-finite values found: {bad[:10]}")
