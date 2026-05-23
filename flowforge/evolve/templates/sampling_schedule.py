"""Fixed sampling-schedule templates over t in [0, 1].

A schedule returns a positive scalar (typically a guidance scale, noise factor,
or denoising-step weight) for a given normalised time `t`. Mutations propose
new *coefficients*; the function structure is fixed.

Each template:
    factory(coefs: dict[str, float]) -> Callable[[float], float]

All templates clip output to a bounded range to prevent NaN/Inf propagation.
"""

from __future__ import annotations

import math
from typing import Callable

OUTPUT_LOW = 0.0
OUTPUT_HIGH = 50.0


def _clip(v: float) -> float:
    if v < OUTPUT_LOW:
        return OUTPUT_LOW
    if v > OUTPUT_HIGH:
        return OUTPUT_HIGH
    return v


def polynomial(coefs: dict[str, float]) -> Callable[[float], float]:
    """y(t) = a0 + a1*t + a2*t^2 + a3*t^3, clipped to [OUTPUT_LOW, OUTPUT_HIGH]."""
    a0 = float(coefs.get("a0", 1.0))
    a1 = float(coefs.get("a1", 0.0))
    a2 = float(coefs.get("a2", 0.0))
    a3 = float(coefs.get("a3", 0.0))

    def schedule(t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        return _clip(a0 + a1 * t + a2 * t * t + a3 * t * t * t)

    schedule.template_id = "polynomial"  # type: ignore[attr-defined]
    schedule.coefs = {"a0": a0, "a1": a1, "a2": a2, "a3": a3}  # type: ignore[attr-defined]
    return schedule


def piecewise(coefs: dict[str, float]) -> Callable[[float], float]:
    """4-segment piecewise-linear on breakpoints b1<b2<b3 in (0,1)."""
    b1 = float(coefs.get("b1", 0.25))
    b2 = float(coefs.get("b2", 0.5))
    b3 = float(coefs.get("b3", 0.75))
    v0 = float(coefs.get("v0", 1.0))
    v1 = float(coefs.get("v1", 1.0))
    v2 = float(coefs.get("v2", 1.0))
    v3 = float(coefs.get("v3", 1.0))
    v4 = float(coefs.get("v4", 1.0))
    # normalise breakpoint order
    bs = sorted([b1, b2, b3])
    # clamp into open interval to avoid div by zero
    bs = [max(1e-6, min(1.0 - 1e-6, b)) for b in bs]
    b1s, b2s, b3s = bs

    def schedule(t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        if t <= b1s:
            frac = t / b1s
            y = v0 + (v1 - v0) * frac
        elif t <= b2s:
            frac = (t - b1s) / (b2s - b1s)
            y = v1 + (v2 - v1) * frac
        elif t <= b3s:
            frac = (t - b2s) / (b3s - b2s)
            y = v2 + (v3 - v2) * frac
        else:
            frac = (t - b3s) / (1.0 - b3s)
            y = v3 + (v4 - v3) * frac
        return _clip(y)

    schedule.template_id = "piecewise"  # type: ignore[attr-defined]
    schedule.coefs = {  # type: ignore[attr-defined]
        "b1": b1s,
        "b2": b2s,
        "b3": b3s,
        "v0": v0,
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "v4": v4,
    }
    return schedule


def cosine(coefs: dict[str, float]) -> Callable[[float], float]:
    """y(t) = dc + amp * cos(2π * omega * t + phi), clipped."""
    omega = float(coefs.get("omega", 1.0))
    phi = float(coefs.get("phi", 0.0))
    amp = float(coefs.get("amp", 0.5))
    dc = float(coefs.get("dc", 1.0))

    def schedule(t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        return _clip(dc + amp * math.cos(2.0 * math.pi * omega * t + phi))

    schedule.template_id = "cosine"  # type: ignore[attr-defined]
    schedule.coefs = {"omega": omega, "phi": phi, "amp": amp, "dc": dc}  # type: ignore[attr-defined]
    return schedule


REGISTRY: dict[str, Callable[[dict[str, float]], Callable[[float], float]]] = {
    "polynomial": polynomial,
    "piecewise": piecewise,
    "cosine": cosine,
}


def build(template_id: str, coefs: dict[str, float]) -> Callable[[float], float]:
    if template_id not in REGISTRY:
        raise ValueError(
            f"unknown schedule template_id: {template_id!r}; allowed: {list(REGISTRY)}"
        )
    return REGISTRY[template_id](coefs)
