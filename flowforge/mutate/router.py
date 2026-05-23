"""Mutation router: picks a backend (local Qwen / HF API / random) by state.

S0_init, S1, S2  → HF API allowed (bootstrap convenience)
S3_evolve_main   → local Qwen ONLY (cost-cap and reproducibility)
S5+              → no mutation (post-search analysis)

The router validates LLM output against `search_space.clamp_genome`; if
clamping warns or JSON is malformed, falls back to a random mutation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from flowforge.evolve.search_space import clamp_genome, random_genome

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are FlowForge's mutation engine. You receive a parent genome (JSON) and you "
    "return a child genome (JSON only, no prose). Modify 1-3 coefficients. The schema "
    "is fixed; coefficient ranges are documented; never invent new keys."
)


@dataclass
class MutateContext:
    parent_score: float | None = None
    parent_ci: tuple[float, float] | None = None
    best_score: float | None = None
    state: str = "S1_baseline_pi0_libero"


class Router:
    """Selects the active LLM backend based on the orchestrator's current state."""

    def __init__(
        self,
        rng_seed: int = 0,
        local_client_factory: Callable[[], Any] | None = None,
        hf_client_factory: Callable[[], Any] | None = None,
    ):
        self.rng = np.random.default_rng(rng_seed)
        self._local_factory = local_client_factory
        self._hf_factory = hf_client_factory
        self._local: Any = None
        self._hf: Any = None

    def _local_client(self):
        if self._local is None and self._local_factory is not None:
            self._local = self._local_factory()
        return self._local

    def _hf_client(self):
        if self._hf is None and self._hf_factory is not None:
            self._hf = self._hf_factory()
        return self._hf

    def select_backend(self, state: str) -> str:
        if state == "S3_evolve_main":
            return "local_qwen"
        if state in {"S0_init", "S1_baseline_pi0_libero", "S2_baseline_parametric"}:
            return "hf_api" if self._hf_factory is not None else "random"
        return "random"

    def mutate(self, parent: dict[str, Any], ctx: MutateContext) -> dict[str, Any]:
        backend = self.select_backend(ctx.state)
        if backend == "random":
            child = random_genome(self.rng)
            cleaned, _ = clamp_genome(child)
            return cleaned

        user_prompt = self._build_user(parent, ctx)
        client = self._local_client() if backend == "local_qwen" else self._hf_client()
        if client is None:
            log.info("backend %s unavailable, falling back to random", backend)
            cleaned, _ = clamp_genome(random_genome(self.rng))
            return cleaned

        try:
            raw = client.complete_json(SYSTEM_PROMPT, user_prompt)
        except (ValueError, RuntimeError) as e:
            log.warning("LLM mutate failed (%s): %s — falling back to random", backend, e)
            cleaned, _ = clamp_genome(random_genome(self.rng))
            return cleaned

        cleaned, warnings = clamp_genome(raw)
        if warnings:
            log.info("clamped LLM proposal: %s", warnings)
        return cleaned

    def _build_user(self, parent: dict[str, Any], ctx: MutateContext) -> str:
        import json as _json

        lines = ["Parent genome:", "```json", _json.dumps(parent, indent=2, sort_keys=True), "```"]
        if ctx.parent_score is not None:
            ci_s = f" CI95=[{ctx.parent_ci[0]:.3f},{ctx.parent_ci[1]:.3f}]" if ctx.parent_ci else ""
            lines.append(f"\nParent success_rate = {ctx.parent_score:.4f}{ci_s}")
        if ctx.best_score is not None:
            lines.append(f"Best-so-far success_rate = {ctx.best_score:.4f}")
        lines.append("\nReturn a JSON object only — no prose, no markdown.")
        return "\n".join(lines)
