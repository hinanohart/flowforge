"""State machine orchestrator (S0 -> S7).

Session-bound execution model:
  * The orchestrator runs *foreground* under `flowforge auto --session-bound`.
  * After each state transition, state.json is fsync'd so a /compact or VM
    idle-out can resume from the last completed state without re-running it.
  * HITL flag short-circuits everything until a human removes it.

This is deliberately *not* a real-time scheduler — each `step()` returns
control to the caller after one logical unit of work (one state transition
for S0/S6/S7, one generation for S3, one trial-batch for S2).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from flowforge._types import State
from flowforge.bench.fitness import make_eval_fn
from flowforge.evolve import EvolveConfig, EvolveLoop
from flowforge.loop import checkpoint

log = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    project_root: Path
    n_generations: int = 30
    population_size: int = 8
    eval_tasks: tuple[str, ...] = ("libero_spatial",)
    eval_seeds: tuple[int, ...] = (0, 1, 2)
    rng_seed: int = 42
    optuna_n_trials: int = 100
    random_grid_n_trials: int = 100
    mutator: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    eval_fn: Callable[[dict[str, Any]], Any] | None = None  # injectable for tests


class Orchestrator:
    """Drives the 8-state machine to completion or HITL."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.state: dict[str, Any] = (
            checkpoint.load(config.project_root) or checkpoint.initial_state()
        )
        self.history: list[dict[str, Any]] = []

    # ---------- public API -----------------------------------------------

    def save(self) -> None:
        checkpoint.update_wallclock(self.state)
        checkpoint.save(self.config.project_root, self.state)

    def step(self) -> str:
        """Advance the machine by one logical unit. Returns the new state."""
        if checkpoint.hitl_required(self.config.project_root, self.state):
            self.state["current"] = State.HITL.value
            self.save()
            return self.state["current"]
        if checkpoint.hard_cap_exceeded(self.state):
            log.warning("42-day hard cap exceeded; jumping to S5_stats_report")
            self.state["current"] = State.S5_STATS_REPORT.value
            self.save()
            return self.state["current"]

        cur = self.state["current"]
        try:
            handler = self._dispatch_table()[cur]
        except KeyError:
            log.error("unknown state %r; emitting HITL", cur)
            checkpoint.write_hitl(self.config.project_root, f"unknown state {cur!r}", self.state)
            self.save()
            return State.HITL.value

        handler()
        self.save()
        return self.state["current"]

    def run_to_completion(self, max_steps: int = 200) -> str:
        """Loop step() until DONE / HITL / hard cap. Bounded for safety."""
        for _ in range(max_steps):
            current = self.step()
            if current in {State.DONE.value, State.HITL.value}:
                return current
        log.warning("max_steps=%d exhausted before DONE; treating as session-bound exit", max_steps)
        return self.state["current"]

    # ---------- state handlers -------------------------------------------

    def _dispatch_table(self) -> dict[str, Callable[[], None]]:
        return {
            State.S0_INIT.value: self._s0_init,
            State.S1_BASELINE_PI0_LIBERO.value: self._s1_baseline,
            State.S2_BASELINE_PARAMETRIC.value: self._s2_baseline_parametric,
            State.S3_EVOLVE_MAIN.value: self._s3_evolve_main,
            State.S5_STATS_REPORT.value: self._s5_stats_report,
            State.S6_DOC_TEST.value: self._s6_doc_test,
            State.S7_RELEASE.value: self._s7_release,
            State.DONE.value: lambda: None,
            State.HITL.value: lambda: None,
        }

    def _s0_init(self) -> None:
        log.info("S0_init: bootstrap presence check")
        bootstrap_done = self.config.project_root / ".flowforge" / "bootstrap_done"
        if not bootstrap_done.is_file():
            checkpoint.write_hitl(
                self.config.project_root,
                "scripts/bootstrap.sh has not been run (missing .flowforge/bootstrap_done)",
                self.state,
            )
            return
        self.state["current"] = State.S1_BASELINE_PI0_LIBERO.value

    def _eval_fn(self):
        return self.config.eval_fn or make_eval_fn(
            list(self.config.eval_tasks), list(self.config.eval_seeds)
        )

    def _s1_baseline(self) -> None:
        log.info("S1_baseline_pi0_libero: zero-shot eval baseline")
        report = self._eval_fn()(
            {
                "sched_template": "polynomial",
                "sched_coefs": {"a0": 1.0, "a1": 0.0, "a2": 0.0, "a3": 0.0},
                "reward_template": "potential",
                "reward_coefs": {"gamma": 0.99, "scale": 0.1},
            }
        )
        self.state["baseline_zero_shot"] = {
            "success_rate": getattr(report, "success_rate", 0.0),
            "ci95_low": getattr(report, "success_rate_ci95_low", 0.0),
            "ci95_high": getattr(report, "success_rate_ci95_high", 0.0),
        }
        self.state["current"] = State.S2_BASELINE_PARAMETRIC.value

    def _s2_baseline_parametric(self) -> None:
        log.info("S2_baseline_parametric: Optuna + random grid")
        from flowforge.baseline import OptunaConfig, RandomGridConfig, run_optuna, run_random_grid

        ef = self._eval_fn()
        rg = run_random_grid(
            ef,
            RandomGridConfig(n_samples=self.config.random_grid_n_trials, seed=self.config.rng_seed),
        )
        self.state["random_grid_best"] = float(rg["best_score"])
        try:
            opt = run_optuna(
                ef,
                sched_template="polynomial",
                reward_template="potential",
                config=OptunaConfig(
                    n_trials=self.config.optuna_n_trials, sampler_seed=self.config.rng_seed
                ),
            )
            self.state["optuna_best"] = float(opt["best_score"])
        except ImportError:
            log.warning("optuna not installed; skipping parametric baseline")
            self.state["optuna_best"] = None
            self.state["optuna_skipped"] = True
        self.state["current"] = State.S3_EVOLVE_MAIN.value

    def _s3_evolve_main(self) -> None:
        log.info("S3_evolve_main: ShinkaEvolve (function-template space)")
        ef = self._eval_fn()
        loop = EvolveLoop(
            EvolveConfig(
                n_generations=self.config.n_generations,
                population_size=self.config.population_size,
                seed=self.config.rng_seed,
            ),
            eval_fn=ef,
        )
        result = loop.run(mutator=self.config.mutator)
        self.state["evolve_best"] = result["best"]
        self.state["evolve_history"] = result["history"]
        checkpoint.write_per_gen_checkpoint(
            self.config.project_root, self.config.n_generations - 1, result["history"][-1]
        )
        self.state["current"] = State.S5_STATS_REPORT.value

    def _s5_stats_report(self) -> None:
        log.info("S5_stats_report: bootstrap CI + Δ table")
        ef = self._eval_fn()
        best = self.state.get("evolve_best") or {
            "sched_template": "polynomial",
            "sched_coefs": {"a0": 1.0, "a1": 0.0, "a2": 0.0, "a3": 0.0},
            "reward_template": "potential",
            "reward_coefs": {"gamma": 0.99, "scale": 0.1},
        }
        report = ef(best)
        baseline_p = self.state.get("baseline_zero_shot", {}).get("success_rate", 0.0)
        evolve_p = float(getattr(report, "success_rate", 0.0))
        delta = evolve_p - float(baseline_p)
        self.state["final_report"] = {
            "baseline_success_rate": baseline_p,
            "evolve_success_rate": evolve_p,
            "delta": delta,
            "evolve_ci95": [
                getattr(report, "success_rate_ci95_low", 0.0),
                getattr(report, "success_rate_ci95_high", 0.0),
            ],
        }
        self.state["current"] = State.S6_DOC_TEST.value

    def _s6_doc_test(self) -> None:
        log.info("S6_doc_test: doc + tests already shipped; sanity-check files exist")
        for rel in ["README.md", "LICENSE", "pyproject.toml", "tests"]:
            if not (self.config.project_root / rel).exists():
                checkpoint.write_hitl(
                    self.config.project_root, f"S6 sanity check failed: {rel} missing", self.state
                )
                return
        self.state["current"] = State.S7_RELEASE.value

    def _s7_release(self) -> None:
        log.info("S7_release: requires user-driven `gh repo create` & tag — leaving DONE marker")
        # The actual `gh` calls are run by scripts/release.sh under user authority,
        # not by the orchestrator, to keep token usage explicit.
        (self.config.project_root / ".flowforge" / "release_ready").write_text(
            time.strftime("%Y-%m-%dT%H:%M:%S%z")
        )
        self.state["current"] = State.DONE.value
