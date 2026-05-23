"""End-to-end smoke test: bootstrap-less orchestrator drives 1 mini-gen.

Runs entirely on stubs. Verifies that the wiring between
orchestrator/evolve/bench/baseline/checkpoint holds.
"""

from pathlib import Path

from flowforge._types import FitnessReport
from flowforge.loop import Orchestrator, OrchestratorConfig


def genome_dependent_eval(g):
    """Returns a score that depends on the genome so that Δ > 0 is observable."""
    a0 = float(g.get("sched_coefs", {}).get("a0", 1.0))
    sr = max(0.0, min(1.0, 0.05 + 0.08 * a0))
    return FitnessReport(
        candidate_id="x",
        n_episodes=2,
        success_rate=sr,
        success_rate_ci95_low=max(0.0, sr - 0.05),
        success_rate_ci95_high=min(1.0, sr + 0.05),
        mean_reward=0.0,
        wall_time_s=0.0,
    )


def test_smoke_end_to_end(tmp_path: Path):
    (tmp_path / ".flowforge").mkdir()
    (tmp_path / ".flowforge" / "bootstrap_done").write_text("ok")
    (tmp_path / "README.md").write_text("# smoke")
    (tmp_path / "LICENSE").write_text("apache")
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "tests").mkdir()

    o = Orchestrator(
        OrchestratorConfig(
            project_root=tmp_path,
            n_generations=1,
            population_size=2,
            optuna_n_trials=2,
            random_grid_n_trials=2,
            eval_fn=genome_dependent_eval,
        )
    )
    final = o.run_to_completion(max_steps=20)
    assert final == "done", o.state
    # Final report has expected keys + records baseline vs evolve numerics.
    rep = o.state.get("final_report")
    assert rep is not None
    assert "evolve_success_rate" in rep
    assert "baseline_success_rate" in rep
    assert "delta" in rep
    # Δ may be zero or positive but must be a finite number.
    import math

    assert math.isfinite(rep["delta"])
