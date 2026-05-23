"""Tests for the S0->S7 orchestrator."""

from pathlib import Path

from flowforge._types import FitnessReport
from flowforge.loop import Orchestrator, OrchestratorConfig, checkpoint


def fast_eval(g):
    return FitnessReport(
        candidate_id="x",
        n_episodes=1,
        success_rate=0.5,
        success_rate_ci95_low=0.4,
        success_rate_ci95_high=0.6,
        mean_reward=0.0,
        wall_time_s=0.0,
    )


def test_orchestrator_blocks_on_missing_bootstrap(tmp_path: Path):
    o = Orchestrator(OrchestratorConfig(project_root=tmp_path, eval_fn=fast_eval))
    o.step()
    assert checkpoint.hitl_flag_path(tmp_path).is_file()


def test_orchestrator_runs_to_done_when_bootstrap_done(tmp_path: Path):
    (tmp_path / ".flowforge").mkdir()
    (tmp_path / ".flowforge" / "bootstrap_done").write_text("ok")
    # README/LICENSE/pyproject/tests are required for S6 sanity check
    (tmp_path / "README.md").write_text("# t")
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
            eval_fn=fast_eval,
        )
    )
    final = o.run_to_completion(max_steps=20)
    assert final in {"done", "hitl_required"}, final


def test_orchestrator_hitl_short_circuits(tmp_path: Path):
    (tmp_path / ".flowforge").mkdir()
    (tmp_path / ".flowforge" / "bootstrap_done").write_text("ok")
    checkpoint.write_hitl(tmp_path, "manual stop", None)
    o = Orchestrator(OrchestratorConfig(project_root=tmp_path, eval_fn=fast_eval))
    final = o.step()
    assert final == "hitl_required"


def test_orchestrator_hard_cap_jumps_to_s5(tmp_path: Path):
    (tmp_path / ".flowforge").mkdir()
    (tmp_path / ".flowforge" / "bootstrap_done").write_text("ok")
    o = Orchestrator(OrchestratorConfig(project_root=tmp_path, eval_fn=fast_eval))
    o.state["total_wallclock_s"] = checkpoint.HARD_CAP_SECONDS + 1
    o.step()
    assert o.state["current"] in {"S5_stats_report", "S6_doc_test", "S7_release_v0_1_0", "done"}
