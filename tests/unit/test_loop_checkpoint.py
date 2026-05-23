"""Tests for state.json persistence + HITL flag handling."""

import json
import time
from pathlib import Path

from flowforge.loop import checkpoint


def test_initial_state_shape():
    st = checkpoint.initial_state()
    assert st["current"] == "S0_init"
    assert st["version"] == checkpoint.SCHEMA_VERSION


def test_save_then_load(tmp_path: Path):
    st = checkpoint.initial_state()
    st["custom"] = 42
    checkpoint.save(tmp_path, st)
    loaded = checkpoint.load(tmp_path)
    assert loaded["custom"] == 42  # type: ignore[index]


def test_load_missing_returns_none(tmp_path: Path):
    assert checkpoint.load(tmp_path) is None


def test_load_corrupt_returns_none(tmp_path: Path):
    p = checkpoint.state_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not json")
    assert checkpoint.load(tmp_path) is None


def test_mark_step_bounds_wallclock_to_step_duration():
    """Only step-bounded compute time counts; idle gaps must NOT accumulate."""
    st = checkpoint.initial_state()
    st["total_wallclock_s"] = 0.0
    # Simulate a long idle gap before the step begins.
    st["last_step_unix"] = int(time.time()) - 3600
    checkpoint.mark_step_start(st)
    time.sleep(0.05)
    checkpoint.mark_step_end(st)
    # The accumulated time must be ~step duration (<1 s), NOT the 3600s idle gap.
    assert st["total_wallclock_s"] < 2.0


def test_update_wallclock_legacy_is_pin_only():
    """update_wallclock no longer accumulates; it only pins last_step_unix."""
    st = checkpoint.initial_state()
    st["last_step_unix"] = int(time.time()) - 5
    st["total_wallclock_s"] = 7.0
    checkpoint.update_wallclock(st)
    assert st["total_wallclock_s"] == 7.0  # no accumulation
    assert st["last_step_unix"] >= int(time.time()) - 1


def test_hard_cap_not_exceeded_fresh():
    st = checkpoint.initial_state()
    assert not checkpoint.hard_cap_exceeded(st)


def test_hard_cap_exceeded_after_42d():
    st = checkpoint.initial_state()
    st["total_wallclock_s"] = checkpoint.HARD_CAP_SECONDS + 1
    assert checkpoint.hard_cap_exceeded(st)


def test_hitl_flag_detected(tmp_path: Path):
    st = checkpoint.initial_state()
    checkpoint.write_hitl(tmp_path, "test reason", st)
    assert checkpoint.hitl_required(tmp_path, st)
    assert checkpoint.hitl_flag_path(tmp_path).is_file()


def test_per_gen_checkpoint(tmp_path: Path):
    p = checkpoint.write_per_gen_checkpoint(tmp_path, 3, {"best": "x"})
    data = json.loads(p.read_text())
    assert data["best"] == "x"
