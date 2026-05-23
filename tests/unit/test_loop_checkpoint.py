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


def test_update_wallclock_accumulates():
    st = checkpoint.initial_state()
    st["last_step_unix"] = int(time.time()) - 5
    checkpoint.update_wallclock(st)
    assert st["total_wallclock_s"] >= 4.0


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
