"""Checkpoint + state.json management.

state.json layout (forwards-compatible):
{
  "current": "S1_baseline_pi0_libero",
  "started_at_unix": 1748000000,
  "last_step_unix": 1748000600,
  "total_wallclock_s": 600.0,
  "generation": 0,
  "best_so_far": {...genome..., "_score": 0.42},
  "history": [...],
  "hitl_required": false,
  "version": 1
}
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
HARD_CAP_SECONDS = 42 * 24 * 3600  # 42 days


def state_path(project_root: str | os.PathLike) -> Path:
    return Path(project_root) / ".flowforge" / "state.json"


def hitl_flag_path(project_root: str | os.PathLike) -> Path:
    return Path(project_root) / ".flowforge" / "HITL_REQUIRED"


def load(project_root: str | os.PathLike) -> dict[str, Any] | None:
    p = state_path(project_root)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        log.error("state.json corrupt: %s", e)
        return None
    if data.get("version", 0) > SCHEMA_VERSION:
        raise RuntimeError(f"state.json schema {data.get('version')} > supported {SCHEMA_VERSION}")
    return data


def initial_state() -> dict[str, Any]:
    return {
        "current": "S0_init",
        "started_at_unix": int(time.time()),
        "last_step_unix": int(time.time()),
        "total_wallclock_s": 0.0,
        "generation": 0,
        "best_so_far": None,
        "history": [],
        "hitl_required": False,
        "version": SCHEMA_VERSION,
    }


def save(project_root: str | os.PathLike, state: dict[str, Any]) -> None:
    p = state_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(p)


def update_wallclock(state: dict[str, Any]) -> None:
    now = int(time.time())
    delta = max(0, now - state.get("last_step_unix", now))
    state["total_wallclock_s"] = float(state.get("total_wallclock_s", 0.0)) + float(delta)
    state["last_step_unix"] = now


def hard_cap_exceeded(state: dict[str, Any]) -> bool:
    return float(state.get("total_wallclock_s", 0.0)) > HARD_CAP_SECONDS


def hitl_required(project_root: str | os.PathLike, state: dict[str, Any]) -> bool:
    return state.get("hitl_required", False) or hitl_flag_path(project_root).is_file()


def write_hitl(
    project_root: str | os.PathLike, reason: str, state: dict[str, Any] | None = None
) -> None:
    p = hitl_flag_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n{reason}\n")
    if state is not None:
        state["hitl_required"] = True
        state["hitl_reason"] = reason


def write_per_gen_checkpoint(
    project_root: str | os.PathLike, generation: int, payload: dict[str, Any]
) -> Path:
    ck_dir = Path(project_root) / ".flowforge" / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    p = ck_dir / f"gen_{generation:04d}.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return p
