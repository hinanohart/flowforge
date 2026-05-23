"""Shared lightweight types for FlowForge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class State(str, Enum):
    S0_INIT = "S0_init"
    S1_BASELINE_PI0_LIBERO = "S1_baseline_pi0_libero"
    S2_BASELINE_PARAMETRIC = "S2_baseline_parametric"
    S3_EVOLVE_MAIN = "S3_evolve_main"
    S5_STATS_REPORT = "S5_stats_report"
    S6_DOC_TEST = "S6_doc_test"
    S7_RELEASE = "S7_release_v0_1_0"
    DONE = "done"
    HITL = "hitl_required"


@dataclass
class EpisodeResult:
    task: str
    seed: int
    success: bool
    reward: float
    n_steps: int
    wall_time_s: float


@dataclass
class FitnessReport:
    candidate_id: str
    n_episodes: int
    success_rate: float
    success_rate_ci95_low: float
    success_rate_ci95_high: float
    mean_reward: float
    wall_time_s: float
    failed: bool = False
    failure_reason: str = ""


@dataclass
class Candidate:
    candidate_id: str
    generation: int
    schedule_fn_src: str  # Python source for sampling schedule
    reward_fn_src: str  # Python source for reward shaping
    parent_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Population:
    generation: int
    candidates: list[Candidate]


SchedFn = Callable[[float], float]
RewardFn = Callable[[Any, Any], float]
