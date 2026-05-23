"""Fixed templates for FlowForge evolution search space.

Schedules: polynomial / piecewise / cosine
Rewards:   potential / dense / sparse
"""

from flowforge.evolve.templates import reward_shaping, sampling_schedule

__all__ = ["sampling_schedule", "reward_shaping"]
