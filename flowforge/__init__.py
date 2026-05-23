"""FlowForge — LLM-driven evolutionary search for flow-matching VLA sampling.

Top-level package; see README.md for honest claims and scope.
"""

__version__ = "0.1.0a1"

from flowforge._types import Candidate, EpisodeResult, FitnessReport, Population, State

__all__ = [
    "__version__",
    "Candidate",
    "EpisodeResult",
    "FitnessReport",
    "Population",
    "State",
]
