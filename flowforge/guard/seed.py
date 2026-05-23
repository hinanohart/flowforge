"""Deterministic seeding across numpy / random / (optional) torch."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed Python random, numpy, and (if importable) torch.

    Does not seed CUDA-deterministic algorithms (training is out of v0.1.0 scope).
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
