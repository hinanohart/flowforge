"""Tests for global seeding."""

import random

import numpy as np
import pytest

from flowforge.guard import seed as seedmod


def test_seed_makes_random_deterministic():
    seedmod.set_global_seed(123)
    a = random.random()
    seedmod.set_global_seed(123)
    b = random.random()
    assert a == b


def test_seed_makes_numpy_deterministic():
    seedmod.set_global_seed(7)
    a = np.random.random()
    seedmod.set_global_seed(7)
    b = np.random.random()
    assert a == b


def test_seed_type_validation():
    with pytest.raises(TypeError):
        seedmod.set_global_seed("not-int")  # type: ignore[arg-type]
