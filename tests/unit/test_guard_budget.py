"""Tests for budget guards."""

import time

import pytest

from flowforge.guard.budget import BudgetExceeded, CallCounter, WallclockTracker, time_budget


def test_time_budget_normal_completion():
    with time_budget(2.0):
        x = sum(range(100))
        assert x == 4950


def test_time_budget_exceeded():
    with pytest.raises(BudgetExceeded):
        with time_budget(0.1):
            time.sleep(0.5)


def test_call_counter_bump():
    cc = CallCounter(max_calls=3, label="infer")
    cc.bump()
    cc.bump()
    cc.bump()
    with pytest.raises(BudgetExceeded):
        cc.bump()


def test_call_counter_invalid_max():
    with pytest.raises(ValueError):
        CallCounter(max_calls=0)


def test_wallclock_tracker_remaining():
    t = WallclockTracker(max_seconds=10.0)
    assert t.remaining() > 9.0


def test_wallclock_tracker_exceeded():
    t = WallclockTracker(max_seconds=0.05)
    time.sleep(0.1)
    with pytest.raises(BudgetExceeded):
        t.check()


def test_time_budget_zero_invalid():
    with pytest.raises(ValueError):
        with time_budget(0):
            pass
