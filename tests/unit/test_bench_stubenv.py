"""Tests for the stub env / episode runner."""

import numpy as np

from flowforge.bench.episode_runner import RunnerConfig, run_episode
from flowforge.bench.libero_adapter import LiberoConfig, StubEnv, make_env


def test_stub_env_reset_returns_obs():
    env = StubEnv(LiberoConfig())
    obs = env.reset(seed=0)
    assert obs.shape == (LiberoConfig().obs_dim,)


def test_stub_env_step_terminates():
    env = StubEnv(LiberoConfig())
    env.reset(seed=0)
    done = False
    n = 0
    while not done and n < 100:
        _, _, done, info = env.step(np.zeros(4))
        n += 1
    assert done
    assert "success" in info


def test_make_env_returns_stub_when_libero_missing():
    env = make_env(LiberoConfig(), success_prob_hint=0.5)
    assert isinstance(env, StubEnv)


def test_episode_runner_runs():
    env = StubEnv(LiberoConfig(), success_prob=0.0)

    class DummyPolicy:
        def reset(self):
            pass

        def infer(self, obs):
            return {"actions": np.zeros(4)}

    res = run_episode(
        env, DummyPolicy(), seed=0, task="libero_spatial", cfg=RunnerConfig(max_steps=40)
    )
    assert res.task == "libero_spatial"
    assert res.n_steps >= 1
