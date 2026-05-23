"""Tests for the mutation router."""

from flowforge.mutate.router import MutateContext, Router


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete_json(self, system, user):
        self.calls += 1
        return self.payload


def test_router_selects_random_when_no_clients():
    r = Router(rng_seed=0)
    ctx = MutateContext(state="S1_baseline_pi0_libero")
    assert r.select_backend(ctx.state) == "random"
    parent = {
        "sched_template": "polynomial",
        "sched_coefs": {"a0": 1.0},
        "reward_template": "potential",
        "reward_coefs": {},
    }
    child = r.mutate(parent, ctx)
    assert "sched_template" in child


def test_router_prefers_hf_in_s1_if_available():
    r = Router(rng_seed=0, hf_client_factory=lambda: FakeClient(_good_payload()))
    assert r.select_backend("S1_baseline_pi0_libero") == "hf_api"
    out = r.mutate(_good_payload(), MutateContext(state="S1_baseline_pi0_libero"))
    assert out["sched_template"] in {"polynomial", "piecewise", "cosine"}


def test_router_requires_local_in_s3():
    r = Router(rng_seed=0, hf_client_factory=lambda: FakeClient(_good_payload()))
    assert r.select_backend("S3_evolve_main") == "local_qwen"
    out = r.mutate(_good_payload(), MutateContext(state="S3_evolve_main"))
    # No local client -> random fallback
    assert "sched_template" in out


def test_router_falls_back_on_llm_exception():
    class Bomb:
        def complete_json(self, *a, **k):
            raise ValueError("bad json")

    r = Router(rng_seed=0, hf_client_factory=lambda: Bomb())
    out = r.mutate(_good_payload(), MutateContext(state="S1_baseline_pi0_libero"))
    assert "sched_coefs" in out


def test_router_clamps_proposal():
    bad = _good_payload()
    bad["sched_coefs"]["a0"] = 1e9
    r = Router(rng_seed=0, hf_client_factory=lambda: FakeClient(bad))
    out = r.mutate(_good_payload(), MutateContext(state="S1_baseline_pi0_libero"))
    assert out["sched_coefs"]["a0"] <= 10.0


def _good_payload():
    return {
        "sched_template": "polynomial",
        "sched_coefs": {"a0": 1.0, "a1": 0.0, "a2": 0.0, "a3": 0.0},
        "reward_template": "potential",
        "reward_coefs": {"gamma": 0.99, "scale": 0.1},
    }
