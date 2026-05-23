# FlowForge roadmap

These versions are **not committed**. They live here so that v0.1.0 scope
stays frozen. None of them will start until Phase 0 (v0.1.0) ships and we
have read the issue tracker.

## v0.1.1 — honest-marketing patch (post v0.1.0a1)

- Wire real LIBERO loader in `flowforge.bench.libero_adapter.make_env`
  (currently raises `NotImplementedError` if LIBERO is importable).
- Add mock-based unit tests for `flowforge.mutate.local_qwen` to bring
  coverage above 70% on that file.
- Add `stub_policy` mock path test for `flowforge.policy.openpi_adapter`.
- Narrow `except Exception` at `flowforge.bench.fitness` and
  `flowforge.evolve.shinka_wrapper` to concrete error classes.

## v0.2 — V-JEPA2 latent eval (+2 mo, conditional)

Use V-JEPA2 latent rollout to accelerate `bench` 10×, candidate policy
latent-space pruning.

## v0.3 — recurrentlens SAE integration (+3 mo, conditional)

Use SAE to diff candidate policies before/after evolution; link to
[hinanohart/recurrentlens](https://github.com/hinanohart/recurrentlens).

## v0.4 — policy-agnostic evolver (+2 mo, conditional)

Move the harness off openpi pi0.5 onto SmolVLA / OpenVLA so users can
plug their own flow-matching VLA.

## v0.5 — FoldDreamer (conditional, may stay backlog)

Only if both Dreamer4 has an official reproduction and AlphaFold3 has
a commercial-OK license.

## Out-of-scope (permanent)

- Pretrained checkpoint releases (Phase 0 wires harness only).
- "First ever" / "state-of-the-art" claims of any kind.
- Automatic GitHub publishing without user-driven `gh` invocation.
