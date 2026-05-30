# FlowForge

LLM-driven evolutionary search over flow-matching VLA sampling schedules and reward shaping; ships a LIBERO-style eval harness with bootstrap CI 95%. v0.1.0a2 is Phase-0 wiring (audit-patched from v0.1.0a1) — the 30-generation evolve run is **not** part of this release.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

## Honest claims (v0.1.0)

FlowForge offers three things, no more:

1. **Sampling-schedule + reward-shaping evolver** for flow-matching VLA policies (openpi pi0.5 / LIBERO). Mutations operate on *function templates* (polynomial / piecewise / cosine for schedules; potential / dense / sparse for rewards), not on policy internals.
2. **Reusable LIBERO eval harness with bootstrap CI**, decoupled from the policy class. Other VLA codebases can import `flowforge.bench` alone.
3. **Headless local-Qwen mutation backend recipe** for ShinkaEvolve on 24 GB consumer GPU.

What FlowForge does **not** claim:

- Not a new VLA model. Builds on Physical-Intelligence/openpi (MIT).
- Not faster than RL fine-tuning at convergence. Aims for sample-efficient *zero-grad* improvement during inference-time.
- Not autonomous. The orchestrator runs in session-bound mode; WSL2 VMs idle out at 60 s by default ([Microsoft docs](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)), so long runs require a wakeful host or systemd unit on bare Linux.

## Status

Alpha. Phase 0 (this release) ships:
- Repo skeleton + 8 state machine (S0–S7)
- AST safety guard, bootstrap CI stats, NaN detector
- Optuna + Random-grid baselines (parametric)
- Built-in evolutionary loop over the function-template space (ShinkaEvolve full integration deferred to v0.2)
- LIBERO adapter via openpi pi0.5 checkpoint
- Local-Qwen mutation router (HF API fallback for bootstrap only)
- CLI (`flowforge run | auto | status`) with checkpoint resume

Training run (S3 30-gen evolve) is **not** included in this release. The framework is designed to run unattended for up to 42 days via cron / systemd; see [docs/cron_resume.md](docs/cron_resume.md).

## Install

```bash
git clone https://github.com/hinanohart/flowforge
cd flowforge
pip install -e ".[evolve,llm,dev]"
bash scripts/bootstrap.sh           # downloads openpi pi0.5 checkpoint + Qwen weights
```

GPU note: bootstrap detects `nvidia-smi`; if absent, creates `.flowforge/HITL_REQUIRED` and exits cleanly. Skeleton + unit tests work on CPU-only hosts.

## Quick start

```bash
flowforge run --task libero_spatial --gens 1 --pop 4 --dry-run
flowforge auto --session-bound --mutator none   # default: warns + random+elitism
flowforge auto --session-bound --mutator local  # LLM-driven (CUDA + Qwen-Coder)
flowforge auto --session-bound --mutator hf     # HF Inference API (HF_API_TOKEN)
flowforge status                    # current state + ETA
```

The `--mutator` flag selects the mutation backend. `none` is the default and
the orchestrator will warn at S3 entry that the evolve loop is degrading to
random+elitism (no LLM). Use `local` for the production path (see
[docs/local_qwen_setup.md](docs/local_qwen_setup.md)), or `hf` for
quick experiments without GPU. `mutator_active` is recorded in `state.json`
and the final report so every run is auditable.

## Architecture

8-state machine, scope frozen for v0.1.0:

| State | Purpose | Wall-clock |
|---|---|---|
| S0_init | bootstrap (openpi DL, Qwen fetch, sandbox check) | 30–90 min |
| S1_baseline_pi0_libero | π0.5 zero-shot eval, bootstrap CI 95% baseline | 8–16 h |
| S2_baseline_parametric | Optuna 100 trial + Random schedule grid 100 trial | 12–24 h |
| S3_evolve_main | Built-in evolve loop, 30 gen × N=8 × 3 task (ShinkaEvolve hook deferred to v0.2) | 10–20 d |
| S5_stats_report | re-run + bootstrap CI + curves + honest Δ report | 3–7 d |
| S6_doc_test | API doc + pytest cov ≥70% | 4–8 h |
| S7_release | tag + branch protection + MIT LICENSE | 1–2 h |

Hard cap: **42 days**. On overrun, S5 emits a `partial` report.

## Function templates (evolve search space)

Schedule templates (`flowforge/evolve/templates/sampling_schedule.py`):
- `polynomial(t, *coefs)` — degree ≤ 3
- `piecewise(t, *breakpoints, *values)` — k ≤ 4 segments
- `cosine(t, omega, phi, amp, dc)` — single-mode cosine over [0,1]

Reward templates (`flowforge/evolve/templates/reward_shaping.py`):
- `potential(s, s_next, gamma)` — Ng-style potential
- `dense(s, target, sigma)` — Gaussian shaping
- `sparse(s, threshold)` — binary thresholded

LLM mutates **coefficients only**; templates are fixed. This bounds the search space and prevents the LLM from breaking sandbox invariants.

## Roadmap

Future versions are not committed; they live in [docs/roadmap.md](docs/roadmap.md) and require Phase 0 completion sign-off before any work starts.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) — π0 / π0.5 policy + LIBERO support (MIT)
- [SakanaAI/ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) — evolutionary search framework (MIT)
- [LIBERO](https://libero-project.github.io/) — benchmark suite
- [Qwen team](https://huggingface.co/Qwen) — open-weight code LLMs
