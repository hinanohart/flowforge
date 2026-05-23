# Headless local-Qwen mutation backend recipe

This document is FlowForge's honest claim #3 deliverable: a reproducible
recipe for running the **local Qwen-Coder** mutation backend on a single
24 GB consumer GPU, completely offline once weights are cached.

Scope (v0.1.0a1):
- Runs the mutation LLM **locally**, no HF Inference API calls during S3.
- Targets the open-weight Qwen-Coder family on Hugging Face.
- The exact model id is **not hard-coded** — it is selected at bootstrap
  time from a top-downloads listing and written to
  `.flowforge/qwen_candidates.json`; the CLI's `_read_qwen_candidate`
  helper picks the first entry.

## 1. Hardware budget

| Component | Minimum | Comfortable |
|---|---|---|
| GPU VRAM | 24 GB (RTX 3090 / 4090 / A5000) | 48 GB+ |
| System RAM | 32 GB | 64 GB |
| Disk | ~20 GB free in `~/.cache/huggingface` | 100 GB+ |
| Driver | CUDA 12.1 or newer | matching driver/cuDNN |

If `nvidia-smi` is absent, `scripts/bootstrap.sh` writes
`.flowforge/HITL_REQUIRED` and exits cleanly; the local backend is not
attempted on CPU-only hosts.

## 2. Install the LLM extras

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[llm,evolve,dev]"
```

`transformers`, `accelerate`, and `huggingface_hub` are pulled in by the
`llm` extra. `torch` is **not** pinned by FlowForge; install the build that
matches your CUDA driver before the `pip install -e .` line.

## 3. Bootstrap discovery (chooses the Qwen variant)

`scripts/bootstrap.sh` queries the HF Hub for `author=Qwen` and
`search=Coder`, sorts by downloads, and writes the top candidates to
`.flowforge/qwen_candidates.json`:

```bash
bash scripts/bootstrap.sh
cat .flowforge/qwen_candidates.json   # list of {"model_id": "Qwen/..."} entries
```

This avoids hard-coding a model version that may be deprecated. If you
prefer a specific version, edit the file by hand; the CLI reads
`data[0]["model_id"]` (or `data[0]` if the entry is a bare string) and
falls back to `Qwen/Qwen2.5-Coder-32B-Instruct` if the file is absent.

## 4. Run S3 with the local mutator

```bash
flowforge auto --session-bound --mutator local --gens 30 --pop 8
```

Behaviour:
- At construction time, `LocalQwenClient` is **lazy** — no weights load
  until the first mutation call.
- `LocalQwenClient._ensure_model()` raises `RuntimeError` if CUDA is
  unavailable. The router does **not** silently fall back to random;
  S3 requires a working local client by contract.
- Each generation calls `complete_json(SYSTEM_PROMPT, user_prompt)`,
  which decodes a JSON object out of the model's reply via
  `flowforge.mutate._json_extract.parse_first_json_block`.
- Malformed JSON or out-of-range coefficients are clamped silently and
  logged at INFO level; the parent is preserved if mutation fails.

## 5. Memory tuning for 24 GB cards

The default load uses `torch_dtype=float16` and `device_map="auto"`, which
fits a 7 B / 14 B Coder model on 24 GB. For 32 B you need either:

- 4-bit AWQ weights (community quantisations published on the Hub), or
- 2× 24 GB GPUs with `device_map="balanced"`, or
- offload to CPU via `device_map={"": "auto"}` + `bitsandbytes` 4-bit.

FlowForge does not bundle quantisation kernels; install `bitsandbytes` /
`auto-gptq` / `autoawq` separately if needed and patch `LocalQwenClient`
to pass the right kwargs.

## 6. Sanity check (no GPU required)

```bash
python -c "
from flowforge.mutate import LocalQwenClient
c = LocalQwenClient(model_id='Qwen/Qwen2.5-Coder-7B-Instruct')
print('constructor OK; model loads lazily on first call')
"
```

This must print the success line on CPU-only hosts — the constructor must
not import torch. If it raises, the lazy-loading invariant is broken;
file an issue.

## 7. What this recipe does **not** guarantee

- It does **not** ship a tested pipeline that reproduces a specific
  published number; v0.1.0a1 is Phase-0 wiring only.
- The eval harness uses `StubEnv` unless real LIBERO is installed
  separately; see `flowforge.bench.libero_adapter.make_env`.
- The 30-gen evolve run is **not** part of v0.1.0a1 and the framework
  does **not** claim to be more efficient than RL fine-tuning at
  convergence — the goal is sample-efficient zero-grad improvement
  during inference-time only.
