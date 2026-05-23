"""FlowForge CLI: `flowforge run | auto | status`."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import click

from flowforge import __version__
from flowforge.loop import Orchestrator, OrchestratorConfig, checkpoint


def _default_root() -> Path:
    """Resolve CWD lazily — avoids stale module-load binding."""
    return Path.cwd()


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _build_mutator(
    mode: str, rng_seed: int, project_root: Path
) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    """Construct a mutator callable for the orchestrator, or None.

    Modes:
      * "none": no LLM mutation (orchestrator emits a warning at S3).
      * "random": Router with random backend (no LLM, deterministic baseline).
      * "hf":    Router with HfApiClient (S0-S2 LLM mutation; HF_API_TOKEN required).
      * "local": Router with LocalQwenClient (S3 production path; CUDA required).
    """
    if mode == "none":
        return None

    from flowforge.mutate import HfApiClient, LocalQwenClient, MutateContext, Router

    local_factory = None
    hf_factory = None
    if mode == "local":
        model_id = _read_qwen_candidate(project_root)
        local_factory = lambda: LocalQwenClient(model_id=model_id)
    elif mode == "hf":
        hf_factory = lambda: HfApiClient()
    elif mode != "random":
        raise click.BadParameter(f"unknown mutator mode: {mode!r}")

    router = Router(
        rng_seed=rng_seed,
        local_client_factory=local_factory,
        hf_client_factory=hf_factory,
    )

    def _mutate(parent: dict[str, Any]) -> dict[str, Any]:
        return router.mutate(parent, MutateContext(state="S3_evolve_main"))

    return _mutate


def _read_qwen_candidate(project_root: Path) -> str:
    """Read the top Qwen candidate from `.flowforge/qwen_candidates.json`.

    Falls back to a sensible default if the bootstrap-produced file is missing.
    """
    p = project_root / ".flowforge" / "qwen_candidates.json"
    if p.is_file():
        try:
            data = json.loads(p.read_text())
            if isinstance(data, list) and data:
                top = data[0]
                if isinstance(top, dict) and isinstance(top.get("model_id"), str):
                    return top["model_id"]
                if isinstance(top, str):
                    return top
        except (json.JSONDecodeError, OSError):
            pass
    return "Qwen/Qwen2.5-Coder-32B-Instruct"


@click.group(invoke_without_command=False)
@click.version_option(version=__version__, prog_name="flowforge")
def cli():
    """FlowForge — LLM-driven evolutionary search for flow-matching VLA."""


@cli.command()
@click.option("--task", default="libero_spatial", show_default=True)
@click.option("--gens", default=2, show_default=True, type=int)
@click.option("--pop", default=4, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--dry-run/--no-dry-run", default=False)
@click.option("--verbose/--quiet", default=False)
def run(task: str, gens: int, pop: int, seed: int, dry_run: bool, verbose: bool):
    """Run a short evolve loop (smoke test)."""
    _logging(verbose)
    from flowforge.bench.fitness import make_eval_fn
    from flowforge.evolve import EvolveConfig, EvolveLoop

    if dry_run:
        click.echo(f"[dry-run] task={task} gens={gens} pop={pop} seed={seed}")
        return

    eval_fn = make_eval_fn([task], [seed, seed + 1, seed + 2])
    loop = EvolveLoop(EvolveConfig(n_generations=gens, population_size=pop, seed=seed), eval_fn)
    result = loop.run()
    best = result["best"]
    last = result["history"][-1]
    click.echo(json.dumps({"best_score": last["best_score"], "best_genome": best}, indent=2))


@cli.command(name="auto")
@click.option(
    "--session-bound/--background",
    default=True,
    help="session-bound runs foreground; background is not supported in v0.1.0 (WSL2 vmIdleTimeout).",
)
@click.option(
    "--max-steps",
    default=20,
    type=int,
    show_default=True,
    help="Max state transitions per invocation. /compact-safe.",
)
@click.option("--root", default=None, type=click.Path(file_okay=False))
@click.option("--gens", default=30, show_default=True, type=int)
@click.option("--pop", default=8, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option(
    "--mutator",
    type=click.Choice(["none", "random", "hf", "local"], case_sensitive=False),
    default="none",
    show_default=True,
    help="Mutator backend. 'none' = no LLM (degrades to random+elitism with warning). "
    "'random' = Router with random backend. 'hf' = HF Inference API (HF_API_TOKEN). "
    "'local' = local Qwen-Coder (CUDA required; reads .flowforge/qwen_candidates.json).",
)
@click.option("--verbose/--quiet", default=False)
def auto(
    session_bound: bool,
    max_steps: int,
    root: str | None,
    gens: int,
    pop: int,
    seed: int,
    mutator: str,
    verbose: bool,
):
    """Drive the S0->S7 state machine until DONE / HITL / max_steps."""
    _logging(verbose)
    if not session_bound:
        click.echo(
            "ERROR: --background unsupported in v0.1.0. WSL2 vmIdleTimeout=60s would kill the VM. "
            "Use scripts/cron_resume.sh on bare Linux for unattended runs.",
            err=True,
        )
        sys.exit(2)

    root_path = Path(root) if root else _default_root()
    mutator_fn = _build_mutator(mutator.lower(), seed, root_path)

    orch = Orchestrator(
        OrchestratorConfig(
            project_root=root_path,
            n_generations=gens,
            population_size=pop,
            rng_seed=seed,
            mutator=mutator_fn,
        )
    )
    final = orch.run_to_completion(max_steps=max_steps)
    click.echo(
        json.dumps(
            {
                "final_state": final,
                "wallclock_s": orch.state.get("total_wallclock_s", 0.0),
                "mutator_mode": mutator.lower(),
                "mutator_active": orch.state.get("mutator_active", False),
            },
            indent=2,
        )
    )


@cli.command(name="status")
@click.option("--root", default=None, type=click.Path(file_okay=False))
def status(root: str | None):
    """Print the current state.json content."""
    root_path = Path(root) if root else _default_root()
    st = checkpoint.load(root_path)
    if st is None:
        click.echo("no state.json yet — run `flowforge auto` first.")
        return
    keys = [
        "current",
        "started_at_unix",
        "total_wallclock_s",
        "generation",
        "baseline_zero_shot",
        "optuna_best",
        "random_grid_best",
        "mutator_active",
        "hitl_required",
    ]
    out = {k: st.get(k) for k in keys if k in st}
    click.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    cli()
