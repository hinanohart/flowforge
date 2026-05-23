"""FlowForge CLI: `flowforge run | auto | status`."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from flowforge import __version__
from flowforge.loop import Orchestrator, OrchestratorConfig, checkpoint

DEFAULT_ROOT = Path.cwd()


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


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
@click.option(
    "--root", default=str(DEFAULT_ROOT), show_default=True, type=click.Path(file_okay=False)
)
@click.option("--gens", default=30, show_default=True, type=int)
@click.option("--pop", default=8, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--verbose/--quiet", default=False)
def auto(
    session_bound: bool, max_steps: int, root: str, gens: int, pop: int, seed: int, verbose: bool
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

    orch = Orchestrator(
        OrchestratorConfig(
            project_root=Path(root),
            n_generations=gens,
            population_size=pop,
            rng_seed=seed,
        )
    )
    final = orch.run_to_completion(max_steps=max_steps)
    click.echo(
        json.dumps(
            {"final_state": final, "wallclock_s": orch.state.get("total_wallclock_s", 0.0)},
            indent=2,
        )
    )


@cli.command(name="status")
@click.option(
    "--root", default=str(DEFAULT_ROOT), show_default=True, type=click.Path(file_okay=False)
)
def status(root: str):
    """Print the current state.json content."""
    st = checkpoint.load(Path(root))
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
        "hitl_required",
    ]
    out = {k: st.get(k) for k in keys if k in st}
    click.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    cli()
