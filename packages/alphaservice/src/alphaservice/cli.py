"""The `alpha` command-line entry point.

Phase 0 ships the commands that prove the runtime layer works end to end:
`version`, `doctor`, and `dev scratch-run`. The research verbs — `data pull`,
`features build`, `train`, `backtest`, `run`, `report`, `runs` — arrive with
their pipelines in later phases (§15).
"""

from __future__ import annotations

from typing import Annotated, Final

import typer
from rich.console import Console
from rich.table import Table

import alphacore
import alphaservice
from alphacore.runtime import (
    ArtifactStore,
    configure_logging,
    hash_seed_is_pinned,
    logger,
    run_context,
    set_global_seed,
)
from alphacore.runtime.determinism import HASH_SEED_ENV_VAR, REQUIRED_HASH_SEED
from alphacore.runtime.manifest import capture_environment, git_revision

SCRATCH_SEED: Final[int] = 0
"""Seed used by `dev scratch-run`. Fixed so the throwaway run is itself
reproducible."""

console = Console()

app = typer.Typer(
    name="alpha",
    help=(
        "Alpha Lab — decide whether a claimed short-horizon edge is real. "
        "Research software, not investment advice."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

dev_app = typer.Typer(name="dev", help="Developer utilities. Not part of the research path.")
app.add_typer(dev_app)


@app.callback()
def main(
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", help="TRACE, DEBUG, INFO, WARNING, ERROR."),
    ] = "INFO",
) -> None:
    """Configure logging before any subcommand runs.

    Args:
        log_level: Minimum level emitted to the console.
    """
    configure_logging(log_level.upper())


@app.command()
def version() -> None:
    """Print component versions and the current code revision."""
    console.print(f"alphacore    {alphacore.__version__}")
    console.print(f"alphaservice {alphaservice.__version__}")
    console.print(f"revision     {git_revision() or 'unknown'}")


@app.command()
def doctor() -> None:
    """Check that this environment can produce reproducible runs.

    Reports every condition invariant I8 depends on. Exits non-zero if any of
    them fails, because a run started from a non-deterministic environment
    produces artifacts that cannot be reproduced — and saying so afterwards is
    too late.
    """
    env = capture_environment()
    store = ArtifactStore()

    table = Table(title="alpha doctor", show_lines=False)
    table.add_column("check")
    table.add_column("value")
    table.add_column("status")

    ok = True

    python_ok = env.python.startswith("3.12")
    ok &= python_ok
    table.add_row("python", env.python, "PASS" if python_ok else "FAIL — 3.12 required")

    hash_ok = hash_seed_is_pinned()
    ok &= hash_ok
    table.add_row(
        HASH_SEED_ENV_VAR,
        "0" if hash_ok else "unset or non-zero",
        "PASS" if hash_ok else f"FAIL — set {HASH_SEED_ENV_VAR}={REQUIRED_HASH_SEED}",
    )

    table.add_row("platform", env.platform, "INFO")
    table.add_row("host", env.host, "INFO")
    table.add_row("revision", git_revision() or "unknown", "INFO")
    table.add_row("artifact store", store.root.as_posix(), "INFO")
    table.add_row("runs on disk", str(len(store.list_runs())), "INFO")
    for name, ver in env.packages.items():
        table.add_row(f"package {name}", ver, "INFO")

    console.print(table)
    if not ok:
        raise typer.Exit(code=1)


@dev_app.command("scratch-run")
def scratch_run(
    name: Annotated[
        str, typer.Option("--name", help="Experiment name recorded in the manifest.")
    ] = "scratch",
) -> None:
    """Create a throwaway run directory with a valid manifest.

    Exercises the full artifact lifecycle — run id, manifest, log sink, phase
    timings, immutable write, terminal status — without touching any market
    data. This is the Phase 0 exit check.

    Args:
        name: Experiment name recorded in the manifest.
    """
    config = {"meta": {"name": name, "seed": SCRATCH_SEED}, "phase": 0}
    set_global_seed(SCRATCH_SEED)

    with run_context(name=name, seed=SCRATCH_SEED, config=config) as (run, timings):
        with timings.phase("scratch"):
            logger.bind(phase="scratch").info("writing scratch artifact")
            run.write_json("scratch.json", {"hello": "alpha lab"})
        run.write_text("config.resolved.yaml", f"meta:\n  name: {name}\n  seed: {SCRATCH_SEED}\n")

    console.print(f"[bold]run created[/bold] {run.path.as_posix()}")
    manifest = run.read_manifest()
    console.print(f"status      {manifest.status.value}")
    console.print(f"config_hash {manifest.config_hash}")
    console.print(f"duration    {manifest.duration_s:.3f}s" if manifest.duration_s else "")
