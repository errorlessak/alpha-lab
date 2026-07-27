"""The `alpha` command-line entry point.

Phases 0 and 1 ship `version`, `doctor`, `dev scratch-run`, `data pull`,
`data inspect`, `data list`, and `config show`. The remaining research verbs —
`features build`, `train`, `backtest`, `run`, `report`, `runs` — arrive with
their pipelines in later phases (§15).

Heavy imports (`alphacore.data`, `alphacore.config`) happen inside the commands
that need them, so `alpha --help` and `alpha version` stay instant.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import typer
from rich.console import Console
from rich.table import Table

import alphacore
import alphaservice
from alphacore.errors import AlphaError
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

if TYPE_CHECKING:
    from alphacore.config import AlphaConfig

SCRATCH_SEED: Final[int] = 0
"""Seed used by `dev scratch-run`. Fixed so the throwaway run is itself
reproducible."""

DATA_DIR_ENV_VAR: Final[str] = "ALPHA__DATA_DIR"
DEFAULT_DATA_DIR: Final[str] = "data"

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

data_app = typer.Typer(name="data", help="Ingest and inspect point-in-time market data.")
app.add_typer(data_app)

config_app = typer.Typer(name="config", help="Resolve and inspect experiment configuration.")
app.add_typer(config_app)


def _data_dir(override: str | None) -> Path:
    """Resolve the lake root.

    Args:
        override: A `--data-dir` value, if given.

    Returns:
        The override, else `$ALPHA__DATA_DIR`, else `./data`.
    """
    return Path(override or os.environ.get(DATA_DIR_ENV_VAR) or DEFAULT_DATA_DIR)


def _load(config_path: Path, overrides: list[str] | None) -> AlphaConfig:
    """Load a config, turning library errors into clean CLI failures.

    A pydantic traceback is the wrong thing to show someone who mistyped a YAML
    key; the message from `ConfigError` already names the field.

    Args:
        config_path: Path to the experiment YAML.
        overrides: `section.key=value` strings.

    Returns:
        The resolved configuration.

    Raises:
        typer.Exit: With code 1 if the config is invalid.
    """
    from alphacore.config import load_config  # noqa: PLC0415 - keeps --help instant

    try:
        return load_config(config_path, overrides=overrides)
    except AlphaError as exc:
        console.print(f"[bold red]config error[/bold red]\n{exc}")
        raise typer.Exit(code=1) from exc


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


ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", exists=True, dir_okay=False, help="Experiment YAML."),
]
SetOption = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override a config value: section.key=value."),
]
DataDirOption = Annotated[
    str | None,
    typer.Option("--data-dir", help="Lake root. Defaults to $ALPHA__DATA_DIR, else ./data."),
]


@data_app.command("pull")
def data_pull(
    config: ConfigOption,
    overrides: SetOption = None,
    data_dir: DataDirOption = None,
    allow_dirty_data: Annotated[
        bool,
        typer.Option(
            "--allow-dirty-data",
            help="Proceed past a quality FAIL, marking every downstream artifact untrusted.",
        ),
    ] = False,
) -> None:
    """Fetch bars into the lake and record them in the catalog.

    Args:
        config: Experiment YAML.
        overrides: `section.key=value` overrides applied last.
        data_dir: Lake root.
        allow_dirty_data: Override a quality `FAIL` (§6.4).

    Raises:
        typer.Exit: With code 1 if ingestion fails.
    """
    from alphacore.pipelines import ingest_symbols  # noqa: PLC0415

    resolved = _load(config, overrides)
    set_global_seed(resolved.meta.seed)

    try:
        results = ingest_symbols(
            resolved, data_dir=_data_dir(data_dir), allow_dirty=allow_dirty_data
        )
    except AlphaError as exc:
        console.print(f"[bold red]ingest failed[/bold red]\n{exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"alpha data pull — {resolved.meta.name}")
    for column in ("dataset", "rows", "span", "quality", "content_hash", "trusted"):
        table.add_column(column)
    for result in results:
        quality = result.quality
        span = (
            f"{quality.first_bar:%Y-%m-%d} to {quality.last_bar:%Y-%m-%d}"
            if quality.first_bar and quality.last_bar
            else "-"
        )
        table.add_row(
            result.dataset_id,
            str(result.n_rows),
            span,
            quality.status.value,
            result.content_hash,
            "yes" if result.trusted else "NO",
        )
    console.print(table)

    for result in results:
        for reason in result.quality.reasons:
            console.print(f"  [yellow]{result.dataset_id}[/yellow]: {reason}")


@data_app.command("inspect")
def data_inspect(
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset id, e.g. yahoo:SPY:1d.")],
    data_dir: DataDirOption = None,
) -> None:
    """Print the catalog entry for one dataset.

    Args:
        dataset: Dataset identifier.
        data_dir: Lake root.

    Raises:
        typer.Exit: With code 1 if the dataset is not catalogued.
    """
    import json  # noqa: PLC0415

    from alphacore.config.schema import Interval  # noqa: PLC0415
    from alphacore.data import CATALOG_FILENAME, QUALITY_FILENAME, Catalog  # noqa: PLC0415
    from alphacore.data.lake import dataset_dir  # noqa: PLC0415

    root = _data_dir(data_dir)
    try:
        record = Catalog(root / CATALOG_FILENAME).require(dataset)
    except AlphaError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"dataset {record.dataset_id}", show_header=False)
    table.add_column("field")
    table.add_column("value")
    for field, value in record.model_dump(mode="json").items():
        table.add_row(field, "-" if value is None else str(value))
    console.print(table)

    quality_path = (
        dataset_dir(root, record.provider, record.symbol, Interval(record.interval))
        / QUALITY_FILENAME
    )
    if not quality_path.is_file():
        return

    report = json.loads(quality_path.read_text("utf-8"))
    quality = Table(title=f"quality report — {report['status']}")
    quality.add_column("measure")
    quality.add_column("value", justify="right")
    for key in (
        "n_rows",
        "n_expected_bars",
        "n_missing_bars",
        "missing_bar_pct",
        "longest_gap_bars",
        "n_unexpected_bars",
        "n_duplicate_timestamps",
        "n_zero_volume",
        "n_ohlc_violations",
        "n_extreme_returns",
    ):
        quality.add_row(key, str(report[key]))
    console.print(quality)
    for reason in report["reasons"]:
        console.print(f"  [yellow]{reason}[/yellow]")


@data_app.command("list")
def data_list(data_dir: DataDirOption = None) -> None:
    """List every dataset in the catalog.

    Args:
        data_dir: Lake root.
    """
    from alphacore.data import CATALOG_FILENAME, Catalog  # noqa: PLC0415

    records = Catalog(_data_dir(data_dir) / CATALOG_FILENAME).list_datasets()
    if not records:
        console.print(
            "No datasets ingested yet. Run:\n"
            "  [bold]alpha data pull -c configs/experiments/smoke.yaml[/bold]"
        )
        return

    table = Table(title="catalogued datasets")
    for column in ("dataset", "rows", "span", "quality", "content_hash", "ingested"):
        table.add_column(column)
    for record in records:
        table.add_row(
            record.dataset_id,
            str(record.n_rows),
            f"{record.start_ts:%Y-%m-%d} to {record.end_ts:%Y-%m-%d}",
            record.quality_status,
            record.content_hash,
            f"{record.ingested_at:%Y-%m-%d %H:%M}",
        )
    console.print(table)


@config_app.command("show")
def config_show(config: ConfigOption, overrides: SetOption = None) -> None:
    """Print a resolved config as YAML, with its hash.

    Args:
        config: Experiment YAML.
        overrides: `section.key=value` overrides applied last.
    """
    import yaml  # noqa: PLC0415

    from alphacore.config import config_hash, to_serialisable  # noqa: PLC0415

    resolved = _load(config, overrides)
    console.print(
        yaml.safe_dump(to_serialisable(resolved), sort_keys=True, default_flow_style=False)
    )
    console.print(f"[bold]config_hash[/bold] {config_hash(resolved)}")
    for warning in resolved.warnings:
        console.print(f"[yellow]warning[/yellow] {warning}")
