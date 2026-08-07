"""The `alpha` command-line entry point.

Every verb §15 asks for is here: `version`, `doctor`, `dev scratch-run`,
`data pull|inspect|list`, `config show`, `features build`, `train`, `backtest`,
`run`, `report`, and `runs list|show|compare`.

**`alpha run` is the only verb that writes a run directory.** The stage verbs —
`features build`, `train`, `backtest` — run the pipeline up to that point and
print what they found. That is deliberate: a half-finished run directory would
satisfy the §16 contract's filenames without satisfying its meaning, and the
API and UI read those directories as though they describe a complete result.

Heavy imports (`alphacore.data`, `alphacore.config`) happen inside the commands
that need them, so `alpha --help` and `alpha version` stay instant.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final

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
    from alphacore.report import Tearsheet

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

features_app = typer.Typer(name="features", help="Build and inspect the causal feature frame.")
app.add_typer(features_app)

runs_app = typer.Typer(name="runs", help="List, inspect and compare completed runs.")
app.add_typer(runs_app)


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


RunsDirOption = Annotated[
    str | None,
    typer.Option(
        "--runs-dir", help="Artifact store root. Defaults to $ALPHA__RUNS_DIR, else ./runs."
    ),
]


def _feature_set(config: AlphaConfig, data_dir: str | None) -> Any:  # noqa: ANN401
    """Load bars and build the aligned feature set, or exit cleanly.

    Args:
        config: The resolved configuration.
        data_dir: Lake root override.

    Returns:
        The `FeatureSet`.

    Raises:
        typer.Exit: With code 1 if the data is missing or the stage fails.
    """
    from alphacore.pipelines import build_feature_set, load_bars  # noqa: PLC0415

    try:
        bars = load_bars(config, data_dir=_data_dir(data_dir))
        return build_feature_set(config, bars)
    except AlphaError as exc:
        console.print(f"[bold red]features failed[/bold red]\n{exc}")
        console.print(
            "If the dataset is not in the lake yet, run "
            "[bold]alpha data pull -c <config>[/bold] first."
        )
        raise typer.Exit(code=1) from exc


def _leaderboard_table(rows: list[dict[str, Any]], *, title: str) -> Table:
    """Render leaderboard rows as a console table.

    Ranked by net Sharpe once a backtest has run, and the shuffled-label
    baseline is always present — it is the only row that says what this
    pipeline scores on pure noise.

    Args:
        rows: Leaderboard rows.
        title: Table title.

    Returns:
        The table.
    """
    table = Table(title=title)
    for column in ("rank", "model", "net Sharpe", "DSR", "PSR", "IC t", "AUC", "flags"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            str(row["rank"]),
            str(row["model"]),
            _fmt(row.get("oos_net_sharpe"), ".3f"),
            _fmt(row.get("dsr"), ".2f"),
            _fmt(row.get("psr"), ".2f"),
            _fmt(row.get("ic_tstat"), ".2f"),
            _fmt(row.get("auc"), ".3f"),
            ", ".join(row.get("flags", [])) or "-",
        )
    return table


def _fmt(value: Any, spec: str = ".3f") -> str:  # noqa: ANN401
    """Format a possibly-absent number for the console.

    Args:
        value: The value.
        spec: Format spec.

    Returns:
        The formatted string, or `-` when the value is absent.
    """
    if value is None:
        return "-"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _open_store(runs_dir: str | None) -> ArtifactStore:
    """Build an artifact store from an optional override.

    Args:
        runs_dir: Store root override.

    Returns:
        The store.
    """
    return ArtifactStore(runs_dir) if runs_dir else ArtifactStore()


def _read_tearsheet(store: ArtifactStore, run_id: str) -> Any:  # noqa: ANN401
    """Read and validate one run's tearsheet, or exit cleanly.

    Args:
        store: The artifact store.
        run_id: The run id.

    Returns:
        The validated `Tearsheet`.

    Raises:
        typer.Exit: With code 1 if the run or its tearsheet is missing.
    """
    from alphacore.report import TEARSHEET_FILENAME, validate_tearsheet  # noqa: PLC0415

    try:
        return validate_tearsheet(store.open_run(run_id).read_json(TEARSHEET_FILENAME))
    except AlphaError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1) from exc


@features_app.command("build")
def features_build(
    config: ConfigOption, overrides: SetOption = None, data_dir: DataDirOption = None
) -> None:
    """Build the feature frame and label set, and report what they contain.

    Nothing is written: features are cheap to rebuild and expensive to
    misattribute, so they live in a run directory only as part of a full run.

    Args:
        config: Experiment YAML.
        overrides: `section.key=value` overrides applied last.
        data_dir: Lake root.
    """
    resolved = _load(config, overrides)
    set_global_seed(resolved.meta.seed)
    feature_set = _feature_set(resolved, data_dir)
    manifest = feature_set.manifest

    table = Table(title=f"alpha features build — {resolved.meta.name}", show_header=False)
    table.add_column("field")
    table.add_column("value")
    table.add_row("symbol", feature_set.symbol)
    table.add_row("rows", f"{len(feature_set.features):,}")
    table.add_row("columns", str(manifest.n_columns))
    table.add_row("warm-up dropped", f"{manifest.max_lookback} bars")
    table.add_row("annualisation", f"{manifest.periods_per_year:.2f} bars/year")
    table.add_row("calendar", manifest.calendar)
    table.add_row("labels", resolved.labels.method)
    table.add_row("binarised", feature_set.binarised or "not required")
    table.add_row(
        "class balance",
        ", ".join(f"{k}: {v:.1%}" for k, v in sorted(feature_set.class_balance.items())),
    )
    console.print(table)

    families = Table(title="feature families")
    families.add_column("family")
    families.add_column("columns", justify="right")
    for family, count in manifest.families.items():
        families.add_row(family, str(count))
    console.print(families)


@app.command()
def train(
    config: ConfigOption, overrides: SetOption = None, data_dir: DataDirOption = None
) -> None:
    """Cut purged folds, tune, and score every model out of fold.

    The leaderboard this prints is economic: each row's Sharpe comes from a
    costed backtest of that model's out-of-fold probabilities (ADR 0007). No
    run directory is written — use `alpha run` for that.

    Args:
        config: Experiment YAML.
        overrides: `section.key=value` overrides applied last.
        data_dir: Lake root.

    Raises:
        typer.Exit: With code 1 if no model could be scored.
    """
    from alphacore.pipelines import train_models  # noqa: PLC0415

    resolved = _load(config, overrides)
    set_global_seed(resolved.meta.seed)
    feature_set = _feature_set(resolved, data_dir)

    try:
        result = train_models(resolved, feature_set)
    except AlphaError as exc:
        console.print(f"[bold red]training failed[/bold red]\n{exc}")
        raise typer.Exit(code=1) from exc

    console.print(_leaderboard_table(result.rows, title=f"alpha train — {resolved.meta.name}"))
    console.print(
        f"{len(result.splits)} folds, "
        f"{sum(s.purged for s in result.splits):,} samples purged, "
        f"{sum(s.embargoed for s in result.splits):,} embargoed"
    )
    console.print(f"configurations evaluated: {result.n_trials_total}")


@app.command()
def backtest(
    config: ConfigOption, overrides: SetOption = None, data_dir: DataDirOption = None
) -> None:
    """Backtest the selected model and print its costs and sensitivity curve.

    Args:
        config: Experiment YAML.
        overrides: `section.key=value` overrides applied last.
        data_dir: Lake root.

    Raises:
        typer.Exit: With code 1 if nothing could be backtested.
    """
    from alphacore.pipelines import train_models  # noqa: PLC0415

    resolved = _load(config, overrides)
    set_global_seed(resolved.meta.seed)
    feature_set = _feature_set(resolved, data_dir)

    try:
        result = train_models(resolved, feature_set)
    except AlphaError as exc:
        console.print(f"[bold red]backtest failed[/bold red]\n{exc}")
        raise typer.Exit(code=1) from exc

    outcome = result.selected_outcome
    if outcome is None:
        console.print("[bold red]No model produced a costed backtest.[/bold red]")
        raise typer.Exit(code=1)

    table = Table(title=f"alpha backtest — {outcome.model}", show_header=False)
    table.add_column("measure")
    table.add_column("value", justify="right")
    table.add_row("net Sharpe", _fmt(outcome.sharpe))
    table.add_row(
        "Sharpe 95% interval",
        "-"
        if outcome.sharpe_ci is None
        else f"[{outcome.sharpe_ci[0]:.3f}, {outcome.sharpe_ci[1]:.3f}]",
    )
    table.add_row("deflated Sharpe", _fmt(outcome.dsr, ".2f"))
    table.add_row("probabilistic Sharpe", _fmt(outcome.psr, ".2f"))
    table.add_row("max drawdown", _fmt(outcome.metrics.get("max_drawdown"), ".2%"))
    table.add_row("break-even round trip", f"{outcome.break_even_bps:.1f} bps")
    table.add_row("assumed round trip", f"{outcome.round_trip_bps:.1f} bps")
    table.add_row("trades", str(int(outcome.trades.get("n_trades", 0))))
    table.add_row("total costs", _fmt(outcome.result.total_costs, ",.2f"))
    table.add_row("flags", ", ".join(outcome.warnings) or "-")
    console.print(table)

    curve = Table(title="cost sensitivity (§12.2)")
    curve.add_column("round trip bps", justify="right")
    curve.add_column("Sharpe", justify="right")
    curve.add_column("total return", justify="right")
    curve.add_column("trades", justify="right")
    for point in outcome.cost_sensitivity:
        curve.add_row(
            f"{point['bps']:.1f}",
            f"{point['sharpe']:.3f}",
            f"{point['total_return']:.2%}",
            str(int(point["n_trades"])),
        )
    console.print(curve)


@app.command()
def run(
    config: ConfigOption,
    overrides: SetOption = None,
    data_dir: DataDirOption = None,
    runs_dir: RunsDirOption = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Override the generated run id. Used by the §12.4 check."),
    ] = None,
    allow_dirty_data: Annotated[
        bool,
        typer.Option("--allow-dirty-data", help="Proceed past a quality FAIL (§6.4)."),
    ] = False,
    skip_ingest: Annotated[
        bool,
        typer.Option("--skip-ingest", help="Use the dataset already in the lake."),
    ] = False,
) -> None:
    """Run the full golden path and write a complete run directory (§16).

    Args:
        config: Experiment YAML.
        overrides: `section.key=value` overrides applied last.
        data_dir: Lake root.
        runs_dir: Artifact store root.
        run_id: Override the generated run id.
        allow_dirty_data: Proceed past a quality `FAIL`.
        skip_ingest: Skip fetching and read what the lake already holds.

    Raises:
        typer.Exit: With code 1 if any stage fails. The run directory is
            finalised as `failed` with its traceback before this returns.
    """
    from alphacore.pipelines import run_experiment  # noqa: PLC0415

    resolved = _load(config, overrides)
    try:
        summary = run_experiment(
            resolved,
            data_dir=_data_dir(data_dir),
            store=_open_store(runs_dir),
            run_id=run_id,
            allow_dirty_data=allow_dirty_data,
            skip_ingest=skip_ingest,
        )
    except AlphaError as exc:
        console.print(f"[bold red]run failed[/bold red]\n{exc}")
        raise typer.Exit(code=1) from exc

    console.print(_leaderboard_table(summary.training.rows, title="leaderboard"))
    headline = summary.headline

    table = Table(title=f"alpha run — {summary.run_id}", show_header=False)
    table.add_column("measure")
    table.add_column("value", justify="right")
    table.add_row("selected model", str(headline["model"]))
    table.add_row("out-of-sample net Sharpe", _fmt(headline["oos_net_sharpe"]))
    table.add_row(
        "Sharpe 95% interval",
        "-"
        if headline["sharpe_ci"] is None
        else f"[{headline['sharpe_ci'][0]:.3f}, {headline['sharpe_ci'][1]:.3f}]",
    )
    table.add_row("deflated Sharpe", _fmt(headline["dsr"], ".2f"))
    table.add_row("break-even round trip", _fmt(headline["break_even_bps"], ".1f") + " bps")
    table.add_row("assumed round trip", _fmt(headline["assumed_round_trip_bps"], ".1f") + " bps")
    table.add_row("probability of overfitting", _fmt(headline["pbo"], ".2f"))
    table.add_row("warnings", ", ".join(headline["warnings"]) or "none")
    console.print(table)

    console.print(f"[bold]run directory[/bold] {summary.path.as_posix()}")
    console.print(summary.tearsheet.overfitting.verdict)


@app.command()
def report(
    run_id: Annotated[str, typer.Argument(help="Run id, or its short suffix.")],
    runs_dir: RunsDirOption = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the note here instead of printing it."),
    ] = None,
) -> None:
    """Render a completed run's research note from its tearsheet.

    The note is regenerated rather than read back: run directories are
    immutable (§4, I9), so `research.md` inside one is never rewritten, and a
    note rendered here always matches the tearsheet it came from.

    Args:
        run_id: The run id.
        runs_dir: Artifact store root.
        output: Destination file. Printed to the console when absent.

    Raises:
        typer.Exit: With code 1 if the run is not found.
    """
    from alphacore.report import render_research_markdown  # noqa: PLC0415

    store = _open_store(runs_dir)
    tearsheet = _read_tearsheet(store, _resolve_run_id(store, run_id))
    markdown = render_research_markdown(tearsheet)

    if output is None:
        console.print(markdown)
        return
    output.write_text(markdown, encoding="utf-8", newline="\n")
    console.print(f"[bold]wrote[/bold] {output.as_posix()}")


def _resolve_run_id(store: ArtifactStore, candidate: str) -> str:
    """Resolve a run id, accepting the short hash suffix §16 prints.

    Args:
        store: The artifact store.
        candidate: A full run id or a unique suffix of one.

    Returns:
        The full run id.

    Raises:
        typer.Exit: With code 1 if nothing matches, or several do.
    """
    known = store.list_runs()
    if candidate in known:
        return candidate
    matches = [name for name in known if name.endswith(candidate) or candidate in name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        console.print(f"[bold red]No run matching {candidate!r} in {store.root.as_posix()}.")
        raise typer.Exit(code=1)
    console.print(f"[bold red]{candidate!r} matches {len(matches)} runs:[/bold red]")
    for name in matches:
        console.print(f"  {name}")
    raise typer.Exit(code=1)


@runs_app.command("list")
def runs_list(runs_dir: RunsDirOption = None) -> None:
    """List every run in the store, newest first.

    Args:
        runs_dir: Artifact store root.
    """
    store = _open_store(runs_dir)
    ids = store.list_runs()
    if not ids:
        console.print(
            f"No runs in {store.root.as_posix()}. Produce one with:\n"
            "  [bold]alpha run -c configs/experiments/smoke.yaml[/bold]"
        )
        return

    table = Table(title=f"runs in {store.root.as_posix()}")
    for column in ("run id", "status", "created", "duration", "seed", "config hash"):
        table.add_column(column)
    for run_id in ids:
        try:
            manifest = store.open_run(run_id).read_manifest()
        except AlphaError:
            table.add_row(run_id, "unreadable", "-", "-", "-", "-")
            continue
        table.add_row(
            manifest.run_id,
            manifest.status.value,
            f"{manifest.created_at:%Y-%m-%d %H:%M}",
            "-" if manifest.duration_s is None else f"{manifest.duration_s:.1f}s",
            str(manifest.seed),
            manifest.config_hash,
        )
    console.print(table)


@runs_app.command("show")
def runs_show(
    run_id: Annotated[str, typer.Argument(help="Run id, or its short suffix.")],
    runs_dir: RunsDirOption = None,
) -> None:
    """Print one run's headline, leaderboard and caveats.

    Args:
        run_id: The run id.
        runs_dir: Artifact store root.
    """
    store = _open_store(runs_dir)
    resolved_id = _resolve_run_id(store, run_id)
    tearsheet = _read_tearsheet(store, resolved_id)
    headline = tearsheet.headline

    table = Table(title=f"run {resolved_id}", show_header=False)
    table.add_column("measure")
    table.add_column("value", justify="right")
    table.add_row("selected model", str(headline["model"]))
    table.add_row("net Sharpe", _fmt(headline["oos_net_sharpe"]))
    table.add_row("deflated Sharpe", _fmt(headline["dsr"], ".2f"))
    table.add_row("break-even round trip", _fmt(headline["break_even_bps"], ".1f") + " bps")
    table.add_row("assumed round trip", _fmt(headline["assumed_round_trip_bps"], ".1f") + " bps")
    table.add_row("probability of overfitting", _fmt(headline["pbo"], ".2f"))
    table.add_row("data", f"{tearsheet.dataset.symbols[0]} {tearsheet.dataset.interval}")
    table.add_row("span", f"{tearsheet.dataset.span[0]} to {tearsheet.dataset.span[1]}")
    table.add_row("annualisation", f"{tearsheet.dataset.periods_per_year:.2f} bars/year")
    table.add_row("config hash", tearsheet.manifest.config_hash)
    table.add_row("data hash", tearsheet.manifest.data_hash or "unrecorded")
    table.add_row("code version", tearsheet.manifest.code_version or "unrecorded")
    table.add_row("warnings", ", ".join(tearsheet.warnings) or "none")
    console.print(table)

    console.print(
        _leaderboard_table(
            [row.model_dump(mode="json") for row in tearsheet.leaderboard], title="leaderboard"
        )
    )
    console.print(tearsheet.overfitting.verdict)


@runs_app.command("compare")
def runs_compare(
    run_ids: Annotated[list[str], typer.Argument(help="Two or more run ids, or their suffixes.")],
    runs_dir: RunsDirOption = None,
) -> None:
    """Compare the headline figures of two or more runs side by side.

    Args:
        run_ids: The runs to compare.
        runs_dir: Artifact store root.

    Raises:
        typer.Exit: With code 1 if fewer than two runs were named.
    """
    minimum = 2
    if len(run_ids) < minimum:
        console.print("[bold red]Comparing needs at least two runs.[/bold red]")
        raise typer.Exit(code=1)

    store = _open_store(runs_dir)
    resolved = [_resolve_run_id(store, candidate) for candidate in run_ids]
    sheets = [_read_tearsheet(store, run_id) for run_id in resolved]

    table = Table(title="alpha runs compare")
    table.add_column("measure")
    for run_id in resolved:
        table.add_column(run_id[-6:], justify="right")

    def row(label: str, values: list[str]) -> None:
        table.add_row(label, *values)

    row("selected model", [str(s.headline["model"]) for s in sheets])
    row("net Sharpe", [_fmt(s.headline["oos_net_sharpe"]) for s in sheets])
    row("deflated Sharpe", [_fmt(s.headline["dsr"], ".2f") for s in sheets])
    row("break-even bps", [_fmt(s.headline["break_even_bps"], ".1f") for s in sheets])
    row("assumed bps", [_fmt(s.headline["assumed_round_trip_bps"], ".1f") for s in sheets])
    row("PBO", [_fmt(s.overfitting.pbo, ".2f") for s in sheets])
    row("max drawdown", [_fmt(_max_drawdown_of(s), ".2%") for s in sheets])
    row("bars", [f"{s.dataset.n_bars:,}" for s in sheets])
    row("seed", [str(s.manifest.seed) for s in sheets])
    row("config hash", [s.manifest.config_hash for s in sheets])
    row("code version", [s.manifest.code_version or "-" for s in sheets])
    row("warnings", [", ".join(s.warnings) or "none" for s in sheets])
    console.print(table)

    hashes = {s.manifest.config_hash for s in sheets}
    if len(hashes) == 1:
        console.print(
            "[yellow]These runs share a config hash.[/yellow] Any difference between them "
            "comes from the data or the code version, not the configuration."
        )


def _max_drawdown_of(tearsheet: Tearsheet) -> float | None:
    """Pull the max drawdown out of a tearsheet's backtest section.

    Args:
        tearsheet: A validated tearsheet.

    Returns:
        The drawdown, or None if the run has no backtest.
    """
    if tearsheet.backtest is None:
        return None
    value = tearsheet.backtest.metrics.get("max_drawdown")
    return None if value is None else float(value)


@app.command()
def openapi(
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the document here instead of printing it."),
    ] = None,
) -> None:
    """Emit the OpenAPI 3.1 document the TypeScript client is generated from.

    Written from the application object rather than fetched from a running
    server, so `just client` needs no port and no process (§14.1).

    Args:
        output: Destination file. Printed to stdout when absent.
    """
    import json  # noqa: PLC0415

    from alphaservice.api.app import create_app  # noqa: PLC0415

    document = json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"
    if output is None:
        console.print_json(document)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8", newline="\n")
    console.print(f"[bold]wrote[/bold] {output.as_posix()}")


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to bind.")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Restart on source changes.")] = False,
    web_dist: Annotated[
        Path | None,
        typer.Option(
            "--web-dist",
            help="Serve a built web bundle from this origin, e.g. `web/dist`.",
            exists=True,
            file_okay=False,
        ),
    ] = None,
) -> None:
    """Run the API service.

    Binds loopback by default. This service reads a research artifact store and
    runs pipelines on request; it has no authentication, and exposing it on a
    routable interface is a decision the operator has to make deliberately.

    Args:
        host: Interface to bind.
        port: Port to bind.
        reload: Restart on source changes.
        web_dist: A built web bundle to serve alongside the API, so both answer
            on one origin. Development normally leaves this unset and lets Vite
            proxy `/api` instead.
    """
    import uvicorn  # noqa: PLC0415

    from alphaservice.api.settings import WEB_DIST_ENV_VAR  # noqa: PLC0415

    if web_dist is not None:
        # Through the environment rather than through `Settings`, because
        # `--reload` re-imports the application in a process this one cannot
        # otherwise reach.
        os.environ[WEB_DIST_ENV_VAR] = str(web_dist)

    if host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            f"[yellow]warning[/yellow] binding {host}, which is not loopback. This API has no "
            "authentication and can start jobs."
        )
    uvicorn.run(
        "alphaservice.api.app:app" if reload else create_api(),
        host=host,
        port=port,
        reload=reload,
        log_config=None,
    )


def create_api() -> Any:  # noqa: ANN401 - the FastAPI app, imported lazily
    """Build the API application.

    Returns:
        The configured FastAPI application, reading its settings from the same
        environment variables the CLI honours.
    """
    from alphaservice.api.app import create_app  # noqa: PLC0415

    return create_app()


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
