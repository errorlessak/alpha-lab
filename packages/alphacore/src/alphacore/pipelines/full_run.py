"""The golden path: config in, run directory out (§16, Phase 10).

`alpha run -c configs/experiments/smoke.yaml` is this function. It composes the
stages built in Phases 1 to 9 — ingest, features, labels, folds, tuning,
training, backtest, metrics, explanation, report — and writes the run directory
contract from §16.

Everything a reviewer needs to disbelieve the result is written alongside the
result: the resolved config, the data's quality report and content hash, the
fold definitions, the leaderboard including the shuffled-label baseline, the
cost-sensitivity curve, the probability of backtest overfitting, and the seed
and code revision that produced all of it. `tearsheet.json` is the canonical
artifact; everything else is either an input to it or a series it points at.

The directory is written once and never mutated (§4, I9). A crashed run is
finalised as `failed` with its traceback rather than left looking valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from alphacore.config.loader import to_serialisable
from alphacore.cv import FOLDS_FILENAME, splits_to_json
from alphacore.errors import (
    AlphaError,
    DataQualityError,
    ModelError,
    PipelineError,
    RunCancelledError,
)
from alphacore.eval import cscv_pbo
from alphacore.explain import shap_summary
from alphacore.models.leaderboard import WEAK_DSR
from alphacore.models.registry import build_model
from alphacore.pipelines.build_features import build_feature_set, load_bars, primary_symbol
from alphacore.pipelines.ingest import ingest_symbols, resolve_calendar
from alphacore.pipelines.train import train_models
from alphacore.report import (
    RESEARCH_FILENAME,
    TEARSHEET_FILENAME,
    WORSE_THAN_CHANCE_PBO,
    BacktestSection,
    CVSection,
    DatasetSection,
    ExplainSection,
    FeaturesSection,
    FoldSection,
    LabelsSection,
    ManifestSection,
    OverfittingSection,
    Tearsheet,
    export_frame,
    export_series,
    export_table,
    render_research_markdown,
)
from alphacore.runtime import logger, run_context, set_global_seed
from alphacore.runtime.artifacts import ArtifactStore
from alphacore.runtime.manifest import capture_environment

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Callable

    from alphacore.config.schema import AlphaConfig
    from alphacore.data.quality import DataQualityReport
    from alphacore.pipelines.backtest import BacktestOutcome
    from alphacore.pipelines.build_features import FeatureSet
    from alphacore.pipelines.train import TrainingResult
    from alphacore.runtime.artifacts import RunDirectory

__all__ = ["RunSummary", "run_experiment"]

_QUALITY_FILENAME = "quality.json"
_FEATURES_FILENAME = "features.json"
_CONFIG_FILENAME = "config.resolved.yaml"
_PBO_MIN_CONFIGS = 2
_PBO_MIN_OBSERVATIONS = 64


@dataclass(frozen=True)
class RunSummary:
    """What one completed run produced.

    Attributes:
        run_id: The run directory's name.
        path: Absolute path to the run directory.
        tearsheet: The canonical result artifact, as written.
        feature_set: The aligned features, labels and bars.
        training: Leaderboard, predictions, folds and backtests.
    """

    run_id: str
    path: Path
    tearsheet: Tearsheet
    feature_set: FeatureSet
    training: TrainingResult

    @property
    def headline(self) -> dict[str, Any]:
        """The figures a reviewer reads first, and the ones that qualify them."""
        return self.tearsheet.headline


def run_experiment(  # noqa: PLR0913 - each argument is a distinct run-level decision
    config: AlphaConfig,
    *,
    data_dir: Path | str = "data",
    store: ArtifactStore | None = None,
    run_id: str | None = None,
    allow_dirty_data: bool = False,
    skip_ingest: bool = False,
    explain: bool = True,
    now: dt.datetime | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> RunSummary:
    """Run the full golden path and write a complete run directory.

    Args:
        config: The resolved configuration.
        data_dir: Lake root.
        store: Artifact store. Defaults to one at `$ALPHA__RUNS_DIR`, else
            `./runs`.
        run_id: Override the generated run id. The determinism check in §12.4
            uses this so two runs of the same config differ in nothing but the
            directory they land in.
        allow_dirty_data: Proceed past a quality `FAIL`, marking every
            downstream artifact untrusted (§6.4).
        skip_ingest: Read bars already in the lake instead of fetching. The
            quality report is then read from the lake rather than recomputed.
        explain: Compute SHAP values for the selected model when it is a tree
            model. Disabled by tests that do not need them.
        now: Creation time, for a reproducible run id.
        should_cancel: Polled between phases. Returning True stops the run with
            a `RunCancelledError`, and the directory is finalised as
            `cancelled` rather than left half-written (§14.2). Checked between
            phases rather than inside them: a phase interrupted halfway would
            leave a partial artifact, which is the thing being avoided.

    Returns:
        The run summary.

    Raises:
        AlphaError: Any stage failure. The run directory is finalised as
            `failed` with the traceback before the error propagates.
        RunCancelledError: If `should_cancel` returns True at a phase boundary.
    """
    symbol = primary_symbol(config)
    set_global_seed(config.meta.seed)

    with run_context(
        name=config.meta.name,
        seed=config.meta.seed,
        config=to_serialisable(config),
        store=store or ArtifactStore(),
        run_id=run_id,
        now=now,
    ) as (run, timings):
        run.write_text(_CONFIG_FILENAME, _resolved_yaml(config))
        _check_cancelled(should_cancel, "before ingest")

        with timings.phase("ingest"):
            quality, data_hash, trusted = _prepare_data(
                config,
                run,
                data_dir=data_dir,
                symbol=symbol,
                allow_dirty=allow_dirty_data,
                skip_ingest=skip_ingest,
            )
            run.update_manifest(data_hash=data_hash)
        _check_cancelled(should_cancel, "after ingest")

        with timings.phase("features"):
            bars = load_bars(config, data_dir=data_dir, symbol=symbol)
            feature_set = build_feature_set(config, bars, symbol=symbol)
            run.write_json(_FEATURES_FILENAME, feature_set.manifest.model_dump(mode="json"))
        _check_cancelled(should_cancel, "after features")

        with timings.phase("train"):
            training = train_models(config, feature_set)
            run.write_text(FOLDS_FILENAME, splits_to_json(training.splits))
        _check_cancelled(should_cancel, "after training")

        selected = training.selected
        outcome = training.selected_outcome
        if selected is None or outcome is None:
            msg = (
                "The leaderboard produced no model with a costed backtest, so there is no "
                "result to report. Every candidate failed; see the run log."
            )
            raise PipelineError(msg)

        with timings.phase("report"):
            series_refs = _export_series(run, training, outcome)
            explanation = (
                _explain(run, config, feature_set, training, selected) if explain else None
            )
            overfitting = _overfitting(training)
            tearsheet = _build_tearsheet(
                config=config,
                run=run,
                feature_set=feature_set,
                training=training,
                outcome=outcome,
                quality=quality,
                data_hash=data_hash,
                trusted=trusted,
                series_refs=series_refs,
                explanation=explanation,
                overfitting=overfitting,
            )
            run.write_json(TEARSHEET_FILENAME, tearsheet.model_dump(mode="json"))
            run.write_text(RESEARCH_FILENAME, render_research_markdown(tearsheet))

    logger.bind(phase="run").success(
        "{} complete — selected {}, net Sharpe {:.3f}",
        run.run_id,
        selected,
        outcome.sharpe,
    )
    return RunSummary(
        run_id=run.run_id,
        path=run.path,
        tearsheet=tearsheet,
        feature_set=feature_set,
        training=training,
    )


def _check_cancelled(should_cancel: Callable[[], bool] | None, where: str) -> None:
    """Stop the run if something has asked it to.

    Args:
        should_cancel: The predicate, or None when nothing can cancel this run.
        where: The phase boundary being checked, named in the error so the run
            log says how far it got.

    Raises:
        RunCancelledError: If the predicate returns True.
    """
    if should_cancel is None or not should_cancel():
        return
    logger.bind(phase="run").warning("run cancelled {}", where)
    msg = f"The run was cancelled {where}."
    raise RunCancelledError(msg)


def _resolved_yaml(config: AlphaConfig) -> str:
    """Render the resolved config as sorted YAML.

    Args:
        config: The resolved configuration.

    Returns:
        YAML text. Sorted, so two runs of the same config produce identical
        bytes (§4, I8).
    """
    import yaml  # noqa: PLC0415 - only needed when a run is written

    return str(yaml.safe_dump(to_serialisable(config), sort_keys=True, default_flow_style=False))


def _prepare_data(
    config: AlphaConfig,
    run: RunDirectory,
    *,
    data_dir: Path | str,
    symbol: str,
    allow_dirty: bool,
    skip_ingest: bool,
) -> tuple[DataQualityReport | None, str | None, bool]:
    """Ingest, or read what is already in the lake, and record the quality report.

    Args:
        config: The resolved configuration.
        run: The writable run directory.
        data_dir: Lake root.
        symbol: The symbol being traded.
        allow_dirty: Proceed past a quality `FAIL`.
        skip_ingest: Skip fetching and use the catalogued dataset.

    Returns:
        The quality report, the dataset's content hash, and whether the data is
        trusted. A skipped ingest reads the report the lake already holds, so a
        run that reuses catalogued bars is no less accountable for their
        quality than one that fetched them.

    Raises:
        DataQualityError: If quality is `FAIL` and `allow_dirty` is False.
        DataError: If `skip_ingest` is set and the dataset is not catalogued.
    """
    if skip_ingest:
        from alphacore.data import CATALOG_FILENAME, QUALITY_FILENAME, Catalog  # noqa: PLC0415
        from alphacore.data.lake import dataset_dir, dataset_id  # noqa: PLC0415
        from alphacore.data.quality import DataQualityReport, QualityStatus  # noqa: PLC0415

        record = Catalog(Path(data_dir) / CATALOG_FILENAME).require(
            dataset_id(config.data.provider, symbol, config.data.interval)
        )
        report_path = (
            dataset_dir(data_dir, config.data.provider, symbol, config.data.interval)
            / QUALITY_FILENAME
        )
        stored = (
            DataQualityReport.model_validate_json(report_path.read_bytes())
            if report_path.is_file()
            else None
        )
        if stored is not None:
            run.write_json(_QUALITY_FILENAME, stored.model_dump(mode="json"))
            if stored.status is QualityStatus.FAIL and not allow_dirty:
                msg = (
                    f"The catalogued dataset for {symbol!r} carries a quality FAIL: "
                    f"{'; '.join(stored.reasons) or 'no reason recorded'}. Re-run with "
                    "--allow-dirty-data to proceed with every artifact marked untrusted."
                )
                raise DataQualityError(msg)
        return stored, record.content_hash, record.quality_status != QualityStatus.FAIL.value

    results = ingest_symbols(config, data_dir=data_dir, allow_dirty=allow_dirty)
    primary = next(
        (r for r in results if r.dataset_id.split(":")[1] == symbol),
        results[0] if results else None,
    )
    if primary is None:  # pragma: no cover - ingest_symbols always returns one
        msg = f"Ingestion returned no dataset for {symbol!r}."
        raise PipelineError(msg)
    run.write_json(_QUALITY_FILENAME, primary.quality.model_dump(mode="json"))
    return primary.quality, primary.content_hash, primary.trusted


def _export_series(
    run: RunDirectory, training: TrainingResult, outcome: BacktestOutcome
) -> dict[str, str]:
    """Write every large series the tearsheet points at (§16).

    Args:
        run: The writable run directory.
        training: The training result, for the leaderboard and predictions.
        outcome: The selected model's backtest.

    Returns:
        A mapping of logical name to artifact name, for `series_refs`.

    Raises:
        ArtifactError: If an artifact already exists.
    """
    refs = {
        "equity": export_frame(run, "equity.parquet", outcome.result.equity),
        "drawdown": export_frame(run, "drawdown.parquet", outcome.result.drawdown),
        "trades": export_frame(run, "trades.parquet", outcome.result.trades),
        "ledger": export_frame(run, "ledger.parquet", outcome.result.fills),
    }
    for window, series in sorted(outcome.rolling.items()):
        refs[f"rolling_sharpe_{window}"] = export_series(
            run, f"rolling_sharpe_{window}.parquet", series, column="sharpe"
        )
    if outcome.rolling:
        # §16 names a single `rolling_sharpe.parquet`. The shortest window is
        # the one the UI plots by default; the others sit beside it.
        shortest = min(outcome.rolling)
        refs["rolling_sharpe"] = export_series(
            run, "rolling_sharpe.parquet", outcome.rolling[shortest], column="sharpe"
        )

    predictions = training.predictions.get(outcome.model)
    if predictions is not None:
        refs["predictions"] = export_frame(run, "predictions.parquet", predictions.frame)

    refs["leaderboard_csv"] = export_table(
        run, "leaderboard.csv", pd.DataFrame(training.rows).set_index("rank")
    )
    refs["metrics_csv"] = export_table(run, "metrics.csv", _metrics_frame(outcome))
    refs["trades_csv"] = export_table(run, "trades.csv", outcome.result.trades)

    importance = predictions.mean_importance if predictions is not None else None
    if importance is not None and not importance.empty:
        refs["feature_importance_csv"] = export_table(
            run, "feature_importance.csv", importance.to_frame("importance")
        )
    return refs


def _metrics_frame(outcome: BacktestOutcome) -> pd.DataFrame:
    """Flatten every metric into the two-column table `metrics.csv` holds.

    Args:
        outcome: The selected model's backtest.

    Returns:
        A frame indexed by metric name with a `value` column, sorted by name so
        the file is stable across runs (§4, I8).
    """
    values: dict[str, float] = {**outcome.metrics}
    values.update({f"trade_{k}": v for k, v in outcome.trades.items()})
    values["break_even_bps"] = outcome.break_even_bps
    values["assumed_round_trip_bps"] = outcome.round_trip_bps
    values["periods_per_year"] = outcome.periods_per_year
    if outcome.psr is not None:
        values["psr"] = outcome.psr
    if outcome.dsr is not None:
        values["dsr"] = outcome.dsr
    if outcome.min_track_record_length is not None:
        values["min_track_record_length"] = outcome.min_track_record_length
    if outcome.sharpe_ci is not None:
        values["sharpe_ci_lower"], values["sharpe_ci_upper"] = outcome.sharpe_ci
    frame = pd.DataFrame({"value": pd.Series(values)}).sort_index()
    frame.index.name = "metric"
    return frame


def _explain(
    run: RunDirectory,
    config: AlphaConfig,
    feature_set: FeatureSet,
    training: TrainingResult,
    selected: str,
) -> ExplainSection | None:
    """Refit the selected model, persist it, and explain it if it is a tree.

    The model is refit on the **last** fold's training rows: that is the most
    recent window the walk-forward trained on, and therefore the fit a reader
    would carry forward. Explaining an average of per-fold models would explain
    something that never made a prediction.

    Args:
        run: The writable run directory.
        config: The resolved configuration.
        feature_set: Aligned features, labels and bars.
        training: The training result, for the folds and predictions.
        selected: The top-ranked model's name.

    Returns:
        The explain section, or None when the selected model is not explainable
        — a linear model's coefficients already are its explanation (§9).
    """
    split = training.splits[-1]
    try:
        model = build_model(selected, seed=config.meta.seed, calibrate=config.models.calibrate)
        # A sequence model windows on grid positions, so the refit needs the
        # same full index the folds were cut from (§10.3). Models that do not
        # window ignore this.
        set_bar_grid = getattr(model, "set_bar_grid", None)
        if set_bar_grid is not None:
            set_bar_grid(feature_set.features.index)
        fitted = model.fit(
            feature_set.features.iloc[split.train_idx],
            feature_set.labels["label"].iloc[split.train_idx],
            sample_weight=feature_set.labels["weight"].iloc[split.train_idx].to_numpy(dtype=float),
        )
    except AlphaError as exc:
        logger.bind(phase="explain").warning("{} could not be refit for export: {}", selected, exc)
        return None

    model_path = run.artifact_path(f"models/{selected}/model.pkl")
    if not model_path.exists():
        fitted.save(model_path)

    try:
        result = shap_summary(
            fitted, feature_set.features.iloc[split.test_idx], seed=config.meta.seed
        )
    except ModelError as exc:
        logger.bind(phase="explain").info("{} is not explainable by TreeSHAP: {}", selected, exc)
        return None

    export_frame(run, "shap_summary.parquet", result.summary_frame())
    return ExplainSection(
        shap_summary_ref="shap_summary.parquet",
        top_features=result.top_features(),
        n_explained=len(result.values),
    )


def _overfitting(training: TrainingResult) -> OverfittingSection:
    """Compute the probability of backtest overfitting across models (§11.3).

    The configurations CSCV compares are the leaderboard's models: each one is
    a different way of turning the same features into the same signal, which is
    exactly the selection process PBO asks about.

    Args:
        training: The training result.

    Returns:
        The overfitting section. `pbo` is None when there are too few
        configurations or too short a sample for the statistic to mean
        anything — an honest gap rather than a fabricated number.
    """
    shuffled = training.outcomes.get("baseline_shuffled")
    shuffled_row = next((r for r in training.rows if r["model"] == "baseline_shuffled"), None)
    section_kwargs: dict[str, Any] = {
        "n_configs_evaluated": training.n_trials_total,
        "shuffled_label_sharpe": shuffled.sharpe if shuffled else None,
        "shuffled_label_ic": shuffled_row.get("ic") if shuffled_row else None,
    }

    frame = pd.DataFrame(
        {name: outcome.returns for name, outcome in sorted(training.outcomes.items())}
    ).dropna()
    if frame.shape[1] < _PBO_MIN_CONFIGS or len(frame) < _PBO_MIN_OBSERVATIONS:
        logger.bind(phase="metrics").info(
            "PBO not computed: {} configurations over {} observations",
            frame.shape[1],
            len(frame),
        )
        return OverfittingSection(**section_kwargs)

    try:
        pbo = cscv_pbo(frame.to_numpy(dtype=float))
    except AlphaError as exc:
        logger.bind(phase="metrics").warning("PBO not computed: {}", exc)
        return OverfittingSection(**section_kwargs)

    return OverfittingSection(
        pbo=float(pbo["pbo"]),
        median_logit=float(pbo["median_logit"]),
        n_splits=int(pbo["n_splits"]),
        **section_kwargs,
    )


def _build_tearsheet(  # noqa: PLR0913 - the tearsheet is assembled from every stage
    *,
    config: AlphaConfig,
    run: RunDirectory,
    feature_set: FeatureSet,
    training: TrainingResult,
    outcome: BacktestOutcome,
    quality: DataQualityReport | None,
    data_hash: str | None,
    trusted: bool,
    series_refs: dict[str, str],
    explanation: ExplainSection | None,
    overfitting: OverfittingSection,
) -> Tearsheet:
    """Assemble the canonical result artifact (§11.4).

    Args:
        config: The resolved configuration.
        run: The run directory, for the manifest already written.
        feature_set: Aligned features, labels and bars.
        training: Leaderboard, folds and predictions.
        outcome: The selected model's backtest.
        quality: The data quality report, if ingestion ran.
        data_hash: Content hash of the input dataset.
        trusted: False when a quality `FAIL` was overridden.
        series_refs: Artifact names for the large series.
        explanation: The explain section, if SHAP ran.
        overfitting: The honesty section.

    Returns:
        The validated tearsheet.
    """
    manifest = run.read_manifest()
    index = pd.DatetimeIndex(feature_set.bars.index)
    top = training.rows[0]

    warnings = set(config.warnings) | set(outcome.warnings)
    if not trusted:
        warnings.add("UNTRUSTED_DATA")
    # The same threshold the leaderboard flags rows on, imported rather than
    # repeated: two copies of it would drift, and the run-level warning would
    # then disagree with the row it came from.
    if top.get("dsr") is not None and top["dsr"] < WEAK_DSR:
        warnings.add("LOW_DSR")
    if overfitting.pbo is not None and overfitting.pbo > WORSE_THAN_CHANCE_PBO:
        warnings.add("HIGH_PBO")

    return Tearsheet(
        run_id=run.run_id,
        created_at=manifest.created_at,
        manifest=ManifestSection(
            config_hash=manifest.config_hash,
            data_hash=data_hash,
            code_version=manifest.code_version,
            seed=config.meta.seed,
            python=capture_environment().python,
            packages=dict(manifest.environment.packages),
            duration_s=manifest.duration_s,
            host=manifest.environment.host,
        ),
        dataset=DatasetSection(
            symbols=(feature_set.symbol,),
            interval=config.data.interval.value,
            calendar=resolve_calendar(config.data),
            span=(index[0].date(), index[-1].date()),
            n_bars=len(feature_set.bars),
            quality=quality.status.value if quality else "UNKNOWN",
            quality_report_ref=_QUALITY_FILENAME if quality else None,
            periods_per_year=feature_set.periods_per_year,
        ),
        features=FeaturesSection(
            n_built=feature_set.manifest.n_columns,
            n_after_prune=_n_after_prune(training),
            manifest_ref=_FEATURES_FILENAME,
            max_lookback=feature_set.manifest.max_lookback,
            families=feature_set.manifest.families,
        ),
        labels=LabelsSection(
            method=config.labels.method,
            horizon_bars=config.labels.horizon_bars,
            class_balance=feature_set.class_balance,
            mean_uniqueness=feature_set.mean_uniqueness,
            binarised=feature_set.binarised,
        ),
        cv=CVSection(
            scheme=config.cv.scheme,
            embargo_pct=config.cv.embargo_pct,
            folds=tuple(
                FoldSection(
                    fold=split.fold,
                    train_span=split.train_span,
                    test_span=split.test_span,
                    n_train=int(split.train_idx.size),
                    n_test=int(split.test_idx.size),
                    purged=split.purged,
                    embargoed=split.embargoed,
                )
                for split in training.splits
            ),
        ),
        leaderboard=tuple(training.rows),  # type: ignore[arg-type]  # validated by pydantic
        selected_model=outcome.model,
        backtest=BacktestSection(
            metrics=outcome.metrics,
            trade_metrics=outcome.trades,
            cost_sensitivity=tuple(outcome.cost_sensitivity),
            break_even_bps=outcome.break_even_bps,
            round_trip_bps=outcome.round_trip_bps,
            kill_switch=(
                outcome.result.kill_switch.to_dict() if outcome.result.kill_switch else None
            ),
            series_refs=series_refs,
        ),
        overfitting=overfitting,
        explain=explanation,
        warnings=tuple(sorted(warnings)),
    )


def _n_after_prune(training: TrainingResult) -> int:
    """How many features survived pruning in the fold that trained last.

    Pruning is fitted per fold (§4, I3), so there is no single global count.
    The last fold's is the one that matters: it is the selection the model a
    reader would carry forward was trained under.

    Args:
        training: The training result.

    Returns:
        The surviving column count, or 0 when nothing recorded a selection.
    """
    for predictions in training.predictions.values():
        if predictions.selected_features:
            return len(predictions.selected_features[-1])
    return 0
