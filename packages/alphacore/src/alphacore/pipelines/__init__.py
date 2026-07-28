"""End-to-end pipelines. Each is a thin composition of library pieces."""

from __future__ import annotations

from alphacore.pipelines.backtest import (
    BacktestOutcome,
    backtest_predictions,
    deflate_outcomes,
    economic_columns,
    sharpe_variance,
)
from alphacore.pipelines.build_features import (
    FeatureSet,
    build_feature_set,
    load_bars,
    primary_symbol,
)
from alphacore.pipelines.full_run import RunSummary, run_experiment
from alphacore.pipelines.ingest import (
    IngestResult,
    ingest_dataset,
    ingest_symbols,
    resolve_calendar,
)
from alphacore.pipelines.train import TrainingResult, train_models

__all__ = [
    "BacktestOutcome",
    "FeatureSet",
    "IngestResult",
    "RunSummary",
    "TrainingResult",
    "backtest_predictions",
    "build_feature_set",
    "deflate_outcomes",
    "economic_columns",
    "ingest_dataset",
    "ingest_symbols",
    "load_bars",
    "primary_symbol",
    "resolve_calendar",
    "run_experiment",
    "sharpe_variance",
    "train_models",
]
