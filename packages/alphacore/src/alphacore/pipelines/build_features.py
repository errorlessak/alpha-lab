"""The feature and label stage (§7, §8, Phase 10).

Reads bars out of the lake, builds the configured feature frame, labels them,
reduces three-class labels to binary, and aligns the three so every row has a
feature vector, a label, and a bar to trade on.

The alignment is the point. Features lose their warm-up from the head; labels
lose their horizon from the tail. Everything downstream — folds, models,
signals, the backtest — indexes off the intersection, so the leaderboard and the
equity curve are describing the same rows rather than two overlapping subsets
that happen to be similar.

Pruning is deliberately not applied here. It is a fitted decision, so it belongs
inside the fold (§4, I3); `fit_predict_oof` applies it with the fold's own
training mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from alphacore.data.lake import DEFAULT_DATA_DIR, read_bars
from alphacore.errors import PipelineError
from alphacore.features import build_features
from alphacore.labels import build_labels, is_binary, to_binary
from alphacore.pipelines.ingest import resolve_calendar
from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from alphacore.config.schema import AlphaConfig
    from alphacore.features import FeatureManifest
    from alphacore.labels import BinariseMethod

__all__ = ["FeatureSet", "build_feature_set", "load_bars", "primary_symbol"]


@dataclass(frozen=True)
class FeatureSet:
    """Features, labels, and the bars they were computed from, mutually aligned.

    Attributes:
        symbol: The instrument these rows describe.
        bars: Validated bars, restricted to the aligned index.
        features: The feature frame, warm-up already dropped.
        labels: The label frame, carrying `label`, `realised_return`, `t_end`
            and `weight`.
        manifest: Column provenance, written to `features.json`.
        binarised: The reduction applied to a three-class label set, or None if
            the labels were already binary.
        class_balance: Class proportions **before** any reduction. The
            three-class balance says more about the barriers than the reduced
            one does, and `binarised` records that a reduction followed.
        mean_uniqueness: Average label uniqueness (§8.2). Captured here because
            `DataFrame.attrs` does not reliably survive every slice, and this
            number belongs in the tearsheet.
    """

    symbol: str
    bars: pd.DataFrame
    features: pd.DataFrame
    labels: pd.DataFrame
    manifest: FeatureManifest
    binarised: str | None
    class_balance: dict[str, float]
    mean_uniqueness: float | None

    @property
    def periods_per_year(self) -> float:
        """Bars per year for this interval and calendar (§11.1)."""
        return self.manifest.periods_per_year


def primary_symbol(config: AlphaConfig) -> str:
    """Return the single symbol a run trades.

    Args:
        config: The resolved configuration.

    Returns:
        The first entry in `data.symbols`.

    Raises:
        PipelineError: If more than one symbol is configured. The engine is
            single-symbol (§12.1), and silently trading only the first of five
            would produce a tearsheet that describes a portfolio nobody asked
            for.
    """
    if len(config.data.symbols) != 1:
        msg = (
            f"This pipeline is single-symbol, but data.symbols names "
            f"{list(config.data.symbols)}. Run one symbol per experiment until the "
            "portfolio backtester lands."
        )
        raise PipelineError(msg)
    return config.data.symbols[0]


def load_bars(
    config: AlphaConfig, *, data_dir: Path | str = DEFAULT_DATA_DIR, symbol: str | None = None
) -> pd.DataFrame:
    """Read one symbol's bars out of the lake.

    Args:
        config: The resolved configuration.
        data_dir: Lake root.
        symbol: Override the symbol. Defaults to the configured one.

    Returns:
        Validated bars over the configured span.

    Raises:
        DataError: If the dataset has not been ingested.
        PipelineError: If the configuration names more than one symbol.
    """
    return read_bars(
        root=data_dir,
        provider=config.data.provider,
        symbol=symbol or primary_symbol(config),
        interval=config.data.interval,
        start=config.data.start,
        end=config.data.end,
    )


def build_feature_set(
    config: AlphaConfig,
    bars: pd.DataFrame,
    *,
    symbol: str | None = None,
    binarise: BinariseMethod = "direction",
) -> FeatureSet:
    """Build features and labels from bars and align them.

    Args:
        config: The resolved configuration.
        bars: Validated bars for one symbol.
        symbol: The symbol these bars describe. Defaults to the configured one.
        binarise: How to reduce a three-class label set. `direction` predicts
            which way it moved; `touch` predicts whether the profit barrier was
            hit. The choice changes what accuracy means, so it is explicit
            rather than inferred.

    Returns:
        The aligned feature set.

    Raises:
        FeatureError: If a spec is unknown or the warm-up consumes the sample.
        LabelError: If the labelling method cannot produce a valid frame.
        PipelineError: If nothing survives the alignment.
    """
    calendar = resolve_calendar(config.data)
    features, manifest = build_features(
        bars, config.features, interval=config.data.interval, calendar=calendar
    )

    labels = build_labels(bars.loc[features.index], config.labels)
    class_balance = dict(labels.attrs.get("class_balance", {}))
    uniqueness = labels.attrs.get("mean_uniqueness")
    reduction: str | None = None
    if not is_binary(labels):
        labels = to_binary(labels, method=binarise)
        reduction = binarise

    common = features.index.intersection(labels.index)
    if common.empty:
        msg = (
            f"No rows survive the alignment of {len(features)} feature rows and "
            f"{len(labels)} labels. The warm-up and the label horizon together consume "
            "the whole sample; ingest more history or shorten one of them."
        )
        raise PipelineError(msg)

    logger.bind(phase="features").info(
        "feature set: {} rows, {} columns, labels via {}",
        len(common),
        manifest.n_columns,
        labels.attrs.get("method", config.labels.method),
    )
    return FeatureSet(
        symbol=symbol or primary_symbol(config),
        bars=bars.loc[common],
        features=features.loc[common],
        labels=labels.loc[common],
        manifest=manifest,
        binarised=reduction,
        class_balance=class_balance,
        mean_uniqueness=None if uniqueness is None else float(uniqueness),
    )
