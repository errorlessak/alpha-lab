"""Labelling and sample weights (§8)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphacore.errors import LabelError
from alphacore.labels.binarise import BinariseMethod, is_binary, to_binary
from alphacore.labels.horizon import horizon_quantile, horizon_return, horizon_sign
from alphacore.labels.meta import meta_label
from alphacore.labels.triple_barrier import triple_barrier
from alphacore.labels.weights import (
    concurrency,
    sample_weights,
    time_decay_weights,
    uniqueness_weights,
)
from alphacore.runtime.logging import logger
from alphacore.schemas.labels import validate_labels

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alphacore.config.schema import LabelsConfig

__all__ = [
    "BinariseMethod",
    "build_labels",
    "concurrency",
    "horizon_quantile",
    "horizon_return",
    "horizon_sign",
    "is_binary",
    "meta_label",
    "sample_weights",
    "time_decay_weights",
    "to_binary",
    "triple_barrier",
    "uniqueness_weights",
    "validate_labels",
]


def build_labels(
    bars: pd.DataFrame,
    config: LabelsConfig,
    *,
    fit_mask: NDArray[np.bool_] | None = None,
    primary_side: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the configured label frame and attach sample weights.

    Args:
        bars: Validated bars.
        config: The `labels` section.
        fit_mask: Training-row mask, required for `horizon_quantile` so bin
            edges are fitted per fold (§8.1).
        primary_side: The primary rule's side per bar, required when
            `meta_labelling` is enabled.

    Returns:
        A validated label frame. `frame.attrs` carries `class_balance` and
        `mean_uniqueness`, both of which land in the tearsheet (§11.4).

    Raises:
        LabelError: If the method is unknown, or a required input is missing.
        LeakageError: If `horizon_quantile` is requested without a fit mask.
    """
    method = config.method
    if method == "horizon_return":
        labels = horizon_return(bars, horizon_bars=config.horizon_bars)
    elif method == "horizon_sign":
        labels = horizon_sign(
            bars,
            horizon_bars=config.horizon_bars,
            min_return=config.triple_barrier.min_return,
        )
    elif method == "horizon_quantile":
        labels = horizon_quantile(
            bars,
            horizon_bars=config.horizon_bars,
            n_bins=config.n_quantile_bins,
            fit_mask=fit_mask,
        )
    elif method == "triple_barrier":
        barrier = config.triple_barrier
        labels = triple_barrier(
            bars,
            profit_mult=barrier.profit_mult,
            loss_mult=barrier.loss_mult,
            vertical_bars=barrier.vertical_bars,
            atr_window=barrier.atr_window,
            min_return=barrier.min_return,
        )
    # No `else`: `LabelsConfig.method` is a Literal covering exactly these four,
    # so mypy proves the chain exhaustive. Adding a fifth method to the Literal
    # without adding a branch here becomes a type error rather than a runtime
    # surprise — a stronger guard than the `raise` it replaces.

    if config.meta_labelling:
        if primary_side is None:
            msg = (
                "labels.meta_labelling is enabled but no primary_side was supplied. "
                "Meta-labelling predicts whether to take the primary rule's trade, so "
                "there has to be a primary rule."
            )
            raise LabelError(msg)
        labels = meta_label(labels, primary_side.reindex(labels.index))

    weights = sample_weights(labels, pd.DatetimeIndex(bars.index), config.sample_weights)
    labels = labels.assign(weight=weights)

    counts = labels["label"].value_counts(normalize=True).sort_index()
    labels.attrs["class_balance"] = {str(k): float(v) for k, v in counts.items()}
    labels.attrs["mean_uniqueness"] = float(
        uniqueness_weights(
            pd.DatetimeIndex(labels.index),
            pd.DatetimeIndex(labels["t_end"]),
            pd.DatetimeIndex(bars.index),
        ).mean()
        / max(
            1.0,
            float(
                concurrency(
                    pd.DatetimeIndex(labels.index),
                    pd.DatetimeIndex(labels["t_end"]),
                    pd.DatetimeIndex(bars.index),
                ).mean()
            ),
        )
    )
    labels.attrs["method"] = method

    logger.bind(phase="labels").info(
        "built {} labels via {}, class balance {}",
        len(labels),
        method,
        labels.attrs["class_balance"],
    )
    return validate_labels(labels)
