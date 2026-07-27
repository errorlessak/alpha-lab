"""Sample weights for overlapping labels (§8.2).

Every estimator assumes its observations are independent. Labels spanning
`[t, t+10]` on daily bars are anything but: ten consecutive observations share
nine bars of the same price path. Left alone, that inflates the effective
sample size roughly tenfold, which makes every standard error too small, every
significance test too generous, and every fitted model too confident.

Two corrections, multiplied together:

* **Uniqueness** — for a label spanning `[t_start, t_end]`, average `1/c_u`
  over the bars `u` in its span, where `c_u` counts labels live at `u`. A label
  sharing every bar with nine others contributes a tenth as much.
* **Time decay** — linear in cumulative uniqueness, from `decay` at the oldest
  observation to `1.0` at the newest. Markets change; a 2007 observation is
  evidence about 2007.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphacore.errors import LabelError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alphacore.config.schema import SampleWeightsConfig

__all__ = ["concurrency", "sample_weights", "time_decay_weights", "uniqueness_weights"]


def concurrency(
    t_start: pd.DatetimeIndex, t_end: pd.DatetimeIndex, bar_index: pd.DatetimeIndex
) -> NDArray[np.int64]:
    """Count how many labels are live at each bar.

    Args:
        t_start: Label start timestamps.
        t_end: Label end timestamps.
        bar_index: The full bar index the labels live on.

    Returns:
        A count per bar in `bar_index`.

    Raises:
        LabelError: If a label timestamp is not on the bar index.
    """
    starts = bar_index.get_indexer(t_start)
    ends = bar_index.get_indexer(t_end)
    if (starts < 0).any() or (ends < 0).any():
        msg = "Every label's t_start and t_end must be a timestamp present in the bar index."
        raise LabelError(msg)

    # A difference array turns per-label span increments into one pass.
    deltas = np.zeros(len(bar_index) + 1, dtype=np.int64)
    np.add.at(deltas, starts, 1)
    np.add.at(deltas, ends + 1, -1)
    return np.cumsum(deltas)[:-1]


def uniqueness_weights(
    t_start: pd.DatetimeIndex, t_end: pd.DatetimeIndex, bar_index: pd.DatetimeIndex
) -> NDArray[np.float64]:
    """Average inverse concurrency over each label's span, normalised to mean 1.

    `w_i = mean(1 / c_u)` for `u` in `[t_start_i, t_end_i]`.

    Normalising to mean 1 keeps the effective learning rate of a weighted
    estimator comparable to the unweighted case: the correction changes the
    *relative* importance of observations, not the overall scale.

    Args:
        t_start: Label start timestamps.
        t_end: Label end timestamps.
        bar_index: The full bar index.

    Returns:
        One weight per label, averaging 1.0.

    Raises:
        LabelError: If a label timestamp is not on the bar index.
    """
    counts = concurrency(t_start, t_end, bar_index)
    inverse = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(inverse)))

    starts = bar_index.get_indexer(t_start)
    ends = bar_index.get_indexer(t_end)
    spans = (ends - starts + 1).astype(float)
    totals = cumulative[ends + 1] - cumulative[starts]

    weights = totals / spans
    mean = float(weights.mean())
    return weights / mean if mean > 0 else np.ones_like(weights)


def time_decay_weights(uniqueness: NDArray[np.float64], decay: float) -> NDArray[np.float64]:
    """Apply linear time decay in cumulative uniqueness.

    The decay is linear in *cumulative uniqueness* rather than in bar count, so
    a stretch of heavily overlapping labels ages the schedule less than a
    stretch of independent ones — which is the correct behaviour, because it
    carries less information.

    Args:
        uniqueness: Per-label uniqueness weights, oldest first.
        decay: Weight of the oldest observation. `1.0` disables decay; `0.0`
            gives the oldest observation zero weight.

    Returns:
        A decay factor per label, ending at 1.0 for the newest.

    Raises:
        LabelError: If `decay` is outside `[0, 1]`.
    """
    if not 0.0 <= decay <= 1.0:
        msg = f"time_decay must be in [0, 1], got {decay}."
        raise LabelError(msg)
    if uniqueness.size == 0:
        return uniqueness.astype(float)
    if decay == 1.0:
        return np.ones_like(uniqueness, dtype=float)

    cumulative = np.cumsum(uniqueness).astype(float)
    total = float(cumulative[-1])
    if total <= 0:
        return np.ones_like(uniqueness, dtype=float)

    slope = (1.0 - decay) / total
    intercept = 1.0 - slope * total
    return np.clip(intercept + slope * cumulative, 0.0, None)


def sample_weights(
    labels: pd.DataFrame, bar_index: pd.DatetimeIndex, config: SampleWeightsConfig
) -> NDArray[np.float64]:
    """Compute the final per-label sample weight.

    The product of uniqueness and time decay, passed to every estimator's
    `sample_weight` (§8.2).

    Args:
        labels: A label frame carrying `t_end`.
        bar_index: The full bar index the labels live on.
        config: The `labels.sample_weights` section.

    Returns:
        One strictly positive weight per label.

    Raises:
        LabelError: If a label timestamp is not on the bar index, or the decay
            is out of range.
    """
    n = len(labels)
    if n == 0:
        return np.zeros(0, dtype=float)

    if config.uniqueness:
        weights = uniqueness_weights(
            pd.DatetimeIndex(labels.index), pd.DatetimeIndex(labels["t_end"]), bar_index
        )
    else:
        weights = np.ones(n, dtype=float)

    weights = weights * time_decay_weights(weights, config.time_decay)

    # A zero weight silently removes an observation without recording that it
    # went; a tiny floor keeps it in the sample and visible in the manifest.
    return np.maximum(weights, np.finfo(float).tiny)
