"""Fixed-horizon labelling (§8.1).

Three variants over the same forward window:

* `horizon_return` — the log return itself, a regression target
* `horizon_sign` — its sign, with a dead zone that **drops** rather than
  labels flat
* `horizon_quantile` — bucketed by quantiles fitted **per fold**

The dead zone drops rather than labelling zero because a move too small to
trade is not an observation about direction; keeping it teaches a model that
noise has a sign.

The per-fold quantile requirement is the one that catches people. Fitting bin
edges on the whole sample uses the test period's return distribution to define
what counts as "strong buy" during training — a leak that inflates results
without any obvious look-ahead in the code. `horizon_quantile` therefore refuses
to run without an explicit `fit_mask` (§8.1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphacore.errors import LabelError, LeakageError

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["horizon_quantile", "horizon_return", "horizon_sign"]

_MIN_QUANTILE_BINS = 2
"""Fewer than two bins is not a classification."""


def _forward(
    bars: pd.DataFrame, horizon_bars: int
) -> tuple[pd.DatetimeIndex, NDArray[np.float64], pd.DatetimeIndex]:
    """Compute the forward window and its log return.

    Args:
        bars: Validated bars.
        horizon_bars: Horizon in bars.

    Returns:
        `(t_start, log_return, t_end)`, truncated to the bars that actually
        have a full horizon ahead of them. Rows without one are dropped rather
        than labelled from a shorter window.

    Raises:
        LabelError: If the horizon is not positive or exceeds the series.
    """
    if horizon_bars < 1:
        msg = f"horizon_bars must be at least 1, got {horizon_bars}."
        raise LabelError(msg)
    if horizon_bars >= len(bars):
        msg = f"horizon_bars={horizon_bars} leaves no labelled rows in a {len(bars)}-bar series."
        raise LabelError(msg)

    close = bars["close"].to_numpy(dtype=float)
    t_start = bars.index[:-horizon_bars]
    t_end = bars.index[horizon_bars:]
    returns = np.log(close[horizon_bars:] / close[:-horizon_bars])
    return t_start, returns, t_end


def _frame(
    t_start: pd.DatetimeIndex,
    t_end: pd.DatetimeIndex,
    label: NDArray[np.int64] | NDArray[np.float64],
    realised_return: NDArray[np.float64],
) -> pd.DataFrame:
    """Assemble a label frame with a placeholder weight of 1.0.

    Weights are attached later by `build_labels`, once the full set of label
    spans is known — uniqueness is a property of the whole collection, not of
    any single label.
    """
    return pd.DataFrame(
        {
            "t_end": pd.DatetimeIndex(t_end),
            "label": label,
            "realised_return": realised_return,
            "weight": np.ones(len(t_start), dtype=float),
        },
        index=pd.DatetimeIndex(t_start, name="ts"),
    )


def horizon_return(bars: pd.DataFrame, *, horizon_bars: int) -> pd.DataFrame:
    """Label with the forward log return.

    `label_t = ln(close_{t+h} / close_t)`

    Args:
        bars: Validated bars.
        horizon_bars: Horizon `h` in bars.

    Returns:
        A label frame whose `label` is a float return.

    Raises:
        LabelError: If the horizon is invalid for this series.
    """
    t_start, returns, t_end = _forward(bars, horizon_bars)
    return _frame(t_start, t_end, returns.astype(float), returns)


def horizon_sign(bars: pd.DataFrame, *, horizon_bars: int, min_return: float = 0.0) -> pd.DataFrame:
    """Label with the sign of the forward return, dropping a dead zone.

    `label_t = sign(ln(close_{t+h} / close_t))`, with `|r| < min_return`
    dropped entirely.

    Args:
        bars: Validated bars.
        horizon_bars: Horizon `h` in bars.
        min_return: Absolute log-return below which the observation is dropped.

    Returns:
        A label frame whose `label` is `-1` or `+1`.

    Raises:
        LabelError: If the horizon is invalid, or the dead zone removes
            everything.
    """
    t_start, returns, t_end = _forward(bars, horizon_bars)
    keep = np.abs(returns) >= min_return if min_return > 0 else np.ones(len(returns), bool)
    if not keep.any():
        msg = (
            f"min_return={min_return} removed every observation. Lower the dead zone or "
            "lengthen the horizon."
        )
        raise LabelError(msg)

    labels = np.where(returns > 0, 1, -1).astype(int)
    return _frame(t_start[keep], t_end[keep], labels[keep], returns[keep])


def horizon_quantile(
    bars: pd.DataFrame,
    *,
    horizon_bars: int,
    n_bins: int,
    fit_mask: NDArray[np.bool_] | None,
) -> pd.DataFrame:
    """Bucket the forward return into quantile bins fitted on the fit rows.

    With `n_bins=5` the buckets read strong sell / sell / hold / buy / strong
    buy, numbered `0..4`.

    Args:
        bars: Validated bars.
        horizon_bars: Horizon `h` in bars.
        n_bins: Number of buckets.
        fit_mask: Boolean mask over `bars` naming the rows the edges may be
            fitted on. Required.

    Returns:
        A label frame whose `label` is an integer bucket. The interior bin
        edges are recorded in `frame.attrs["bin_edges"]` so a fold's edges can
        be inspected and compared.

    Raises:
        LeakageError: If `fit_mask` is None or the wrong length. Global
            quantile edges are a classic leak (§8.1, §19).
        LabelError: If the horizon or bin count is invalid.
    """
    if fit_mask is None:
        msg = (
            "horizon_quantile needs an explicit fit_mask naming the training rows. "
            "Quantile edges fitted on the whole sample use the test period's return "
            "distribution to define the classes, which inflates every out-of-sample "
            "score (§8.1). Edges must be fitted per fold."
        )
        raise LeakageError(msg)
    if len(fit_mask) != len(bars):
        msg = f"fit_mask has {len(fit_mask)} entries for a {len(bars)}-bar frame."
        raise LeakageError(msg)
    if n_bins < _MIN_QUANTILE_BINS:
        msg = f"n_bins must be at least {_MIN_QUANTILE_BINS}, got {n_bins}."
        raise LabelError(msg)

    t_start, returns, t_end = _forward(bars, horizon_bars)
    fit_rows = fit_mask[: len(returns)]
    fitted = returns[fit_rows]
    if fitted.size < n_bins:
        msg = (
            f"Only {fitted.size} fit rows for {n_bins} quantile bins; the edges would be "
            "degenerate. Widen the training window."
        )
        raise LabelError(msg)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(fitted, quantiles)
    labels = np.searchsorted(edges, returns, side="right").astype(int)

    frame = _frame(t_start, t_end, labels, returns)
    frame.attrs["bin_edges"] = [float(edge) for edge in edges]
    return frame
