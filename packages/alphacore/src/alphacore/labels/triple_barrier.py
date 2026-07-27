"""Triple-barrier labelling (§8.1).

For each bar `t`, three barriers:

* upper at `close_t * (1 + profit_mult * sigma_t)`
* lower at `close_t * (1 - loss_mult * sigma_t)`
* vertical at `t + vertical_bars`

where `sigma_t` is ATR-based and expressed as a fraction of price. The label is
which barrier is touched first, evaluated against each bar's **high and low**
rather than its close — a barrier that was touched intrabar was touched, and
pretending otherwise flatters the strategy.

**The tie-break.** When a single bar touches both the profit and the loss
barrier, the loss is assumed to fill first. Nothing in a daily bar says which
came first, and assuming the favourable one is exactly how a backtest
manufactures an edge that does not exist (§12.1).

`t_end` is the touch bar, not the vertical barrier. That distinction is what
makes purging correct: a label resolved on bar `t+2` blocks two bars of
training data, not ten.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphacore.errors import LabelError
from alphacore.features.primitives import atr

__all__ = ["triple_barrier"]


def triple_barrier(
    bars: pd.DataFrame,
    *,
    profit_mult: float,
    loss_mult: float,
    vertical_bars: int,
    atr_window: int,
    min_return: float = 0.0,
) -> pd.DataFrame:
    """Label each bar by which barrier its forward window touches first.

    Args:
        bars: Validated bars.
        profit_mult: Upper barrier width in ATR multiples.
        loss_mult: Lower barrier width in ATR multiples.
        vertical_bars: Maximum holding period in bars.
        atr_window: ATR smoothing period defining barrier width.
        min_return: Vertical-barrier outcomes with `|realised_return|` below
            this are labelled 0 rather than by sign.

    Returns:
        A label frame with `label` in `{-1, 0, 1}`, `t_end` at the touch bar,
        and `realised_return` measured to that bar.

    Raises:
        LabelError: If a parameter is invalid, or the ATR warm-up plus the
            vertical barrier leave no labelled rows.
    """
    if vertical_bars < 1:
        msg = f"vertical_bars must be at least 1, got {vertical_bars}."
        raise LabelError(msg)
    if profit_mult <= 0 or loss_mult <= 0:
        msg = f"Barrier multiples must be positive, got profit={profit_mult} loss={loss_mult}."
        raise LabelError(msg)

    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = atr(high, low, close, atr_window) / close

    n = len(bars)
    usable = np.flatnonzero(np.isfinite(sigma) & (sigma > 0))
    usable = usable[usable + 1 < n]
    if usable.size == 0:
        msg = (
            f"No labelled rows: an ATR window of {atr_window} plus a vertical barrier of "
            f"{vertical_bars} consumes a {n}-bar series."
        )
        raise LabelError(msg)

    starts: list[int] = []
    ends: list[int] = []
    labels: list[int] = []

    for raw_index in usable:
        i = int(raw_index)
        upper = close[i] * (1.0 + profit_mult * sigma[i])
        lower = close[i] * (1.0 - loss_mult * sigma[i])
        last = min(i + vertical_bars, n - 1)

        touch = last
        outcome = 0
        for j in range(i + 1, last + 1):
            hit_lower = low[j] <= lower
            hit_upper = high[j] >= upper
            if hit_lower or hit_upper:
                # Both in one bar: the stop is assumed to fill first (§12.1).
                outcome = -1 if hit_lower else 1
                touch = j
                break

        starts.append(i)
        ends.append(touch)
        labels.append(outcome)

    start_idx = np.asarray(starts)
    end_idx = np.asarray(ends)
    realised = np.log(close[end_idx] / close[start_idx])

    label_array = np.asarray(labels, dtype=int)
    expired = label_array == 0
    if min_return > 0:
        # A vertical-barrier outcome larger than the dead zone still carries
        # directional information, so it is labelled by sign rather than flat.
        label_array = np.where(
            expired & (np.abs(realised) >= min_return),
            np.sign(realised).astype(int),
            label_array,
        )

    return pd.DataFrame(
        {
            "t_end": bars.index[end_idx],
            "label": label_array,
            "realised_return": realised,
            "weight": np.ones(len(start_idx), dtype=float),
        },
        index=pd.DatetimeIndex(bars.index[start_idx], name="ts"),
    )
