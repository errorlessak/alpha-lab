"""Frame-level feature transforms (§7.1).

Operations applied to a computed feature frame rather than to a price array:
lagging, and dropping the warm-up region.

Both are causal by construction. `add_lags` shifts *forward* only — a negative
lag would be a look-ahead, and the function refuses one rather than trusting
the caller. `drop_warmup` removes rows; it never fills them, because a
back-filled warm-up looks like real data to an estimator and biases every
parameter fitted to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from alphacore.errors import FeatureError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["LAG_SUFFIX", "add_lags", "drop_warmup", "lag_column_name"]

LAG_SUFFIX = "_lag"


def lag_column_name(column: str, lag: int) -> str:
    """Build the name of a lagged column.

    Args:
        column: Source column name.
        lag: Number of bars shifted.

    Returns:
        e.g. `'sma_window10_lag2'`.
    """
    return f"{column}{LAG_SUFFIX}{lag}"


def add_lags(frame: pd.DataFrame, lags: Sequence[int]) -> pd.DataFrame:
    """Append lagged copies of every column.

    Args:
        frame: Feature frame.
        lags: Positive lags in bars. Applied in ascending order so column order
            is deterministic regardless of how the config listed them (§4, I8).

    Returns:
        A new frame with the original columns followed by the lagged ones.

    Raises:
        FeatureError: If any lag is below 1. A lag of 0 duplicates a column; a
            negative lag reads the future.
    """
    if not lags:
        return frame.copy()

    invalid = [lag for lag in lags if lag < 1]
    if invalid:
        msg = (
            f"lags must all be >= 1, got {invalid}. A lag of 0 duplicates a column and a "
            "negative lag reads the future (§4, I1)."
        )
        raise FeatureError(msg)

    pieces = [frame]
    for lag in sorted(set(lags)):
        shifted = frame.shift(lag)
        shifted.columns = [lag_column_name(column, lag) for column in frame.columns]
        pieces.append(shifted)
    return pd.concat(pieces, axis=1)


def drop_warmup(frame: pd.DataFrame, n_bars: int) -> pd.DataFrame:
    """Remove the leading warm-up rows.

    Args:
        frame: Feature frame.
        n_bars: Rows to drop from the head.

    Returns:
        A new frame with the first `n_bars` rows removed.

    Raises:
        FeatureError: If `n_bars` is negative.
    """
    if n_bars < 0:
        msg = f"Cannot drop a negative number of warm-up bars, got {n_bars}."
        raise FeatureError(msg)
    return frame.iloc[n_bars:].copy() if n_bars else frame.copy()
