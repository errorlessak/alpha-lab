"""A deliberately dumb vectorised backtester, for cross-checking (§12.5).

Positions shifted by `execution_lag`, returns multiplied, no stops, no
participation cap, no carry. Nothing here is clever, and that is the point: it
is short enough to verify by reading, so when it disagrees with the event-driven
engine under a frictionless configuration, the engine is wrong.

The two must agree to `1e-8`. Any divergence is an accounting bug — the class of
bug that never announces itself and shows up only as a Sharpe ratio that is
slightly too good.

This is a test oracle. It is not exported for use in a real run, and it charges
no costs, so nothing it produces should reach a tearsheet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphacore.errors import BacktestError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alphacore.config.schema import BacktestConfig

__all__ = ["reference_backtest"]


def reference_backtest(
    bars: pd.DataFrame,
    *,
    signal: NDArray[np.int_],
    config: BacktestConfig,
) -> pd.DataFrame:
    """Compound open-to-open returns against a lagged position.

    The engine rebalances to a target weight at each bar's open, so the
    reference compounds **open-to-open** returns rather than close-to-close.
    Getting that wrong would make the two disagree by exactly one bar's return,
    which looks like a subtle bug and is actually a specification difference.

    Args:
        bars: Validated bars.
        signal: Desired position per bar.
        config: The `backtest` section. Only `initial_capital` and
            `execution_lag_bars` are honoured; costs are ignored by design.

    Returns:
        A frame indexed by `ts` with an `equity` column.

    Raises:
        BacktestError: If the signal is misaligned.
    """
    n = len(bars)
    positions = np.asarray(signal, dtype=float)
    if positions.size != n:
        msg = f"signal has {positions.size} entries for {n} bars."
        raise BacktestError(msg)

    open_ = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)

    # The weight held *into* bar i is the signal from `lag` bars earlier.
    held = np.concatenate((np.zeros(config.execution_lag_bars), positions))[:n]

    # Bar i's return runs from its own open to its own close, then from that
    # close to the next open. Marking at the close each bar makes the
    # open-to-close leg the one that accrues within the bar.
    prices = np.concatenate((open_[:1], close))
    returns = np.diff(prices) / prices[:-1]

    equity = np.empty(n)
    value = float(config.initial_capital)
    for i in range(n):
        value *= 1.0 + held[i] * returns[i]
        equity[i] = value

    return pd.DataFrame({"equity": equity}, index=pd.DatetimeIndex(bars.index, name="ts"))
