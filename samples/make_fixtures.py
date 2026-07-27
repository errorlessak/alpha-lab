"""Regenerate the committed sample fixtures.

The fixtures are **synthetic**, deliberately. Committing real vendor history
would raise a licensing question and would still go stale; a synthetic series
generated from a fixed seed is redistributable, reproducible, and — because it
is aligned to real NYSE sessions — exercises the calendar and quality machinery
exactly as vendor data would.

`SYNTH` is a geometric random walk with volatility clustering, drifting at
roughly 7% a year. It has no predictable structure by construction, which makes
it the right series for the honest headline result in §19: a pipeline run on it
must find no edge.

Run with `uv run python samples/make_fixtures.py`. Output is byte-identical
across machines given the same seed and the same `exchange_calendars` release.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from alphacore.config.schema import Interval
from alphacore.data.calendars import expected_bar_index

SEED: Final[int] = 20260727
START: Final[str] = "2015-01-02"
END: Final[str] = "2024-12-31"
SYMBOL: Final[str] = "SYNTH"
CALENDAR: Final[str] = "XNYS"

INITIAL_PRICE: Final[float] = 100.0
ANNUAL_DRIFT: Final[float] = 0.07
BASE_ANNUAL_VOL: Final[float] = 0.16
TRADING_DAYS: Final[int] = 252

VOL_PERSISTENCE: Final[float] = 0.94
VOL_SHOCK: Final[float] = 0.12
INTRABAR_RANGE: Final[float] = 0.6
BASE_VOLUME: Final[float] = 8.0e7
VOLUME_VOL_BETA: Final[float] = 3.0

OUTPUT = Path(__file__).parent / f"{SYMBOL}_1d.csv"


def build_bars() -> pd.DataFrame:
    """Generate a synthetic daily OHLCV series on the NYSE session grid.

    Returns:
        A frame with a `ts` column and canonical bar columns, guaranteed to
        satisfy `low <= min(open, close) <= max(open, close) <= high`.
    """
    rng = np.random.default_rng(SEED)
    sessions = expected_bar_index(START, END, Interval.D1, CALENDAR)
    n = len(sessions)

    daily_drift = ANNUAL_DRIFT / TRADING_DAYS
    base_sigma = BASE_ANNUAL_VOL / np.sqrt(TRADING_DAYS)

    # Volatility clusters: log-vol follows an AR(1), so calm and turbulent
    # regimes persist the way they do in real returns.
    log_vol = np.empty(n)
    log_vol[0] = np.log(base_sigma)
    shocks = rng.standard_normal(n)
    for i in range(1, n):
        log_vol[i] = (
            VOL_PERSISTENCE * log_vol[i - 1]
            + (1 - VOL_PERSISTENCE) * np.log(base_sigma)
            + VOL_SHOCK * shocks[i]
        )
    sigma = np.exp(log_vol)

    returns = daily_drift - 0.5 * sigma**2 + sigma * rng.standard_normal(n)
    close = INITIAL_PRICE * np.exp(np.cumsum(returns))

    prev_close = np.concatenate(([INITIAL_PRICE], close[:-1]))
    gap = sigma * rng.standard_normal(n) * 0.3
    open_ = prev_close * np.exp(gap)

    # Build the wick around the bar body so the OHLC ordering invariant holds
    # by construction rather than by luck.
    body_low = np.minimum(open_, close)
    body_high = np.maximum(open_, close)
    span = sigma * INTRABAR_RANGE
    low = body_low * np.exp(-np.abs(rng.standard_normal(n)) * span)
    high = body_high * np.exp(np.abs(rng.standard_normal(n)) * span)

    volume = BASE_VOLUME * np.exp(
        VOLUME_VOL_BETA * (sigma - base_sigma) / base_sigma + rng.standard_normal(n) * 0.2
    )

    return pd.DataFrame(
        {
            "ts": sessions.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": np.round(open_, 4),
            "high": np.round(high, 4),
            "low": np.round(low, 4),
            "close": np.round(close, 4),
            "volume": np.round(volume, 0),
            "adj_factor": 1.0,
            "symbol": SYMBOL,
        }
    )


def main() -> None:
    """Write the fixture CSV."""
    bars = build_bars()
    OUTPUT.write_text(bars.to_csv(index=False, lineterminator="\n"), encoding="utf-8")


if __name__ == "__main__":
    main()
