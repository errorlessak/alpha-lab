"""Yahoo Finance provider via `yfinance` (§6.1).

Two decisions worth stating, because both are places where a naive integration
quietly corrupts a backtest.

**Adjustment (I7).** `auto_adjust=False` is used deliberately. Yahoo's adjusted
series is restated every time a dividend or split is announced, so the same
request returns different numbers next month and the run stops being
reproducible. Worse, the adjusted open is a price that never traded, and the
backtester fills at the open. So unadjusted OHLC is stored, and the adjustment
travels as its own `adj_factor` column — which is exactly what
`adjustment: unadjusted_plus_factor` means.

**Truncation.** Yahoo caps intraday history and returns a short series rather
than an error. `max_lookback` publishes those caps so the caller can warn
instead of silently backtesting a month of data it thinks is twenty years.

`yfinance` is imported lazily, inside `fetch`. It pulls in `requests`,
`curl_cffi`, and `beautifulsoup4`; none of that belongs in the import path of a
library whose job is arithmetic.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from alphacore.data.providers.base import retry
from alphacore.errors import ProviderError
from alphacore.runtime.logging import logger
from alphacore.schemas.bars import normalise_bars, validate_bars

if TYPE_CHECKING:
    from alphacore.config.schema import Interval

__all__ = ["MAX_LOOKBACK", "YahooProvider"]

_YF_INTERVAL: Final[dict[str, str]] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "1h",  # Yahoo has no native 4h; resampled from 1h below.
    "1d": "1d",
}

MAX_LOOKBACK: Final[dict[str, dt.timedelta | None]] = {
    "1m": dt.timedelta(days=30),
    "5m": dt.timedelta(days=60),
    "15m": dt.timedelta(days=60),
    "1h": dt.timedelta(days=730),
    "4h": dt.timedelta(days=730),
    "1d": None,
}
"""Yahoo's published history limits per interval. These are the caps that make
a request for five years of one-minute bars return thirty days with no error."""

_RESAMPLE_RULE: Final[dict[str, str]] = {"4h": "4h"}


class YahooProvider:
    """Fetches bars from Yahoo Finance."""

    name = "yahoo"

    def __init__(self, *, max_attempts: int = 5) -> None:
        """Configure the provider.

        Args:
            max_attempts: Total fetch attempts before giving up.
        """
        self.max_attempts = max_attempts

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"YahooProvider(max_attempts={self.max_attempts})"

    def max_lookback(self, interval: Interval) -> dt.timedelta | None:
        """Report how far back Yahoo serves this interval.

        Args:
            interval: Bar interval.

        Returns:
            The history cap, or None for daily bars which are unrestricted.
        """
        return MAX_LOOKBACK.get(interval.value)

    def _download(
        self, symbol: str, yf_interval: str, start: dt.datetime, end: dt.datetime
    ) -> pd.DataFrame:
        """Call yfinance once.

        Args:
            symbol: Instrument identifier.
            yf_interval: Yahoo's interval code.
            start: First timestamp wanted.
            end: Exclusive upper bound passed to Yahoo.

        Returns:
            Yahoo's raw frame.

        Raises:
            ProviderError: If yfinance is not installed.
        """
        try:
            import yfinance  # noqa: PLC0415 - deliberately lazy; see the module docstring
        except ImportError as exc:  # pragma: no cover - yfinance is a hard dependency
            msg = "yfinance is required for the yahoo provider but is not installed."
            raise ProviderError(msg) from exc

        frame: Any = yfinance.download(
            tickers=symbol,
            start=start.date().isoformat(),
            end=(end.date() + dt.timedelta(days=1)).isoformat(),
            interval=yf_interval,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        return pd.DataFrame(frame)

    def fetch(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
    ) -> pd.DataFrame:
        """Fetch bars for one symbol.

        Args:
            symbol: Instrument identifier.
            interval: Bar interval.
            start: First timestamp wanted, inclusive.
            end: Last timestamp wanted, inclusive.

        Returns:
            A validated frame satisfying `RAW_BARS_SCHEMA`, carrying unadjusted
            OHLC and a separate `adj_factor`.

        Raises:
            ProviderError: If the interval is unsupported, or every attempt
                fails, or Yahoo returns nothing for the span.
            SchemaValidationError: If the response cannot be made
                schema-conformant.
        """
        yf_interval = _YF_INTERVAL.get(interval.value)
        if yf_interval is None:
            msg = f"Yahoo does not serve interval {interval.value!r}."
            raise ProviderError(msg)

        self._warn_if_truncated(symbol, interval, start)

        raw = retry(
            lambda: self._download(symbol, yf_interval, start, end),
            description=f"yahoo fetch {symbol} {interval.value}",
            max_attempts=self.max_attempts,
        )
        if raw.empty:
            msg = (
                f"Yahoo returned no bars for {symbol!r} at {interval.value} between "
                f"{start.date()} and {end.date()}. Check the symbol and the span."
            )
            raise ProviderError(msg)

        bars = validate_bars(normalise_bars(self._reshape(raw, symbol), symbol=symbol))
        if (rule := _RESAMPLE_RULE.get(interval.value)) is not None:
            bars = validate_bars(_resample(bars, rule, symbol))
        return bars

    def _warn_if_truncated(self, symbol: str, interval: Interval, start: dt.datetime) -> None:
        """Log loudly when the request predates Yahoo's history horizon.

        Args:
            symbol: Instrument identifier.
            interval: Bar interval.
            start: First timestamp wanted.
        """
        cap = self.max_lookback(interval)
        if cap is None:
            return
        horizon = dt.datetime.now(dt.UTC) - cap
        requested = start if start.tzinfo else start.replace(tzinfo=dt.UTC)
        if requested < horizon:
            logger.bind(phase="ingest").warning(
                "{} {} requested from {} but Yahoo only serves back to {} — the series "
                "will be truncated, and every metric computed on it describes a shorter "
                "sample than the config claims.",
                symbol,
                interval.value,
                requested.date(),
                horizon.date(),
            )

    @staticmethod
    def _reshape(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Flatten Yahoo's response into canonical columns.

        Yahoo returns a MultiIndex column frame when `tickers` is a list and a
        flat one otherwise, and names the adjusted series `Adj Close`. The
        adjustment factor is recovered as `adj_close / close`.

        Args:
            raw: Yahoo's frame.
            symbol: Instrument identifier.

        Returns:
            A frame with `open`, `high`, `low`, `close`, `volume`,
            `adj_factor`, `symbol`.

        Raises:
            SchemaValidationError: If a required column is absent.
        """
        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [str(level[0]) for level in frame.columns]
        frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]

        if "adj_close" in frame.columns and "close" in frame.columns:
            close = pd.to_numeric(frame["close"], errors="coerce")
            adj_close = pd.to_numeric(frame["adj_close"], errors="coerce")
            frame["adj_factor"] = (adj_close / close).fillna(1.0)
            frame = frame.drop(columns=["adj_close"])
        else:
            frame["adj_factor"] = 1.0

        frame["symbol"] = symbol
        frame.index.name = "ts"
        return frame


def _resample(bars: pd.DataFrame, rule: str, symbol: str) -> pd.DataFrame:
    """Aggregate bars to a coarser interval.

    Args:
        bars: Validated finer-grained bars.
        rule: A pandas resample rule, e.g. `'4h'`.
        symbol: Instrument identifier for the rebuilt `symbol` column.

    Returns:
        Resampled bars, left-labelled and left-closed so the timestamp remains
        the bar's opening moment — the convention the whole system depends on
        (see `alphacore.data.calendars`).
    """
    grouped = bars.resample(rule, label="left", closed="left")
    out = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
            "adj_factor": grouped["adj_factor"].last(),
        }
    ).dropna(subset=["open", "high", "low", "close"])
    out["symbol"] = symbol
    return normalise_bars(out, symbol=symbol)
