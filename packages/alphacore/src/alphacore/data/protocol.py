"""The provider contract (§6.1).

Everything downstream of ingestion depends on one guarantee: whatever a
provider returns, it satisfies `schemas.bars.RAW_BARS_SCHEMA`. A provider that
cannot honour that raises rather than returning something approximate.

`max_lookback` exists because vendors silently truncate. Yahoo will not serve
one-minute bars older than about thirty days, and a request for five years of
them returns thirty days of data with no error at all. A run whose config asks
for 2005 to 2025 and whose lake holds a month of it is not a small problem — it is
every metric in the tearsheet being computed on the wrong sample.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    from alphacore.config.schema import Interval

__all__ = ["BarProvider"]


@runtime_checkable
class BarProvider(Protocol):
    """Fetches raw bars for one symbol over one span."""

    name: str

    def fetch(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
    ) -> pd.DataFrame:
        """Fetch bars for a single symbol.

        Args:
            symbol: The instrument identifier, in the provider's own namespace.
            interval: Bar interval.
            start: First timestamp wanted, inclusive, UTC.
            end: Last timestamp wanted, inclusive, UTC.

        Returns:
            A frame satisfying `schemas.bars.RAW_BARS_SCHEMA`: a UTC nanosecond
            index named `ts`, and columns `open`, `high`, `low`, `close`,
            `volume`, `adj_factor`, `symbol`.

        Raises:
            ProviderError: If the provider cannot serve the request after
                retries, or returns data that cannot be made schema-conformant.
        """
        ...

    def max_lookback(self, interval: Interval) -> dt.timedelta | None:
        """Report how far back this provider will serve a given interval.

        Args:
            interval: Bar interval.

        Returns:
            The maximum age of the oldest available bar, or None when the
            provider imposes no limit. Callers warn when a config's `start`
            precedes this horizon, rather than silently accepting a truncated
            series.
        """
        ...
