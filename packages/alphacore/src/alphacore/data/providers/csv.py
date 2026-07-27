"""Local CSV provider (§6.1).

The offline provider. Every test that needs bars uses it, the committed
fixtures under `samples/` are read through it, and the docker demo runs on it —
so `git clone` to rendered tearsheet never depends on a vendor being up.

Files are looked up as `{root}/{symbol}_{interval}.csv`, falling back to
`{root}/{symbol}.csv`. Column names are matched case-insensitively, so a file
exported from anywhere sensible works without editing.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from alphacore.errors import ProviderError
from alphacore.schemas.bars import normalise_bars, validate_bars

if TYPE_CHECKING:
    from alphacore.config.schema import Interval

__all__ = ["CsvProvider"]


class CsvProvider:
    """Reads bars from a directory of CSV files."""

    name = "csv"

    def __init__(self, root: Path | str) -> None:
        """Bind the provider to a directory.

        Args:
            root: Directory holding the CSV files.
        """
        self.root = Path(root)

    def __repr__(self) -> str:
        """Return a debug representation naming the source directory."""
        return f"CsvProvider({self.root.as_posix()!r})"

    def path_for(self, symbol: str, interval: Interval) -> Path:
        """Resolve the file backing one symbol and interval.

        Args:
            symbol: Instrument identifier.
            interval: Bar interval.

        Returns:
            The path to read.

        Raises:
            ProviderError: If neither the interval-qualified nor the bare file
                exists. The message lists what the directory does contain,
                because a typo in a symbol is the usual cause.
        """
        candidates = [
            self.root / f"{symbol}_{interval.value}.csv",
            self.root / f"{symbol}.csv",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        present = sorted(p.name for p in self.root.glob("*.csv")) if self.root.is_dir() else []
        msg = (
            f"No CSV for symbol {symbol!r} at {interval.value} in {self.root}. "
            f"Looked for {[c.name for c in candidates]}; directory contains {present}."
        )
        raise ProviderError(msg)

    def fetch(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
    ) -> pd.DataFrame:
        """Read bars for one symbol and clip them to the requested span.

        Args:
            symbol: Instrument identifier.
            interval: Bar interval.
            start: First timestamp wanted, inclusive.
            end: Last timestamp wanted, inclusive.

        Returns:
            A validated frame satisfying `RAW_BARS_SCHEMA`.

        Raises:
            ProviderError: If the file is missing or unreadable.
            SchemaValidationError: If the file's contents cannot be made
                schema-conformant.
        """
        path = self.path_for(symbol, interval)
        try:
            raw = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            msg = f"Could not read {path}: {exc}"
            raise ProviderError(msg) from exc

        bars = validate_bars(normalise_bars(raw, symbol=symbol))
        lower = (
            pd.Timestamp(start).tz_convert("UTC")
            if pd.Timestamp(start).tz
            else (pd.Timestamp(start).tz_localize("UTC"))
        )
        upper = (
            pd.Timestamp(end).tz_convert("UTC")
            if pd.Timestamp(end).tz
            else (pd.Timestamp(end).tz_localize("UTC"))
        )
        return bars.loc[(bars.index >= lower) & (bars.index <= upper)]

    def max_lookback(self, interval: Interval) -> dt.timedelta | None:  # noqa: ARG002
        """Report the provider's history limit.

        Args:
            interval: Bar interval. Unused — a local file is as old as it is.

        Returns:
            None. A CSV has whatever history it has, and clipping is the
            caller's business.
        """
        return None
