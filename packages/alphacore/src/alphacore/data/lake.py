"""The point-in-time data lake (§6.3).

Layout: `{root}/{provider}/{symbol}/{interval}/year=YYYY/bars.parquet`, ZSTD
compressed. Partitioning by year keeps a twenty-year daily series in twenty
small files, so re-ingesting 2025 does not rewrite 2005.

The lake is append-and-replace at year granularity, never in-place edit: a
re-pull replaces whole year files. The `content_hash` returned by `write_bars`
is what lands in the catalog and, from there, in the run manifest. A run pinned
to a hash either finds that exact data or fails loudly — which is the entire
point of storing a hash rather than a row count.
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alphacore.errors import DataError
from alphacore.runtime.hashing import hash_files
from alphacore.schemas.bars import BAR_COLUMNS, TS_DTYPE, TS_INDEX_NAME, validate_bars

if TYPE_CHECKING:
    from alphacore.config.schema import Interval

__all__ = [
    "BARS_FILENAME",
    "COMPRESSION",
    "DEFAULT_DATA_DIR",
    "dataset_dir",
    "dataset_id",
    "parse_dataset_id",
    "read_bars",
    "write_bars",
]

DEFAULT_DATA_DIR: Final[Path] = Path("data")
BARS_FILENAME: Final[str] = "bars.parquet"
COMPRESSION: Final[str] = "zstd"
COMPRESSION_LEVEL: Final[int] = 3

_RAW_SUBDIR: Final[str] = "raw"


def dataset_id(provider: str, symbol: str, interval: Interval) -> str:
    """Build the catalog identifier for a dataset.

    Args:
        provider: Provider name.
        symbol: Instrument identifier.
        interval: Bar interval.

    Returns:
        An identifier of the form `'yahoo:SPY:1d'`.
    """
    return f"{provider}:{symbol}:{interval.value}"


def parse_dataset_id(value: str) -> tuple[str, str, str]:
    """Split a dataset identifier into its parts.

    Args:
        value: An identifier such as `'yahoo:SPY:1d'`.

    Returns:
        `(provider, symbol, interval)` as strings.

    Raises:
        DataError: If the identifier does not have exactly three parts.
    """
    parts = value.split(":")
    expected_parts = 3
    if len(parts) != expected_parts or not all(parts):
        msg = f"Malformed dataset id {value!r}; expected 'provider:symbol:interval'."
        raise DataError(msg)
    return parts[0], parts[1], parts[2]


def dataset_dir(root: Path | str, provider: str, symbol: str, interval: Interval) -> Path:
    """Resolve the directory holding one dataset's parquet files.

    Args:
        root: Lake root, normally `data/`.
        provider: Provider name.
        symbol: Instrument identifier.
        interval: Bar interval.

    Returns:
        `{root}/raw/{provider}/{symbol}/{interval}`.
    """
    return Path(root) / _RAW_SUBDIR / provider / symbol / interval.value


def _year_partitions(bars: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    """Split bars into per-year chunks, in ascending year order.

    Args:
        bars: Validated bars.

    Returns:
        `(year, frame)` pairs, sorted. Sorting matters: it is what makes the
        set of written files, and therefore the content hash, independent of
        the order pandas happened to group them in (§4, I8).
    """
    years = pd.Index(bars.index.year)
    return [(int(year), bars[years == year]) for year in sorted(years.unique())]


def write_bars(
    bars: pd.DataFrame,
    *,
    root: Path | str,
    provider: str,
    symbol: str,
    interval: Interval,
    replace: bool = True,
) -> tuple[list[Path], str]:
    """Write bars to the lake, partitioned by year.

    Args:
        bars: Validated bars for one symbol.
        root: Lake root.
        provider: Provider name.
        symbol: Instrument identifier.
        interval: Bar interval.
        replace: Remove any existing partitions for this dataset first. True is
            the honest default — a partial overwrite would leave the directory
            holding a mixture of two pulls whose content hash describes neither.

    Returns:
        The paths written, sorted, and the dataset's `content_hash`.

    Raises:
        DataError: If `bars` is empty. An empty write would produce a catalog
            row for a dataset that does not exist.
        SchemaValidationError: If `bars` does not satisfy the bar schema.
    """
    if bars.empty:
        msg = f"Refusing to write an empty dataset for {dataset_id(provider, symbol, interval)}."
        raise DataError(msg)

    validated = validate_bars(bars, check_ohlc_ordering=False)
    directory = dataset_dir(root, provider, symbol, interval)
    if replace and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for year, chunk in _year_partitions(validated):
        partition = directory / f"year={year}"
        partition.mkdir(parents=True, exist_ok=True)
        path = partition / BARS_FILENAME
        table = pa.Table.from_pandas(chunk, preserve_index=True)
        pq.write_table(
            table,
            path,
            compression=COMPRESSION,
            compression_level=COMPRESSION_LEVEL,
            version="2.6",
        )
        written.append(path)

    written.sort()
    return written, hash_files(written)


def read_bars(
    *,
    root: Path | str,
    provider: str,
    symbol: str,
    interval: Interval,
    start: dt.date | dt.datetime | str | None = None,
    end: dt.date | dt.datetime | str | None = None,
) -> pd.DataFrame:
    """Read a dataset back out of the lake.

    Args:
        root: Lake root.
        provider: Provider name.
        symbol: Instrument identifier.
        interval: Bar interval.
        start: Optional inclusive lower bound.
        end: Optional inclusive upper bound.

    Returns:
        Validated bars, sorted by timestamp.

    Raises:
        DataError: If the dataset is not in the lake.
        SchemaValidationError: If what was stored no longer satisfies the
            schema, which would mean the files were edited outside the lake.
    """
    directory = dataset_dir(root, provider, symbol, interval)
    files = sorted(directory.glob(f"year=*/{BARS_FILENAME}"))
    if not files:
        msg = (
            f"No data in the lake for {dataset_id(provider, symbol, interval)} "
            f"(looked in {directory}). Run `alpha data pull` first."
        )
        raise DataError(msg)

    frames = [pq.read_table(path).to_pandas() for path in files]
    bars = pd.concat(frames).sort_index(kind="stable")
    bars.index = pd.DatetimeIndex(bars.index).astype(TS_DTYPE)
    bars.index.name = TS_INDEX_NAME
    bars = bars.loc[:, list(BAR_COLUMNS)]
    bars["symbol"] = bars["symbol"].astype("category")

    if start is not None:
        bars = bars.loc[bars.index >= _bound(start)]
    if end is not None:
        bars = bars.loc[bars.index <= _bound(end)]
    return validate_bars(bars, check_ohlc_ordering=False)


def _bound(value: dt.date | dt.datetime | str) -> pd.Timestamp:
    """Normalise a span bound to a UTC timestamp.

    Args:
        value: A date, datetime, or ISO string.

    Returns:
        A tz-aware UTC `pd.Timestamp`.
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")
