"""The dataset catalog (§6.3).

A duckdb file, `data/catalog.duckdb`, holding one row per ingested dataset:
what it is, when it was pulled, how many rows it has, what its quality verdict
was, and — the load-bearing column — its `content_hash`.

The catalog is what lets a run say "I was computed against *this* data" and be
checked. Without it, reproducibility is a claim about a filename.

duckdb is used rather than SQLite because the catalog is read alongside parquet
by the API, and duckdb queries parquet natively. It is a single embedded file
either way; no server, nothing to run.
"""

from __future__ import annotations

import datetime as dt
from contextlib import closing
from pathlib import Path
from typing import Any, Final

import duckdb
from pydantic import BaseModel, ConfigDict

from alphacore.errors import DataError

__all__ = ["CATALOG_FILENAME", "Catalog", "DatasetRecord"]

CATALOG_FILENAME: Final[str] = "catalog.duckdb"

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id       VARCHAR PRIMARY KEY,
    provider         VARCHAR NOT NULL,
    symbol           VARCHAR NOT NULL,
    interval         VARCHAR NOT NULL,
    start_ts         TIMESTAMPTZ NOT NULL,
    end_ts           TIMESTAMPTZ NOT NULL,
    n_rows           BIGINT NOT NULL,
    content_hash     VARCHAR NOT NULL,
    ingested_at      TIMESTAMPTZ NOT NULL,
    provider_version VARCHAR,
    calendar         VARCHAR NOT NULL,
    adjustment       VARCHAR NOT NULL,
    quality_status   VARCHAR NOT NULL,
    quality_summary  VARCHAR
);
"""

_COLUMNS: Final[tuple[str, ...]] = (
    "dataset_id",
    "provider",
    "symbol",
    "interval",
    "start_ts",
    "end_ts",
    "n_rows",
    "content_hash",
    "ingested_at",
    "provider_version",
    "calendar",
    "adjustment",
    "quality_status",
    "quality_summary",
)


class DatasetRecord(BaseModel):
    """One row of the catalog: a dataset as ingested at a point in time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    provider: str
    symbol: str
    interval: str
    start_ts: dt.datetime
    end_ts: dt.datetime
    n_rows: int
    content_hash: str
    ingested_at: dt.datetime
    provider_version: str | None = None
    calendar: str
    adjustment: str
    quality_status: str
    quality_summary: str | None = None


class Catalog:
    """Read/write access to the duckdb dataset catalog."""

    def __init__(self, path: Path | str) -> None:
        """Bind to a catalog file, creating it and its schema if absent.

        Args:
            path: Path to the duckdb file, normally `data/catalog.duckdb`.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(_SCHEMA)

    def __repr__(self) -> str:
        """Return a debug representation naming the catalog file."""
        return f"Catalog({self.path.as_posix()!r})"

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Open a connection to the catalog, pinned to UTC.

        `TIMESTAMPTZ` stores an instant but renders it in the session
        timezone, so without this the same catalog row reads back as
        `2015-01-02T00:00:00+00:00` in London and `2015-01-02T05:30:00+05:30`
        in Kolkata. The instants agree, but the strings do not — and those
        strings reach the API, the UI, and anyone comparing two runs by eye.
        Everything in this system is UTC (§6.2); the catalog is no exception.

        Returns:
            A duckdb connection. Callers close it; connections are per-operation
            so a long-lived handle cannot hold a write lock across a slow pull.

        Raises:
            DataError: If the file cannot be opened, most often because another
                process holds it.
        """
        try:
            con = duckdb.connect(str(self.path))
            con.execute("SET TimeZone = 'UTC'")
        except duckdb.Error as exc:
            msg = f"Could not open the dataset catalog at {self.path}: {exc}"
            raise DataError(msg) from exc
        return con

    def upsert(self, record: DatasetRecord) -> None:
        """Insert or replace a dataset row.

        Re-ingesting a dataset replaces its row rather than appending: the
        catalog describes the lake's *current* contents, and two rows claiming
        the same `dataset_id` with different hashes would make "which data did
        this run use?" unanswerable.

        Args:
            record: The row to write.

        Raises:
            DataError: If the write fails.
        """
        values = [getattr(record, column) for column in _COLUMNS]
        placeholders = ", ".join("?" for _ in _COLUMNS)
        try:
            with closing(self._connect()) as con:
                con.execute("DELETE FROM datasets WHERE dataset_id = ?", [record.dataset_id])
                con.execute(
                    f"INSERT INTO datasets ({', '.join(_COLUMNS)}) VALUES ({placeholders})",  # noqa: S608
                    values,
                )
        except duckdb.Error as exc:
            msg = f"Could not write catalog row for {record.dataset_id!r}: {exc}"
            raise DataError(msg) from exc

    def get(self, dataset_id: str) -> DatasetRecord | None:
        """Fetch one dataset row.

        Args:
            dataset_id: Identifier such as `'yahoo:SPY:1d'`.

        Returns:
            The record, or None if the catalog has no such dataset.
        """
        with closing(self._connect()) as con:
            row = con.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM datasets WHERE dataset_id = ?",  # noqa: S608
                [dataset_id],
            ).fetchone()
        return None if row is None else _to_record(row)

    def require(self, dataset_id: str) -> DatasetRecord:
        """Fetch one dataset row, failing if it is absent.

        Args:
            dataset_id: Identifier such as `'yahoo:SPY:1d'`.

        Returns:
            The record.

        Raises:
            DataError: If the dataset is not catalogued, listing what is.
        """
        record = self.get(dataset_id)
        if record is None:
            known = [r.dataset_id for r in self.list_datasets()]
            msg = f"Dataset {dataset_id!r} is not in the catalog. Known datasets: {known}."
            raise DataError(msg)
        return record

    def list_datasets(self) -> list[DatasetRecord]:
        """List every catalogued dataset.

        Returns:
            Records ordered by `dataset_id`, so listings are stable across runs
            (§4, I8).
        """
        with closing(self._connect()) as con:
            rows = con.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM datasets ORDER BY dataset_id"  # noqa: S608
            ).fetchall()
        return [_to_record(row) for row in rows]

    def delete(self, dataset_id: str) -> bool:
        """Remove a dataset row.

        Args:
            dataset_id: Identifier to remove.

        Returns:
            True if a row was removed.
        """
        existed = self.get(dataset_id) is not None
        with closing(self._connect()) as con:
            con.execute("DELETE FROM datasets WHERE dataset_id = ?", [dataset_id])
        return existed


def _to_record(row: tuple[Any, ...]) -> DatasetRecord:
    """Convert a duckdb row tuple into a `DatasetRecord`.

    Args:
        row: Values in `_COLUMNS` order.

    Returns:
        The parsed record.
    """
    return DatasetRecord.model_validate(dict(zip(_COLUMNS, row, strict=True)))
