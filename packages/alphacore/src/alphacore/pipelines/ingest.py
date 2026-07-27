"""The ingestion pipeline (§6, Phase 1).

One symbol at a time: fetch, normalise, validate, assess quality, write to the
lake, record in the catalog. The order is deliberate — nothing reaches the lake
before it has passed the schema, and nothing reaches the catalog before its
quality has been measured and its content hashed.

A `FAIL` verdict aborts by default. `allow_dirty` overrides it, and the
resulting `IngestResult` carries `trusted=False` so every caller downstream can
see that the data was let through on an override rather than on merit.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from alphacore.data.calendars import default_calendar_for
from alphacore.data.catalog import CATALOG_FILENAME, Catalog, DatasetRecord
from alphacore.data.lake import DEFAULT_DATA_DIR, dataset_dir, dataset_id, write_bars
from alphacore.data.providers import get_provider
from alphacore.data.quality import (
    QUALITY_FILENAME,
    DataQualityReport,
    QualityStatus,
    assess_quality,
)
from alphacore.errors import DataQualityError
from alphacore.runtime.hashing import canonical_json
from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    from alphacore.config.schema import AlphaConfig, DataConfig
    from alphacore.data.protocol import BarProvider

__all__ = ["IngestResult", "ingest_dataset", "ingest_symbols", "resolve_calendar"]


@dataclass(frozen=True)
class IngestResult:
    """What one symbol's ingestion produced.

    Attributes:
        dataset_id: Catalog identifier, e.g. `'yahoo:SPY:1d'`.
        n_rows: Bars written.
        content_hash: blake2b over the written parquet files. This is what a
            run manifest pins to.
        paths: Parquet files written, sorted.
        quality: The quality report.
        quality_path: Where the report was written, beside the dataset.
        trusted: False when a `FAIL` verdict was overridden with `allow_dirty`.
            Every artifact derived from an untrusted dataset must say so.
    """

    dataset_id: str
    n_rows: int
    content_hash: str
    paths: tuple[Path, ...]
    quality: DataQualityReport
    quality_path: Path
    trusted: bool


def resolve_calendar(config: DataConfig) -> str:
    """Decide which session calendar applies to a data config.

    Args:
        config: The resolved `data` section.

    Returns:
        The configured calendar, or the provider's default when none is set.
    """
    return config.calendar or default_calendar_for(config.provider)


def _as_utc_datetime(value: dt.date, *, end_of_day: bool) -> dt.datetime:
    """Convert a config date into a UTC datetime bound.

    Args:
        value: The date from `data.start` or `data.end`.
        end_of_day: True to land on 23:59:59 rather than midnight, so an
            inclusive `end` date includes that day's intraday bars.

    Returns:
        A timezone-aware UTC datetime.
    """
    time = dt.time(23, 59, 59) if end_of_day else dt.time(0, 0, 0)
    return dt.datetime.combine(value, time, tzinfo=dt.UTC)


def ingest_dataset(  # noqa: PLR0913 - each parameter is a distinct ingestion input
    symbol: str,
    config: DataConfig,
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    provider: BarProvider | None = None,
    allow_dirty: bool = False,
    now: dt.datetime | None = None,
) -> IngestResult:
    """Fetch, validate, store, and catalogue one symbol.

    Args:
        symbol: Instrument identifier.
        config: The resolved `data` section.
        data_dir: Lake root. The catalog lives at `{data_dir}/catalog.duckdb`.
        provider: Override the provider. Tests inject a fake here; production
            builds one from the config.
        allow_dirty: Proceed even when quality is `FAIL`, marking the result
            untrusted.
        now: Ingest timestamp. Defaults to the current UTC time.

    Returns:
        The ingestion result.

    Raises:
        ProviderError: If the provider cannot serve the request.
        SchemaValidationError: If the bars fail the schema.
        DataQualityError: If quality is `FAIL` and `allow_dirty` is False.
        DataError: If the fetch returns no bars for the span.
    """
    bar_provider = provider if provider is not None else get_provider(config)
    calendar = resolve_calendar(config)
    identifier = dataset_id(config.provider, symbol, config.interval)

    logger.bind(phase="ingest").info(
        "fetching {} {} from {} ({} to {})",
        symbol,
        config.interval.value,
        config.provider,
        config.start,
        config.end,
    )
    raw = bar_provider.fetch(
        symbol,
        config.interval,
        _as_utc_datetime(config.start, end_of_day=False),
        _as_utc_datetime(config.end, end_of_day=True),
    )

    n_duplicates = int(pd.Index(raw.index).duplicated().sum())
    quality = assess_quality(
        raw,
        dataset_id=identifier,
        interval=config.interval,
        calendar=calendar,
        thresholds=config.quality,
        start=config.start,
        end=config.end,
        n_duplicates_dropped=n_duplicates,
    )

    trusted = quality.status is not QualityStatus.FAIL
    if quality.is_fatal:
        detail = "; ".join(quality.reasons) or "no reason recorded"
        if not allow_dirty:
            msg = (
                f"Dataset {identifier} failed its quality thresholds: {detail}. "
                "Fix the data, relax data.quality in the config, or re-run with "
                "--allow-dirty-data to proceed with every downstream artifact marked "
                "untrusted."
            )
            raise DataQualityError(msg)
        logger.bind(phase="ingest").warning(
            "{} quality FAIL overridden by --allow-dirty-data: {}", identifier, detail
        )

    paths, content_hash = write_bars(
        raw,
        root=data_dir,
        provider=config.provider,
        symbol=symbol,
        interval=config.interval,
    )

    # Written after the parquet, and outside the year partitions, so it never
    # perturbs content_hash — the hash describes the bars, not the verdict on
    # them.
    quality_path = (
        dataset_dir(data_dir, config.provider, symbol, config.interval) / QUALITY_FILENAME
    )
    quality_path.write_bytes(canonical_json(quality.model_dump(mode="json")) + b"\n")

    catalog = Catalog(Path(data_dir) / CATALOG_FILENAME)
    catalog.upsert(
        DatasetRecord(
            dataset_id=identifier,
            provider=config.provider,
            symbol=symbol,
            interval=config.interval.value,
            start_ts=pd.Timestamp(raw.index[0]).to_pydatetime(),
            end_ts=pd.Timestamp(raw.index[-1]).to_pydatetime(),
            n_rows=len(raw),
            content_hash=content_hash,
            ingested_at=(now or dt.datetime.now(dt.UTC)),
            provider_version=getattr(bar_provider, "version", None),
            calendar=calendar,
            adjustment=config.adjustment.value,
            quality_status=quality.status.value,
            quality_summary="; ".join(quality.reasons) or None,
        )
    )

    logger.bind(phase="ingest").success(
        "{}: {} bars, quality {}, hash {}",
        identifier,
        len(raw),
        quality.status.value,
        content_hash,
    )
    return IngestResult(
        dataset_id=identifier,
        n_rows=len(raw),
        content_hash=content_hash,
        paths=tuple(paths),
        quality=quality,
        quality_path=quality_path,
        trusted=trusted,
    )


def ingest_symbols(
    config: AlphaConfig,
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    provider: BarProvider | None = None,
    allow_dirty: bool = False,
) -> list[IngestResult]:
    """Ingest every symbol in a config, plus the benchmark.

    Args:
        config: The resolved configuration.
        data_dir: Lake root.
        provider: Override the provider.
        allow_dirty: Proceed past a `FAIL` verdict, marking results untrusted.

    Returns:
        One result per symbol, in the order requested. The benchmark is
        appended if it is not already among the symbols — an information ratio
        computed against a benchmark nobody ingested is not a number anyone
        should see.

    Raises:
        ProviderError: If a provider cannot serve a request.
        DataQualityError: If any dataset fails quality and `allow_dirty` is
            False.
    """
    symbols = list(config.data.symbols)
    if config.data.benchmark and config.data.benchmark not in symbols:
        symbols.append(config.data.benchmark)

    return [
        ingest_dataset(
            symbol,
            config.data,
            data_dir=data_dir,
            provider=provider,
            allow_dirty=allow_dirty,
        )
        for symbol in symbols
    ]
