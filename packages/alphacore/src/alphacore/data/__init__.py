"""Point-in-time data: providers, session calendars, the lake, and the catalog."""

from __future__ import annotations

from alphacore.data.calendars import (
    CRYPTO_CALENDAR,
    DEFAULT_EQUITY_CALENDAR,
    available_calendars,
    default_calendar_for,
    expected_bar_index,
    missing_bars,
)
from alphacore.data.catalog import CATALOG_FILENAME, Catalog, DatasetRecord
from alphacore.data.lake import (
    DEFAULT_DATA_DIR,
    dataset_dir,
    dataset_id,
    parse_dataset_id,
    read_bars,
    write_bars,
)
from alphacore.data.protocol import BarProvider
from alphacore.data.providers import CsvProvider, YahooProvider, get_provider
from alphacore.data.quality import (
    QUALITY_FILENAME,
    DataQualityReport,
    QualityStatus,
    assess_quality,
)

__all__ = [
    "CATALOG_FILENAME",
    "CRYPTO_CALENDAR",
    "DEFAULT_DATA_DIR",
    "DEFAULT_EQUITY_CALENDAR",
    "QUALITY_FILENAME",
    "BarProvider",
    "Catalog",
    "CsvProvider",
    "DataQualityReport",
    "DatasetRecord",
    "QualityStatus",
    "YahooProvider",
    "assess_quality",
    "available_calendars",
    "dataset_dir",
    "dataset_id",
    "default_calendar_for",
    "expected_bar_index",
    "get_provider",
    "missing_bars",
    "parse_dataset_id",
    "read_bars",
    "write_bars",
]
