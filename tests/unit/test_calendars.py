"""Session calendars and expected-bar grids (§6.2)."""

from __future__ import annotations

import pandas as pd
import pytest

from alphacore.config.schema import Interval
from alphacore.data.calendars import (
    CRYPTO_CALENDAR,
    DEFAULT_EQUITY_CALENDAR,
    available_calendars,
    default_calendar_for,
    expected_bar_index,
    missing_bars,
)
from alphacore.errors import DataError
from alphacore.schemas.bars import TS_DTYPE, TS_INDEX_NAME


def test_available_calendars_includes_both_defaults():
    names = available_calendars()
    assert DEFAULT_EQUITY_CALENDAR in names
    assert CRYPTO_CALENDAR in names


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("binance", CRYPTO_CALENDAR),
        ("yahoo", DEFAULT_EQUITY_CALENDAR),
        ("csv", DEFAULT_EQUITY_CALENDAR),
        ("something-new", DEFAULT_EQUITY_CALENDAR),
    ],
)
def test_default_calendar_for(provider, expected):
    assert default_calendar_for(provider) == expected


def test_daily_equity_grid_skips_weekends_and_holidays():
    # 2024-01-01 is a holiday; 6th and 7th are a weekend.
    index = expected_bar_index("2024-01-01", "2024-01-08", Interval.D1, "XNYS")
    assert [str(t.date()) for t in index] == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
    ]


def test_grid_dtype_is_utc_nanoseconds():
    index = expected_bar_index("2024-01-02", "2024-01-05", Interval.D1, "XNYS")
    assert index.dtype == TS_DTYPE
    assert index.name == TS_INDEX_NAME


def test_crypto_grid_is_continuous():
    index = expected_bar_index("2024-01-01", "2024-01-02", Interval.H4, CRYPTO_CALENDAR)
    assert len(index) == 12  # two full days at four-hour bars
    assert (index.to_series().diff().dropna() == pd.Timedelta(hours=4)).all()


def test_crypto_grid_includes_weekends():
    index = expected_bar_index("2024-01-06", "2024-01-07", Interval.D1, CRYPTO_CALENDAR)
    assert len(index) == 2


def test_intraday_equity_bars_start_at_the_session_open():
    index = expected_bar_index("2024-01-02", "2024-01-02", Interval.H1, "XNYS")
    # NYSE opens 09:30 New York = 14:30 UTC in January.
    assert index[0] == pd.Timestamp("2024-01-02 14:30:00", tz="UTC")
    assert len(index) == 7  # a 6.5-hour session, final bar partial


def test_unknown_calendar_names_the_fix():
    with pytest.raises(DataError, match="available_calendars"):
        expected_bar_index("2024-01-01", "2024-01-05", Interval.D1, "NOT-A-CALENDAR")


def test_reversed_span_is_an_error():
    with pytest.raises(DataError, match="precedes start"):
        expected_bar_index("2024-02-01", "2024-01-01", Interval.D1, "XNYS")


def test_span_before_the_calendar_starts_is_empty():
    index = expected_bar_index("1600-01-01", "1600-02-01", Interval.D1, "XNYS")
    assert len(index) == 0
    assert index.dtype == TS_DTYPE


def test_missing_bars_finds_the_hole():
    expected = expected_bar_index("2024-01-02", "2024-01-10", Interval.D1, "XNYS")
    actual = expected.delete([2, 3])
    missing = missing_bars(actual, expected)
    assert list(missing) == [expected[2], expected[3]]


def test_missing_bars_with_nothing_present_returns_everything():
    expected = expected_bar_index("2024-01-02", "2024-01-10", Interval.D1, "XNYS")
    assert len(missing_bars([], expected)) == len(expected)


def test_missing_bars_with_everything_present_is_empty():
    expected = expected_bar_index("2024-01-02", "2024-01-10", Interval.D1, "XNYS")
    assert len(missing_bars(expected, expected)) == 0
