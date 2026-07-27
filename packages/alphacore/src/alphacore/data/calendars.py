"""Session calendars and expected-bar grids (§6.2).

Knowing which bars *should* exist is what makes "missing data" a measurable
quantity rather than a vibe. A daily SPY series with 4,900 rows over twenty
years is not obviously wrong until you compare it against the 5,031 sessions
the NYSE actually held.

**Timestamp convention: a bar is stamped with its left edge**, the moment the
bar opens. This is not cosmetic. Invariant I2 says a signal derived from bar `t`
executes at bar `t + lag`'s open; if bars were stamped with their close, "the
open of bar t+1" would be an expression about a timestamp that appears nowhere
in the index, and every off-by-one would be invisible.

Daily bars are stamped at the session date, midnight UTC.
"""

from __future__ import annotations

import datetime as dt
import functools
import math
from typing import TYPE_CHECKING, Final

import exchange_calendars as xcals
import pandas as pd

from alphacore.config.schema import Interval
from alphacore.errors import DataError
from alphacore.schemas.bars import TS_DTYPE, TS_INDEX_NAME

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "CRYPTO_CALENDAR",
    "DEFAULT_EQUITY_CALENDAR",
    "PROVIDER_DEFAULT_CALENDAR",
    "REFERENCE_SPAN",
    "available_calendars",
    "bars_per_session",
    "default_calendar_for",
    "expected_bar_index",
    "minutes_per_session",
    "missing_bars",
    "periods_per_year",
    "sessions_per_year",
]

REFERENCE_SPAN: Final[tuple[str, str]] = ("2015-01-01", "2024-12-31")
"""Decade used to measure a calendar's average session count and length. Fixed
so `periods_per_year` is deterministic: measuring over "the last ten years"
would make an annualisation factor — and therefore every Sharpe ratio — depend
on the day the run happened to execute."""

_REFERENCE_YEARS: Final[float] = 10.0
_MINUTES_PER_DAY: Final[int] = 1440

CRYPTO_CALENDAR: Final[str] = "24/7"
DEFAULT_EQUITY_CALENDAR: Final[str] = "XNYS"

PROVIDER_DEFAULT_CALENDAR: Final[dict[str, str]] = {
    "binance": CRYPTO_CALENDAR,
    "yahoo": DEFAULT_EQUITY_CALENDAR,
    "alphavantage": DEFAULT_EQUITY_CALENDAR,
    "csv": DEFAULT_EQUITY_CALENDAR,
}
"""Calendar assumed when a config does not name one. `csv` defaults to the
equity calendar because that is the common case for a local fixture; a crypto
CSV should say so explicitly via `data.calendar`."""


def available_calendars() -> tuple[str, ...]:
    """List the calendar codes this installation understands.

    Returns:
        Sorted calendar codes, e.g. `('24/7', 'XNYS', ...)`.
    """
    return tuple(sorted(xcals.get_calendar_names()))


def default_calendar_for(provider: str) -> str:
    """Pick a calendar for a provider that did not specify one.

    Args:
        provider: Provider name from `data.provider`.

    Returns:
        A calendar code. Unknown providers fall back to the equity calendar,
        which is the conservative choice: it expects *fewer* bars than a 24/7
        grid, so it cannot manufacture a missing-data warning out of nothing.
    """
    return PROVIDER_DEFAULT_CALENDAR.get(provider, DEFAULT_EQUITY_CALENDAR)


@functools.lru_cache(maxsize=8)
def _get_calendar(code: str) -> xcals.ExchangeCalendar:
    """Load and cache an exchange calendar.

    Args:
        code: An `exchange_calendars` code, e.g. `'XNYS'` or `'24/7'`.

    Returns:
        The calendar.

    Raises:
        DataError: If the code is not recognised.
    """
    try:
        return xcals.get_calendar(code)
    except Exception as exc:  # xcals raises several unrelated types
        msg = (
            f"Unknown market calendar {code!r}. Set data.calendar to one of the codes "
            f"listed by alphacore.data.calendars.available_calendars()."
        )
        raise DataError(msg) from exc


def _as_timestamp(value: dt.date | dt.datetime | str | pd.Timestamp) -> pd.Timestamp:
    """Normalise a date-ish value to a tz-naive UTC-day Timestamp.

    Args:
        value: A date, datetime, ISO string, or Timestamp.

    Returns:
        A tz-naive `pd.Timestamp`, which is what `exchange_calendars` expects
        for session bounds.
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_convert("UTC").tz_localize(None) if stamp.tz is not None else stamp


def expected_bar_index(
    start: dt.date | str | pd.Timestamp,
    end: dt.date | str | pd.Timestamp,
    interval: Interval,
    calendar: str,
) -> pd.DatetimeIndex:
    """Build the grid of bars that should exist over a span.

    Args:
        start: First date, inclusive.
        end: Last date, inclusive.
        interval: Bar interval.
        calendar: Calendar code, e.g. `'XNYS'` or `'24/7'`.

    Returns:
        A UTC nanosecond `DatetimeIndex` named `ts`, holding the left edge of
        every expected bar. Empty if the span contains no sessions.

    Raises:
        DataError: If the calendar is unknown, or `end` precedes `start`, or
            the span falls outside the calendar's supported range.
    """
    cal = _get_calendar(calendar)
    first, last = _as_timestamp(start), _as_timestamp(end)
    if last < first:
        msg = f"expected_bar_index: end ({last.date()}) precedes start ({first.date()})."
        raise DataError(msg)

    # Clip to what the calendar actually covers: asking XNYS about 1885 is a
    # config error, but asking about "today + 1" during a live pull is not.
    first = max(first, _as_timestamp(cal.first_session))
    last = min(last, _as_timestamp(cal.last_session))
    if last < first:
        return _empty_index()

    try:
        if interval is Interval.D1:
            sessions = cal.sessions_in_range(first, last)
            index = pd.DatetimeIndex(sessions).tz_localize("UTC")
        else:
            index = pd.DatetimeIndex(
                cal.trading_index(
                    first,
                    last,
                    period=interval.pandas_freq,
                    intervals=False,
                    closed="left",
                    force=True,
                )
            )
    except Exception as exc:  # xcals raises several unrelated types
        msg = (
            f"Could not build a {interval.value} grid on calendar {calendar!r} "
            f"between {first.date()} and {last.date()}: {exc}"
        )
        raise DataError(msg) from exc

    return _finalise_index(index)


def _empty_index() -> pd.DatetimeIndex:
    """Return an empty, correctly typed bar index.

    Returns:
        An empty UTC nanosecond `DatetimeIndex` named `ts`.
    """
    # Built with the dtype up front: pandas refuses to `astype` a tz-naive
    # index to a tz-aware one, even when it is empty.
    return pd.DatetimeIndex([], dtype=TS_DTYPE, name=TS_INDEX_NAME)


def _finalise_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Coerce a calendar-produced index to the canonical bar dtype.

    pandas 3 defaults to microsecond resolution while §6.2 specifies
    nanoseconds, so the cast is explicit rather than inherited.

    Args:
        index: Any DatetimeIndex, tz-aware or naive.

    Returns:
        A UTC nanosecond `DatetimeIndex` named `ts`.
    """
    out = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    out = pd.DatetimeIndex(out).astype(TS_DTYPE)
    out.name = TS_INDEX_NAME
    return out


@functools.lru_cache(maxsize=8)
def sessions_per_year(calendar: str) -> float:
    """Measure a calendar's average trading sessions per year.

    Measured over `REFERENCE_SPAN` rather than assumed, so a calendar's real
    holiday schedule is reflected. The NYSE comes out near 251.6, not the
    conventional 252.

    Args:
        calendar: Calendar code.

    Returns:
        Sessions per year.

    Raises:
        DataError: If the calendar is unknown.
    """
    cal = _get_calendar(calendar)
    start = max(_as_timestamp(REFERENCE_SPAN[0]), _as_timestamp(cal.first_session))
    end = min(_as_timestamp(REFERENCE_SPAN[1]), _as_timestamp(cal.last_session))
    return len(cal.sessions_in_range(start, end)) / _REFERENCE_YEARS


@functools.lru_cache(maxsize=8)
def minutes_per_session(calendar: str) -> float:
    """Measure a calendar's average session length in minutes.

    Args:
        calendar: Calendar code.

    Returns:
        Minutes per session — 390 for the NYSE, 1440 for a 24/7 market.

    Raises:
        DataError: If the calendar is unknown.
    """
    cal = _get_calendar(calendar)
    start = max(_as_timestamp(REFERENCE_SPAN[0]), _as_timestamp(cal.first_session))
    end = min(_as_timestamp(REFERENCE_SPAN[1]), _as_timestamp(cal.last_session))
    sessions = cal.sessions_in_range(start, end)
    if len(sessions) == 0:
        return float(_MINUTES_PER_DAY)
    opens = pd.DatetimeIndex(cal.opens.loc[sessions])
    closes = pd.DatetimeIndex(cal.closes.loc[sessions])
    return float((closes - opens).mean().total_seconds() / 60.0)


def bars_per_session(interval: Interval, calendar: str) -> int:
    """How many bars of this interval a single session contains.

    Rounded **up**, matching `expected_bar_index`, which uses `force=True` and
    therefore emits a final partial bar. Annualisation has to agree with the
    bar grid the system actually produces; taking 6.5 hourly bars from a
    6.5-hour session when the grid emits 7 would leave every Sharpe ratio
    quietly 4% wrong.

    Args:
        interval: Bar interval.
        calendar: Calendar code.

    Returns:
        Bars per session, at least 1.

    Raises:
        DataError: If the calendar is unknown.
    """
    if not interval.is_intraday:
        return 1
    return max(1, math.ceil(minutes_per_session(calendar) / interval.minutes))


def periods_per_year(interval: Interval, calendar: str) -> float:
    """Bars per year for an interval on a calendar (§11.1).

    `periods_per_year = sessions_per_year * bars_per_session`

    Never hardcoded, and always stored in the tearsheet: a Sharpe ratio with an
    undocumented annualisation factor is unfalsifiable.

    Args:
        interval: Bar interval.
        calendar: Calendar code.

    Returns:
        Bars per year.

    Raises:
        DataError: If the calendar is unknown.
    """
    return sessions_per_year(calendar) * bars_per_session(interval, calendar)


def missing_bars(actual: Iterable[pd.Timestamp], expected: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Find expected bars that the data does not contain.

    Args:
        actual: Timestamps present in the dataset.
        expected: The grid from `expected_bar_index`.

    Returns:
        The expected timestamps with no corresponding bar, in order. Bars
        present but *not* expected are not reported here — that is a separate
        finding, surfaced by the quality report as an unexpected-bar count.
    """
    present = pd.DatetimeIndex(list(actual))
    if present.empty:
        return _finalise_index(expected)
    present = _finalise_index(present)
    return _finalise_index(expected.difference(present))
