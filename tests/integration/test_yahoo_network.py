"""Live Yahoo provider checks (§6.1).

Deselected by default. Network tests that run in CI turn a vendor outage into a
red build, which trains everyone to ignore red builds. Run them deliberately:

    uv run pytest -m network
"""

from __future__ import annotations

import datetime as dt

import pytest

from alphacore.config.schema import Interval, QualityThresholds
from alphacore.data.providers import YahooProvider
from alphacore.data.quality import QualityStatus, assess_quality

pytestmark = pytest.mark.network


def test_daily_spy_satisfies_the_bar_schema():
    bars = YahooProvider().fetch(
        "SPY",
        Interval.D1,
        dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
        dt.datetime(2024, 6, 30, tzinfo=dt.UTC),
    )
    assert len(bars) > 100
    assert bars.index.is_monotonic_increasing
    assert bars.index.is_unique
    assert (bars["close"] > 0).all()


def test_daily_spy_lands_on_the_nyse_grid():
    bars = YahooProvider().fetch(
        "SPY",
        Interval.D1,
        dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
        dt.datetime(2024, 6, 30, tzinfo=dt.UTC),
    )
    report = assess_quality(
        bars,
        dataset_id="yahoo:SPY:1d",
        interval=Interval.D1,
        calendar="XNYS",
        thresholds=QualityThresholds(),
        start=dt.date(2024, 1, 1),
        end=dt.date(2024, 6, 30),
    )
    assert report.status is not QualityStatus.FAIL
    assert report.n_unexpected_bars == 0


def test_unadjusted_close_is_preserved_alongside_the_factor():
    bars = YahooProvider().fetch(
        "SPY",
        Interval.D1,
        dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
        dt.datetime(2024, 6, 30, tzinfo=dt.UTC),
    )
    # I7: the backtester must be able to trade the price that actually printed,
    # so the adjustment lives in its own column rather than being baked in.
    assert (bars["adj_factor"] > 0).all()
    assert (bars["adj_factor"] <= 1.0 + 1e-9).all()
