"""Data quality assessment (§6.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacore.config.schema import Interval, QualityThresholds
from alphacore.data.calendars import expected_bar_index
from alphacore.data.quality import QualityStatus, assess_quality
from alphacore.schemas.bars import empty_bars, normalise_bars, validate_bars

CALENDAR = "XNYS"
START = "2024-01-02"
END = "2024-06-28"


def make_bars(*, n: int | None = None) -> pd.DataFrame:
    sessions = expected_bar_index(START, END, Interval.D1, CALENDAR)
    if n is not None:
        sessions = sessions[:n]
    count = len(sessions)
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(count) * 0.01))
    frame = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(count, 1_000_000.0),
            "adj_factor": np.ones(count),
            "symbol": "SYNTH",
        },
        index=sessions,
    )
    return validate_bars(normalise_bars(frame))


def assess(bars: pd.DataFrame, thresholds: QualityThresholds | None = None, **kwargs):
    return assess_quality(
        bars,
        dataset_id="csv:SYNTH:1d",
        interval=Interval.D1,
        calendar=CALENDAR,
        thresholds=thresholds or QualityThresholds(),
        start=pd.Timestamp(START).date(),
        end=pd.Timestamp(END).date(),
        **kwargs,
    )


def test_a_complete_dataset_passes():
    report = assess(make_bars())
    assert report.status is QualityStatus.PASS
    assert report.reasons == ()
    assert report.n_missing_bars == 0
    assert not report.is_fatal


def test_report_records_the_span_and_row_count():
    bars = make_bars()
    report = assess(bars)
    assert report.n_rows == len(bars)
    assert report.first_bar == bars.index[0].to_pydatetime()
    assert report.last_bar == bars.index[-1].to_pydatetime()


def test_rows_per_year_is_reported():
    assert assess(make_bars()).rows_per_year["2024"] > 100


def test_an_empty_dataset_fails():
    report = assess_quality(
        empty_bars(),
        dataset_id="csv:SYNTH:1d",
        interval=Interval.D1,
        calendar=CALENDAR,
        thresholds=QualityThresholds(),
    )
    assert report.status is QualityStatus.FAIL
    assert "empty" in report.reasons[0]


def test_a_few_missing_bars_warn():
    bars = make_bars().drop(index=make_bars().index[[5]])
    report = assess(bars)
    assert report.status is QualityStatus.WARN
    assert report.n_missing_bars == 1
    assert len(report.example_missing_bars) == 1


def test_many_missing_bars_fail():
    bars = make_bars()
    report = assess(bars.iloc[:50], QualityThresholds(max_missing_bar_pct=2.0))
    assert report.status is QualityStatus.FAIL
    assert "expected bars are missing" in report.reasons[0]


def test_longest_gap_measures_consecutive_absences():
    bars = make_bars()
    # A ten-bar hole plus one isolated absence elsewhere.
    dropped = bars.drop(index=bars.index[[*range(20, 30), 80]])
    report = assess(dropped, QualityThresholds(max_missing_bar_pct=100.0))
    assert report.longest_gap_bars == 10
    assert report.longest_gap_start == bars.index[20].to_pydatetime()


def test_no_missing_bars_means_no_gap():
    assert assess(make_bars()).longest_gap_bars == 0


def test_zero_volume_above_threshold_fails():
    bars = make_bars().copy()
    bars.iloc[:20, bars.columns.get_loc("volume")] = 0.0
    report = assess(bars, QualityThresholds(max_zero_volume_pct=1.0))
    assert report.status is QualityStatus.FAIL
    assert report.n_zero_volume == 20


def test_ohlc_violations_fail_when_configured_to():
    bars = make_bars().copy()
    bars.iloc[3, bars.columns.get_loc("high")] = 1.0
    report = assess(bars, QualityThresholds(fail_on_ohlc_violation=True))
    assert report.status is QualityStatus.FAIL
    assert report.n_ohlc_violations == 1
    assert len(report.example_ohlc_violations) == 1


def test_ohlc_violations_only_warn_when_tolerated():
    bars = make_bars().copy()
    bars.iloc[3, bars.columns.get_loc("high")] = 1.0
    report = assess(bars, QualityThresholds(fail_on_ohlc_violation=False))
    assert report.status is QualityStatus.WARN


def test_duplicates_above_threshold_fail():
    report = assess(make_bars(), QualityThresholds(max_duplicate_pct=0.0), n_duplicates_dropped=3)
    assert report.status is QualityStatus.FAIL
    assert report.n_duplicate_timestamps == 3


def test_extreme_return_is_flagged():
    bars = make_bars().copy()
    close = bars.columns.get_loc("close")
    bars.iloc[60, close] = float(bars.iloc[60, close]) * 5.0
    # Keep the bar internally consistent so only the return check fires.
    bars.iloc[60, bars.columns.get_loc("high")] = float(bars.iloc[60, close]) * 1.01
    report = assess(bars)
    assert report.n_extreme_returns >= 1
    assert report.status is QualityStatus.WARN


def test_short_series_skips_the_outlier_check():
    # Fewer than 30 rows gives no usable dispersion estimate, so the check is
    # skipped rather than run on noise.
    report = assess(make_bars(n=10), QualityThresholds(max_missing_bar_pct=100.0))
    assert report.n_extreme_returns == 0


def test_bars_outside_the_session_grid_warn():
    bars = make_bars()
    weekend = bars.iloc[[0]].copy()
    weekend.index = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-06", tz="UTC")], name=bars.index.name
    ).astype(bars.index.dtype)
    combined = validate_bars(pd.concat([bars, weekend]).sort_index())
    report = assess(combined)
    assert report.n_unexpected_bars == 1
    assert any("session grid" in reason for reason in report.reasons)


def test_report_is_frozen():
    report = assess(make_bars())
    with pytest.raises(ValueError, match="frozen"):
        report.status = QualityStatus.FAIL


def test_report_serialises_to_json():
    payload = assess(make_bars()).model_dump(mode="json")
    assert payload["status"] == "PASS"
    assert payload["dataset_id"] == "csv:SYNTH:1d"
