"""Data quality assessment (§6.4).

The report answers one question: *is this dataset fit to draw a conclusion
from?* It is written to `runs/{run_id}/quality.json` and rendered in the UI's
data tab.

A `FAIL` aborts the pipeline. `--allow-dirty-data` overrides that, and the
override is not free: the run manifest is stamped `data_quality: FAIL` and
every downstream artifact is marked untrusted. Proceeding on bad data is
sometimes the right call; doing it silently never is.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from alphacore.data.calendars import expected_bar_index
from alphacore.schemas.bars import ohlc_violations

if TYPE_CHECKING:
    from alphacore.config.schema import Interval, QualityThresholds

__all__ = [
    "QUALITY_FILENAME",
    "DataQualityReport",
    "QualityStatus",
    "assess_quality",
]

QUALITY_FILENAME: Final[str] = "quality.json"

_MAX_LISTED_TIMESTAMPS: Final[int] = 20
"""Cap on how many offending timestamps a report enumerates. A dataset with
40,000 missing bars needs a count and a handful of examples, not a 40,000-line
JSON artifact."""

_MIN_ROWS_FOR_Z: Final[int] = 30


class QualityStatus(StrEnum):
    """Outcome of a quality assessment.

    `WARN` means something is off but the thresholds tolerate it. `FAIL` means
    the configured thresholds were breached and the pipeline stops.
    """

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class DataQualityReport(BaseModel):
    """Everything measurable about a dataset's fitness (§6.4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    status: QualityStatus
    reasons: tuple[str, ...] = Field(
        default=(), description="Plain-language findings, worst first. Empty on a clean PASS."
    )

    n_rows: int = Field(ge=0)
    first_bar: dt.datetime | None = None
    last_bar: dt.datetime | None = None

    n_expected_bars: int = Field(ge=0)
    n_missing_bars: int = Field(ge=0)
    missing_bar_pct: float = Field(ge=0.0)
    example_missing_bars: tuple[dt.datetime, ...] = ()
    longest_gap_bars: int = Field(default=0, ge=0)
    longest_gap_start: dt.datetime | None = None

    n_unexpected_bars: int = Field(
        default=0, ge=0, description="Bars present that the session calendar does not predict."
    )
    n_duplicate_timestamps: int = Field(default=0, ge=0)
    duplicate_pct: float = Field(default=0.0, ge=0.0)

    n_zero_volume: int = Field(default=0, ge=0)
    zero_volume_pct: float = Field(default=0.0, ge=0.0)

    n_ohlc_violations: int = Field(default=0, ge=0)
    example_ohlc_violations: tuple[dt.datetime, ...] = ()

    n_extreme_returns: int = Field(default=0, ge=0)
    example_extreme_returns: tuple[dt.datetime, ...] = ()

    rows_per_year: dict[str, int] = Field(default_factory=dict)

    @property
    def is_fatal(self) -> bool:
        """Whether this report should stop the pipeline."""
        return self.status is QualityStatus.FAIL


def _pct(numerator: int, denominator: int) -> float:
    """Compute a percentage, treating an empty denominator as zero.

    Args:
        numerator: Count of interest.
        denominator: Total.

    Returns:
        `100 * numerator / denominator`, or 0.0 when `denominator` is 0.
    """
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def _timestamps(index: pd.Index) -> tuple[dt.datetime, ...]:
    """Take a bounded sample of timestamps for inclusion in the report.

    Args:
        index: Timestamps to sample.

    Returns:
        Up to `_MAX_LISTED_TIMESTAMPS` timestamps as datetimes.
    """
    return tuple(pd.Timestamp(t).to_pydatetime() for t in index[:_MAX_LISTED_TIMESTAMPS])


def _longest_gap(
    missing: pd.DatetimeIndex, expected: pd.DatetimeIndex
) -> tuple[int, pd.Timestamp | None]:
    """Find the longest unbroken run of missing bars.

    A hundred scattered missing bars is a data-vendor annoyance. A hundred
    *consecutive* missing bars is a delisting, a halt, or a broken ingest — and
    it silently deletes a period from every walk-forward fold that spans it.

    Args:
        missing: Expected timestamps with no bar.
        expected: The full expected grid, defining adjacency.

    Returns:
        The run length in bars, and the timestamp it starts at.
    """
    if len(missing) == 0:
        return 0, None
    positions = np.asarray(expected.get_indexer(missing))
    breaks = np.flatnonzero(np.diff(positions) != 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(positions) - 1]))
    lengths = ends - starts + 1
    best = int(np.argmax(lengths))
    return int(lengths[best]), pd.Timestamp(missing[starts[best]])


def _extreme_returns(bars: pd.DataFrame, z_threshold: float) -> pd.DatetimeIndex:
    """Flag bars whose log return is an extreme outlier.

    Uses median and median absolute deviation rather than mean and standard
    deviation: a single 1000x bad tick inflates the standard deviation enough
    to hide itself, which is exactly the failure this check exists to catch.

    Args:
        bars: Validated bars.
        z_threshold: Absolute robust z-score above which a bar is flagged.

    Returns:
        Timestamps of the flagged bars.
    """
    if len(bars) < _MIN_ROWS_FOR_Z:
        return pd.DatetimeIndex([])
    returns = np.log(bars["close"].to_numpy(dtype=float))
    returns = np.diff(returns, prepend=returns[0])
    median = float(np.median(returns))
    mad = float(np.median(np.abs(returns - median)))
    if mad == 0.0:
        return pd.DatetimeIndex([])
    robust_z = 0.6745 * (returns - median) / mad
    return pd.DatetimeIndex(bars.index[np.abs(robust_z) > z_threshold])


def assess_quality(
    bars: pd.DataFrame,
    *,
    dataset_id: str,
    interval: Interval,
    calendar: str,
    thresholds: QualityThresholds,
    start: dt.date | None = None,
    end: dt.date | None = None,
    n_duplicates_dropped: int = 0,
) -> DataQualityReport:
    """Measure a dataset against its configured thresholds.

    Args:
        bars: Validated bars for one symbol.
        dataset_id: Catalog identifier, e.g. `'yahoo:SPY:1d'`.
        interval: Bar interval.
        calendar: Calendar code used to build the expected grid.
        thresholds: The configured PASS/WARN/FAIL boundaries.
        start: Span start for the expected grid. Defaults to the first bar.
        end: Span end for the expected grid. Defaults to the last bar.
        n_duplicates_dropped: Duplicate timestamps removed during
            normalisation. Passed in because by the time bars are validated the
            duplicates are gone, and a count of zero would be misleading.

    Returns:
        The completed report.

    Raises:
        DataError: If the expected grid cannot be built for this calendar.
    """
    n_rows = len(bars)
    if n_rows == 0:
        return DataQualityReport(
            dataset_id=dataset_id,
            status=QualityStatus.FAIL,
            reasons=("dataset is empty",),
            n_rows=0,
            n_expected_bars=0,
            n_missing_bars=0,
            missing_bar_pct=0.0,
        )

    first, last = bars.index[0], bars.index[-1]
    expected = expected_bar_index(start or first.date(), end or last.date(), interval, calendar)
    present = pd.DatetimeIndex(bars.index)
    missing = expected.difference(present)
    unexpected = present.difference(expected)
    gap_bars, gap_start = _longest_gap(missing, expected)

    zero_volume = int((bars["volume"] == 0).sum())
    violations = ohlc_violations(bars)
    extremes = _extreme_returns(bars, thresholds.extreme_return_z)

    missing_pct = _pct(len(missing), len(expected))
    zero_volume_pct = _pct(zero_volume, n_rows)
    duplicate_pct = _pct(n_duplicates_dropped, n_rows + n_duplicates_dropped)

    reasons: list[str] = []
    status = QualityStatus.PASS

    if missing_pct > thresholds.max_missing_bar_pct:
        status = QualityStatus.FAIL
        reasons.append(
            f"{len(missing)} of {len(expected)} expected bars are missing "
            f"({missing_pct:.2f}% > {thresholds.max_missing_bar_pct}%)"
        )
    elif len(missing) > 0:
        status = QualityStatus.WARN
        reasons.append(f"{len(missing)} expected bars are missing ({missing_pct:.2f}%)")

    if zero_volume_pct > thresholds.max_zero_volume_pct:
        status = QualityStatus.FAIL
        reasons.append(
            f"{zero_volume} bars have zero volume "
            f"({zero_volume_pct:.2f}% > {thresholds.max_zero_volume_pct}%)"
        )

    if duplicate_pct > thresholds.max_duplicate_pct:
        status = QualityStatus.FAIL
        reasons.append(
            f"{n_duplicates_dropped} duplicate timestamps were dropped "
            f"({duplicate_pct:.2f}% > {thresholds.max_duplicate_pct}%)"
        )

    if len(violations) > 0:
        if thresholds.fail_on_ohlc_violation:
            status = QualityStatus.FAIL
        elif status is QualityStatus.PASS:
            status = QualityStatus.WARN
        reasons.append(
            f"{len(violations)} bars violate low <= min(open, close) <= max(open, close) <= high"
        )

    if len(extremes) > 0 and status is QualityStatus.PASS:
        status = QualityStatus.WARN
    if len(extremes) > 0:
        reasons.append(
            f"{len(extremes)} bars have a robust return z-score above {thresholds.extreme_return_z}"
        )

    if len(unexpected) > 0 and status is QualityStatus.PASS:
        status = QualityStatus.WARN
    if len(unexpected) > 0:
        reasons.append(
            f"{len(unexpected)} bars fall outside the {calendar} session grid — "
            "the calendar may be wrong for this instrument"
        )

    per_year = bars.index.year.value_counts().sort_index()

    return DataQualityReport(
        dataset_id=dataset_id,
        status=status,
        reasons=tuple(reasons),
        n_rows=n_rows,
        first_bar=pd.Timestamp(first).to_pydatetime(),
        last_bar=pd.Timestamp(last).to_pydatetime(),
        n_expected_bars=len(expected),
        n_missing_bars=len(missing),
        missing_bar_pct=missing_pct,
        example_missing_bars=_timestamps(missing),
        longest_gap_bars=gap_bars,
        longest_gap_start=None if gap_start is None else gap_start.to_pydatetime(),
        n_unexpected_bars=len(unexpected),
        n_duplicate_timestamps=n_duplicates_dropped,
        duplicate_pct=duplicate_pct,
        n_zero_volume=zero_volume,
        zero_volume_pct=zero_volume_pct,
        n_ohlc_violations=len(violations),
        example_ohlc_violations=_timestamps(violations),
        n_extreme_returns=len(extremes),
        example_extreme_returns=_timestamps(extremes),
        rows_per_year={str(year): int(count) for year, count in per_year.items()},
    )
