"""The canonical bar schema (§6.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacore.errors import SchemaValidationError
from alphacore.schemas.bars import (
    BAR_COLUMNS,
    TS_DTYPE,
    TS_INDEX_NAME,
    empty_bars,
    normalise_bars,
    ohlc_violations,
    validate_bars,
)


def make_raw(n: int = 5, *, symbol: str = "SYNTH") -> pd.DataFrame:
    ts = pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC")
    close = np.linspace(100.0, 100.0 + n - 1, n)
    return pd.DataFrame(
        {
            "ts": ts,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000.0),
            "adj_factor": np.ones(n),
            "symbol": symbol,
        }
    )


# --- normalisation ---------------------------------------------------------


def test_normalise_sets_a_utc_nanosecond_index():
    bars = normalise_bars(make_raw())
    assert bars.index.name == TS_INDEX_NAME
    assert bars.index.dtype == TS_DTYPE


def test_normalise_localises_a_naive_index():
    raw = make_raw()
    raw["ts"] = raw["ts"].dt.tz_localize(None)
    assert normalise_bars(raw).index.tz is not None


def test_normalise_accepts_a_datetime_index():
    raw = make_raw().set_index("ts")
    assert list(normalise_bars(raw).columns) == list(BAR_COLUMNS)


def test_normalise_lowercases_and_reorders_columns():
    raw = make_raw().rename(columns={"open": "Open", "close": "Adj Close"})
    raw["close"] = raw["Adj Close"]
    raw = raw.drop(columns=["Adj Close"])
    assert list(normalise_bars(raw).columns) == list(BAR_COLUMNS)


def test_normalise_defaults_the_adjustment_factor():
    raw = make_raw().drop(columns=["adj_factor"])
    assert (normalise_bars(raw)["adj_factor"] == 1.0).all()


def test_normalise_fills_the_symbol_when_given_one():
    raw = make_raw().drop(columns=["symbol"])
    assert (normalise_bars(raw, symbol="ABC")["symbol"] == "ABC").all()


def test_normalise_without_a_symbol_anywhere_raises():
    raw = make_raw().drop(columns=["symbol"])
    with pytest.raises(SchemaValidationError, match="no 'symbol' column"):
        normalise_bars(raw)


def test_normalise_requires_a_timestamp():
    raw = make_raw().drop(columns=["ts"])
    with pytest.raises(SchemaValidationError, match="DatetimeIndex"):
        normalise_bars(raw, symbol="X")


def test_normalise_names_missing_columns():
    raw = make_raw().drop(columns=["high"])
    with pytest.raises(SchemaValidationError, match=r"missing required column\(s\): \['high'\]"):
        normalise_bars(raw)


def test_normalise_sorts_and_keeps_the_last_duplicate():
    raw = make_raw(3)
    duplicate = raw.iloc[[0]].copy()
    duplicate["close"] = 999.0
    duplicate["high"] = 1000.0
    shuffled = pd.concat([raw.iloc[[2, 1]], raw.iloc[[0]], duplicate])

    bars = normalise_bars(shuffled)
    assert bars.index.is_monotonic_increasing
    # The second copy is the vendor's correction, so it is the one that wins.
    assert bars["close"].iloc[0] == 999.0


def test_empty_bars_has_the_canonical_shape():
    empty = empty_bars()
    assert list(empty.columns) == list(BAR_COLUMNS)
    assert empty.index.dtype == TS_DTYPE


# --- validation ------------------------------------------------------------


def test_valid_bars_pass():
    bars = validate_bars(normalise_bars(make_raw()))
    assert len(bars) == 5


def test_negative_price_is_rejected():
    raw = make_raw()
    raw.loc[2, "low"] = -1.0
    with pytest.raises(SchemaValidationError, match="greater_than"):
        validate_bars(normalise_bars(raw))


def test_negative_volume_is_rejected():
    raw = make_raw()
    raw.loc[1, "volume"] = -5.0
    with pytest.raises(SchemaValidationError):
        validate_bars(normalise_bars(raw))


def test_extra_column_is_rejected():
    frame = normalise_bars(make_raw())
    frame["forward_return"] = 0.01
    with pytest.raises(SchemaValidationError):
        validate_bars(frame)


def test_ohlc_ordering_violation_is_rejected_and_located():
    raw = make_raw()
    raw.loc[3, "high"] = raw.loc[3, "close"] - 5.0
    with pytest.raises(SchemaValidationError, match="ohlc_ordering"):
        validate_bars(normalise_bars(raw))


def test_ohlc_check_can_be_relaxed_for_dirty_data():
    raw = make_raw()
    raw.loc[3, "high"] = raw.loc[3, "close"] - 5.0
    bars = validate_bars(normalise_bars(raw), check_ohlc_ordering=False)
    assert len(bars) == 5


def test_ohlc_violations_reports_the_offending_timestamps():
    raw = make_raw()
    raw.loc[3, "high"] = raw.loc[3, "close"] - 5.0
    bars = normalise_bars(raw)
    violations = ohlc_violations(bars)
    assert len(violations) == 1
    assert violations[0] == bars.index[3]


def test_ohlc_violations_on_empty_input():
    assert len(ohlc_violations(empty_bars())) == 0


def test_error_message_names_the_failing_check_and_a_timestamp():
    raw = make_raw()
    raw.loc[3, "high"] = raw.loc[3, "close"] - 5.0
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_bars(normalise_bars(raw))
    message = str(excinfo.value)
    assert "RawBars schema" in message
    assert "failing row" in message
