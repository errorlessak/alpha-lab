"""Frozen indicator values (§7.2).

`indicators/reference.csv` holds a 60-bar input series and the expected output
of every recursive indicator, computed by the naive oracle in
`reference_indicators.py`. The primitives must match to `1e-9`.

The oracle test in `tests/unit/test_primitives.py` proves the two
implementations agree *today*; this file proves neither of them has moved since
the values were frozen. Editing both in the same wrong way passes the first
check and fails this one, which is the entire reason the file exists.

Regenerate deliberately, and only when you can explain the diff:

    uv run python tests/golden/indicators/make_golden.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphacore.features import primitives as p

GOLDEN = Path(__file__).parent / "indicators" / "reference.csv"
TOL = 1e-9


@pytest.fixture(scope="module")
def golden() -> pd.DataFrame:
    """Load the frozen reference table."""
    assert GOLDEN.is_file(), (
        f"{GOLDEN} is missing. Regenerate with "
        "`uv run python tests/golden/indicators/make_golden.py`."
    )
    return pd.read_csv(GOLDEN)


def assert_matches(actual: np.ndarray, expected: pd.Series, name: str) -> None:
    """Assert a computed series matches its frozen column to `1e-9`."""
    np.testing.assert_allclose(
        actual, expected.to_numpy(dtype=float), rtol=0, atol=TOL, err_msg=name
    )


def test_sma_matches_the_frozen_values(golden):
    assert_matches(p.sma(golden["close"].to_numpy(), 10), golden["sma_10"], "sma_10")


def test_ema_matches_the_frozen_values(golden):
    assert_matches(p.ema(golden["close"].to_numpy(), 12), golden["ema_12"], "ema_12")


def test_wma_matches_the_frozen_values(golden):
    assert_matches(p.wma(golden["close"].to_numpy(), 10), golden["wma_10"], "wma_10")


def test_rsi_matches_the_frozen_values(golden):
    assert_matches(p.rsi(golden["close"].to_numpy(), 14), golden["rsi_14"], "rsi_14")


def test_atr_matches_the_frozen_values(golden):
    actual = p.atr(
        golden["high"].to_numpy(), golden["low"].to_numpy(), golden["close"].to_numpy(), 14
    )
    assert_matches(actual, golden["atr_14"], "atr_14")


def test_macd_matches_the_frozen_values(golden):
    line, signal, hist = p.macd(golden["close"].to_numpy(), 12, 26, 9)
    assert_matches(line, golden["macd_line"], "macd_line")
    assert_matches(signal, golden["macd_signal"], "macd_signal")
    assert_matches(hist, golden["macd_hist"], "macd_hist")


def test_bollinger_matches_the_frozen_values(golden):
    upper, lower, pct_b, bandwidth = p.bollinger(golden["close"].to_numpy(), 20, 2.0)
    assert_matches(upper, golden["bb_upper"], "bb_upper")
    assert_matches(lower, golden["bb_lower"], "bb_lower")
    assert_matches(pct_b, golden["bb_pct_b"], "bb_pct_b")
    assert_matches(bandwidth, golden["bb_bandwidth"], "bb_bandwidth")


def test_the_frozen_input_is_a_valid_ohlc_series(golden):
    # A golden file whose input breaks the bar invariant would freeze values
    # nobody should be reproducing.
    body_low = golden[["open", "close"]].min(axis=1)
    body_high = golden[["open", "close"]].max(axis=1)
    assert (golden["low"] <= body_low).all()
    assert (body_high <= golden["high"]).all()
    assert (golden["volume"] >= 0).all()
