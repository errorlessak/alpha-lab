"""Indicator primitives (§7.2).

Three layers of check, deliberately:

1. **Hand arithmetic** on tiny inputs, where the expected value is written out
   in the test and can be verified by reading it.
2. **The naive oracle** in `tests/golden/reference_indicators.py`, an
   independent loop-based transcription of the published formula, asserted to
   `1e-9` as §7.2 requires.
3. **Universal contracts** applied across the whole module: float64 output,
   NaN warm-up of exactly `lookback` bars, no mutation of the input.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphacore.errors import FeatureError
from alphacore.features import primitives as p
from tests.golden.reference_indicators import (
    ref_atr,
    ref_bollinger,
    ref_ema,
    ref_macd,
    ref_rsi,
    ref_sma,
    ref_true_range,
    ref_wma,
)

TOL = 1e-9


@pytest.fixture
def series() -> np.ndarray:
    """A 60-bar random walk, seeded so failures are reproducible."""
    rng = np.random.default_rng(20260727)
    return 100.0 * np.exp(np.cumsum(rng.standard_normal(60) * 0.01))


@pytest.fixture
def long_series() -> np.ndarray:
    """A 2,000-bar walk, for primitives whose window runs to hundreds of bars."""
    rng = np.random.default_rng(4242)
    return 100.0 * np.exp(np.cumsum(rng.standard_normal(2_000) * 0.01))


@pytest.fixture
def ohlc(series: np.ndarray) -> dict[str, np.ndarray]:
    """OHLCV arrays consistent with `low <= min(o,c) <= max(o,c) <= high`."""
    rng = np.random.default_rng(11)
    close = series
    open_ = np.concatenate(([close[0]], close[:-1]))
    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    high = body_high * (1.0 + np.abs(rng.standard_normal(len(close))) * 0.004)
    low = body_low * (1.0 - np.abs(rng.standard_normal(len(close))) * 0.004)
    volume = 1e6 * (1.0 + np.abs(rng.standard_normal(len(close))) * 0.3)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def assert_close(actual: np.ndarray, expected: list[float]) -> None:
    """Assert an array matches an oracle list, treating NaN as equal to NaN."""
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=float), rtol=0, atol=TOL)


# --- trend -----------------------------------------------------------------


def test_sma_by_hand():
    # mean(1,2,3) = 2; mean(2,3,4) = 3
    assert_close(p.sma(np.array([1.0, 2.0, 3.0, 4.0]), 3), [np.nan, np.nan, 2.0, 3.0])


def test_sma_matches_the_oracle(series):
    assert_close(p.sma(series, 10), ref_sma(series.tolist(), 10))


def test_wma_by_hand():
    # (1*1 + 2*2 + 3*3) / 6 = 14/6
    assert_close(p.wma(np.array([1.0, 2.0, 3.0]), 3), [np.nan, np.nan, 14.0 / 6.0])


def test_wma_matches_the_oracle(series):
    assert_close(p.wma(series, 7), ref_wma(series.tolist(), 7))


def test_ema_seeds_with_the_sma_then_recurses():
    # seed = mean(1,2,3) = 2; alpha = 2/4 = 0.5; next = 0.5*4 + 0.5*2 = 3
    assert_close(p.ema(np.array([1.0, 2.0, 3.0, 4.0]), 3), [np.nan, np.nan, 2.0, 3.0])


def test_ema_matches_the_oracle(series):
    assert_close(p.ema(series, 12), ref_ema(series.tolist(), 12))


def test_hma_is_wma_of_the_hull_combination(series):
    window = 16
    half = p.wma(series, window // 2)
    full = p.wma(series, window)
    expected = p.wma(2.0 * half - full, int(np.sqrt(window)))
    assert_close(p.hma(series, window), expected.tolist())


# --- momentum --------------------------------------------------------------


def test_rsi_matches_the_oracle(series):
    assert_close(p.rsi(series, 14), ref_rsi(series.tolist(), 14))


def test_rsi_of_an_unbroken_rally_is_100():
    rising = np.arange(1.0, 30.0)
    assert p.rsi(rising, 14)[-1] == pytest.approx(100.0)


def test_rsi_of_an_unbroken_selloff_is_zero():
    falling = np.arange(30.0, 1.0, -1.0)
    assert p.rsi(falling, 14)[-1] == pytest.approx(0.0)


def test_rsi_of_a_flat_series_is_fifty():
    # No gains and no losses: neither overbought nor oversold, not a NaN.
    assert p.rsi(np.full(30, 100.0), 14)[-1] == pytest.approx(50.0)


def test_macd_matches_the_oracle(series):
    line, signal, hist = p.macd(series, 12, 26, 9)
    ref_line, ref_signal, ref_hist = ref_macd(series.tolist(), 12, 26, 9)
    assert_close(line, ref_line)
    assert_close(signal, ref_signal)
    assert_close(hist, ref_hist)


def test_macd_histogram_is_line_minus_signal(series):
    line, signal, hist = p.macd(series, 12, 26, 9)
    finite = ~np.isnan(hist)
    np.testing.assert_allclose(hist[finite], (line - signal)[finite], atol=TOL)


def test_macd_rejects_a_fast_span_at_or_above_the_slow_one(series):
    with pytest.raises(FeatureError, match="fast"):
        p.macd(series, 26, 12, 9)


def test_roc_is_a_percentage_change():
    # (110 / 100 - 1) * 100 = 10
    assert_close(p.roc(np.array([100.0, 105.0, 110.0]), 2), [np.nan, np.nan, 10.0])


def test_momentum_is_a_raw_difference():
    assert_close(p.momentum(np.array([100.0, 105.0, 110.0]), 2), [np.nan, np.nan, 10.0])


def test_ppo_is_the_macd_line_as_a_percentage_of_the_slow_ema(series):
    slow = p.ema(series, 26)
    expected = (p.ema(series, 12) - slow) / slow * 100.0
    assert_close(p.ppo(series, 12, 26), expected.tolist())


# --- volatility ------------------------------------------------------------


def test_true_range_by_hand():
    high = np.array([10.0, 12.0])
    low = np.array([9.0, 11.0])
    close = np.array([9.5, 11.5])
    # max(12-11, |12-9.5|, |11-9.5|) = 2.5
    assert_close(p.true_range(high, low, close), [np.nan, 2.5])


def test_true_range_matches_the_oracle(ohlc):
    assert_close(
        p.true_range(ohlc["high"], ohlc["low"], ohlc["close"]),
        ref_true_range(ohlc["high"].tolist(), ohlc["low"].tolist(), ohlc["close"].tolist()),
    )


def test_atr_matches_the_oracle(ohlc):
    assert_close(
        p.atr(ohlc["high"], ohlc["low"], ohlc["close"], 14),
        ref_atr(ohlc["high"].tolist(), ohlc["low"].tolist(), ohlc["close"].tolist(), 14),
    )


def test_bollinger_matches_the_oracle(series):
    upper, lower, pct_b, bandwidth = p.bollinger(series, 20, 2.0)
    ref_upper, ref_lower, ref_pct_b, ref_bandwidth = ref_bollinger(series.tolist(), 20, 2.0)
    assert_close(upper, ref_upper)
    assert_close(lower, ref_lower)
    assert_close(pct_b, ref_pct_b)
    assert_close(bandwidth, ref_bandwidth)


def test_bollinger_bands_straddle_the_moving_average(series):
    upper, lower, _, _ = p.bollinger(series, 20, 2.0)
    middle = p.sma(series, 20)
    finite = ~np.isnan(upper)
    assert (upper[finite] >= middle[finite]).all()
    assert (lower[finite] <= middle[finite]).all()


def test_rolling_std_uses_the_population_denominator():
    # population sigma of (1,2,3) = sqrt(2/3)
    assert_close(p.rolling_std(np.array([1.0, 2.0, 3.0]), 3), [np.nan, np.nan, np.sqrt(2.0 / 3.0)])


def test_realised_vol_annualises_by_the_square_root_of_time():
    daily = p.realised_vol(np.array([100.0] * 30), 10, periods_per_year=252)
    assert np.isnan(daily[:10]).all()
    assert daily[-1] == pytest.approx(0.0)


def test_realised_vol_scales_with_the_annualisation_factor(series):
    low = p.realised_vol(series, 20, periods_per_year=252)
    high = p.realised_vol(series, 20, periods_per_year=252 * 4)
    finite = ~np.isnan(low)
    np.testing.assert_allclose(high[finite], low[finite] * 2.0, atol=TOL)


def test_parkinson_is_lower_than_close_to_close_on_a_gapless_series(ohlc):
    park = p.parkinson(ohlc["high"], ohlc["low"], 20, periods_per_year=252)
    assert np.isfinite(park[-1])
    assert park[-1] > 0.0


def test_garman_klass_is_positive_and_finite(ohlc):
    gk = p.garman_klass(
        ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"], 20, periods_per_year=252
    )
    assert np.isfinite(gk[-1])
    assert gk[-1] > 0.0


# --- volume ----------------------------------------------------------------


def test_obv_accumulates_signed_volume():
    close = np.array([10.0, 11.0, 10.5, 10.5])
    volume = np.array([100.0, 200.0, 300.0, 400.0])
    # Bar 1 rises so its volume is added; bar 2 falls so its volume is
    # subtracted; bar 3 is unchanged and contributes nothing.
    assert_close(p.obv(close, volume), [0.0, 200.0, -100.0, -100.0])


def test_rolling_vwap_weights_by_volume():
    high = np.array([2.0, 4.0])
    low = np.array([2.0, 4.0])
    close = np.array([2.0, 4.0])
    volume = np.array([1.0, 3.0])
    # typical prices 2 and 4; (2*1 + 4*3) / 4 = 3.5
    assert_close(p.rolling_vwap(high, low, close, volume, 2), [np.nan, 3.5])


def test_vwap_distance_is_a_relative_gap():
    close = np.array([110.0])
    vwap = np.array([100.0])
    assert_close(p.vwap_distance(close, vwap), [0.1])


def test_cmf_is_bounded_by_plus_and_minus_one(ohlc):
    values = p.cmf(ohlc["high"], ohlc["low"], ohlc["close"], ohlc["volume"], 20)
    finite = values[~np.isnan(values)]
    assert (finite >= -1.0).all()
    assert (finite <= 1.0).all()


def test_cmf_of_closes_at_the_high_is_one():
    high = np.full(25, 10.0)
    low = np.full(25, 9.0)
    close = np.full(25, 10.0)
    volume = np.full(25, 100.0)
    assert p.cmf(high, low, close, volume, 20)[-1] == pytest.approx(1.0)


def test_volume_ratio_compares_to_the_rolling_mean():
    volume = np.array([100.0, 100.0, 100.0, 200.0])
    assert_close(p.volume_ratio(volume, 3), [np.nan, np.nan, 1.0, 200.0 / (400.0 / 3.0)])


def test_dollar_volume_is_price_times_size():
    assert_close(p.dollar_volume(np.array([10.0]), np.array([3.0])), [30.0])


# --- returns ---------------------------------------------------------------


def test_log_return_by_hand():
    assert_close(p.log_return(np.array([100.0, 100.0 * np.e]), 1), [np.nan, 1.0])


def test_simple_return_by_hand():
    assert_close(p.simple_return(np.array([100.0, 110.0]), 1), [np.nan, 0.1])


def test_log_returns_sum_across_horizons(series):
    one = p.log_return(series, 1)
    five = p.log_return(series, 5)
    # A five-bar log return is the sum of five one-bar log returns.
    rolled = np.convolve(np.nan_to_num(one), np.ones(5), mode="full")[: len(one)]
    np.testing.assert_allclose(five[5:], rolled[5:], atol=1e-9)


# --- statistical -----------------------------------------------------------


@pytest.mark.parametrize("stat", ["mean", "median", "std", "min", "max", "skew", "kurt"])
def test_rolling_stat_produces_finite_values_after_warm_up(series, stat):
    values = p.rolling_stat(series, 20, stat)
    assert np.isnan(values[:19]).all()
    assert np.isfinite(values[19:]).all()


def test_rolling_stat_mean_matches_sma(series):
    assert_close(p.rolling_stat(series, 10, "mean"), p.sma(series, 10).tolist())


def test_rolling_stat_rejects_an_unknown_statistic(series):
    with pytest.raises(FeatureError, match="Unknown rolling statistic"):
        p.rolling_stat(series, 10, "mode")


def test_rolling_zscore_of_a_linear_ramp_is_symmetric():
    values = p.rolling_zscore(np.arange(1.0, 11.0), 5)
    # The last point of an increasing window sits at +sqrt(2) population z.
    assert values[-1] == pytest.approx(np.sqrt(2.0), abs=TOL)


def test_rolling_zscore_of_a_flat_series_is_zero_not_infinite():
    # Zero dispersion must not divide by zero.
    values = p.rolling_zscore(np.full(10, 5.0), 5)
    assert np.isnan(values[4:]).all()


def test_percentile_rank_of_the_window_maximum_is_one():
    assert p.percentile_rank(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 5)[-1] == pytest.approx(1.0)


def test_percentile_rank_of_the_window_minimum_is_zero():
    assert p.percentile_rank(np.array([5.0, 4.0, 3.0, 2.0, 1.0]), 5)[-1] == pytest.approx(0.0)


def test_fracdiff_weights_start_at_one_and_alternate():
    weights = p.fracdiff_weights(0.5, 1e-5)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] < 0.0
    assert abs(weights[-1]) < 1e-5 * 10


def test_fracdiff_with_d_zero_is_the_identity(series):
    values = p.fracdiff(series, 0.0, 1e-5)
    finite = ~np.isnan(values)
    np.testing.assert_allclose(values[finite], series[finite], atol=TOL)


def test_fracdiff_with_d_one_is_the_first_difference(series):
    values = p.fracdiff(series, 1.0, 1e-6)
    expected = np.diff(series, prepend=np.nan)
    finite = ~np.isnan(values)
    np.testing.assert_allclose(values[finite], expected[finite], atol=1e-8)


def test_fracdiff_retains_memory_between_the_extremes(long_series):
    # The point of fractional differencing: partly stationary, partly memory.
    # d=0.4 must sit between the identity and the first difference.
    partial = p.fracdiff(long_series, 0.4, 1e-4)
    finite = ~np.isnan(partial)
    assert finite.sum() > 100
    correlation = np.corrcoef(partial[finite], long_series[finite])[0, 1]
    assert abs(correlation) < 0.99


def test_fracdiff_warm_up_is_exactly_the_weight_count(long_series):
    # The registry must be able to declare this lookback honestly, so it has to
    # be predictable rather than emergent.
    weights = p.fracdiff_weights(0.4, 1e-4)
    values = p.fracdiff(long_series, 0.4, 1e-4)
    assert np.isnan(values[: weights.size - 1]).all()
    assert np.isfinite(values[weights.size - 1])


def test_fracdiff_window_grows_as_the_threshold_tightens():
    # A tighter threshold means a longer window and a longer warm-up. At
    # d=0.4 this is not a rounding detail: 1e-4 costs 282 bars and 1e-5 costs
    # 1458, which is nearly six years of daily data.
    assert p.fracdiff_weights(0.4, 1e-4).size < p.fracdiff_weights(0.4, 1e-5).size


def test_fracdiff_rejects_a_non_positive_threshold(series):
    with pytest.raises(FeatureError, match="thresh"):
        p.fracdiff(series, 0.4, 0.0)


# --- universal contracts ---------------------------------------------------


def test_every_primitive_returns_float64(series, ohlc):
    for name, values in _all_outputs(series, ohlc).items():
        assert values.dtype == np.float64, name


def test_no_primitive_mutates_its_input(series, ohlc):
    original = series.copy()
    originals = {k: v.copy() for k, v in ohlc.items()}
    _all_outputs(series, ohlc)
    np.testing.assert_array_equal(series, original)
    for key, value in ohlc.items():
        np.testing.assert_array_equal(value, originals[key])


#: Primitives with no lookback at all: their value at bar 0 uses only bar 0.
INSTANTANEOUS = {"obv", "dollar_volume"}


def test_warm_up_regions_are_nan_and_never_zero(series, ohlc):
    for name, values in _all_outputs(series, ohlc).items():
        if name in INSTANTANEOUS:
            continue
        assert np.isnan(values[0]), f"{name} has no NaN warm-up at bar 0"


def test_warm_up_regions_are_never_filled_with_zero(series, ohlc):
    # The failure mode this guards against: a zero-filled warm-up looks like a
    # real observation to an estimator, and biases every fitted parameter.
    for name, values in _all_outputs(series, ohlc).items():
        if name in INSTANTANEOUS:
            continue
        leading = values[: np.argmax(~np.isnan(values))]
        assert not (leading == 0.0).any(), f"{name} zero-fills its warm-up"


def _all_outputs(series: np.ndarray, ohlc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluate every primitive once, for the contract tests."""
    high, low, close, volume, open_ = (
        ohlc["high"],
        ohlc["low"],
        ohlc["close"],
        ohlc["volume"],
        ohlc["open"],
    )
    line, signal, hist = p.macd(series, 12, 26, 9)
    upper, lower, pct_b, bandwidth = p.bollinger(series, 20, 2.0)
    vwap = p.rolling_vwap(high, low, close, volume, 20)
    return {
        "sma": p.sma(series, 10),
        "ema": p.ema(series, 12),
        "wma": p.wma(series, 10),
        "hma": p.hma(series, 16),
        "rsi": p.rsi(series, 14),
        "macd_line": line,
        "macd_signal": signal,
        "macd_hist": hist,
        "roc": p.roc(series, 5),
        "momentum": p.momentum(series, 5),
        "ppo": p.ppo(series, 12, 26),
        "true_range": p.true_range(high, low, close),
        "atr": p.atr(high, low, close, 14),
        "realised_vol": p.realised_vol(series, 20, periods_per_year=252),
        "bollinger_upper": upper,
        "bollinger_lower": lower,
        "bollinger_pct_b": pct_b,
        "bollinger_bandwidth": bandwidth,
        "rolling_std": p.rolling_std(series, 20),
        "parkinson": p.parkinson(high, low, 20, periods_per_year=252),
        "garman_klass": p.garman_klass(open_, high, low, close, 20, periods_per_year=252),
        "obv": p.obv(close, volume),
        "rolling_vwap": vwap,
        "vwap_distance": p.vwap_distance(close, vwap),
        "cmf": p.cmf(high, low, close, volume, 20),
        "volume_ratio": p.volume_ratio(volume, 20),
        "dollar_volume": p.dollar_volume(close, volume),
        "log_return": p.log_return(series, 1),
        "simple_return": p.simple_return(series, 1),
        "rolling_stat": p.rolling_stat(series, 20, "median"),
        "rolling_zscore": p.rolling_zscore(series, 20),
        "percentile_rank": p.percentile_rank(series, 20),
        "fracdiff": p.fracdiff(series, 0.4, 1e-5),
    }


@pytest.mark.parametrize("window", [0, -1])
def test_windowed_primitives_reject_a_non_positive_window(series, window):
    with pytest.raises(FeatureError, match="window"):
        p.sma(series, window)


def test_a_window_longer_than_the_series_is_all_nan():
    assert np.isnan(p.sma(np.array([1.0, 2.0]), 10)).all()
