"""Indicator primitives (§7.2).

Written in-house rather than vendored (ADR 0001). The reason is causality: a
third-party indicator's warm-up behaviour and centring are implementation
details nobody has certified as leak-free, and "I assume the maintainers got it
right" is not an answer to "how do you know your features cannot see the
future?".

**Contract, honoured by every function here:**

* pure — no mutation of any input array
* `float64` in, `float64` out
* the warm-up region is `NaN`, never `0` and never back-filled
* the value at index `t` depends only on inputs at indices `<= t` (I1)
* the formula appears in the docstring

**Conventions**, fixed once so they cannot drift:

* Wilder smoothing (RSI, ATR): `avg_t = (avg_{t-1} * (n - 1) + x_t) / n`
* EMA is seeded with the SMA of the first `n` observations at index `n - 1`,
  then `y_t = a * x_t + (1 - a) * y_{t-1}` with `a = 2 / (n + 1)`. Seeding
  with `x_0` instead — as `pandas.ewm(adjust=False)` does — makes the first
  hundred values depend on where the data happened to start.
* Rolling dispersion uses the **population** denominator `n`, matching the
  original Bollinger definition.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from alphacore.errors import FeatureError

__all__ = [
    "atr",
    "bollinger",
    "cmf",
    "dollar_volume",
    "ema",
    "fracdiff",
    "fracdiff_weights",
    "garman_klass",
    "hma",
    "log_return",
    "macd",
    "momentum",
    "obv",
    "parkinson",
    "percentile_rank",
    "ppo",
    "realised_vol",
    "roc",
    "rolling_stat",
    "rolling_std",
    "rolling_vwap",
    "rolling_zscore",
    "rsi",
    "simple_return",
    "sma",
    "true_range",
    "volume_ratio",
    "vwap_distance",
    "wma",
]

Array = NDArray[np.float64]

PERCENT: Final[float] = 100.0
_RSI_NEUTRAL: Final[float] = 50.0
_MIN_BARS_FOR_TRUE_RANGE: Final[int] = 2
"""True range needs a previous close, so a single bar cannot produce one."""
_PARKINSON_FACTOR: Final[float] = 1.0 / (4.0 * np.log(2.0))
_ROLLING_STATS: Final[tuple[str, ...]] = (
    "mean",
    "median",
    "std",
    "min",
    "max",
    "skew",
    "kurt",
)


def _as_float(values: NDArray[np.floating] | list[float]) -> Array:
    """Copy an input into a contiguous float64 array.

    The copy is what guarantees purity: callers hand us their price arrays and
    must get them back untouched.

    Args:
        values: Input sequence.

    Returns:
        A new float64 array.
    """
    return np.array(values, dtype=np.float64, copy=True)


def _check_window(window: int, *, name: str = "window") -> int:
    """Validate a lookback window.

    Args:
        window: The proposed window length.
        name: Parameter name, used in the error message.

    Returns:
        The window, unchanged.

    Raises:
        FeatureError: If the window is not a positive integer.
    """
    if not isinstance(window, int | np.integer) or window < 1:
        msg = f"{name} must be a positive integer, got {window!r}."
        raise FeatureError(msg)
    return int(window)


def _rolling(values: Array, window: int) -> pd.core.window.rolling.Rolling:
    """Build a pandas rolling view requiring a full window.

    `min_periods=window` is what produces the NaN warm-up: a partial window
    yields no value rather than a value computed from fewer observations.

    Args:
        values: Input array.
        window: Window length.

    Returns:
        The rolling object.
    """
    return pd.Series(values).rolling(window=window, min_periods=window)


# --------------------------------------------------------------------------
# Trend
# --------------------------------------------------------------------------


def sma(values: Array, window: int) -> Array:
    """Simple moving average.

    `SMA_t = (1/n) * sum(x_{t-n+1} .. x_t)`

    Args:
        values: Input series, normally close.
        window: Number of bars averaged.

    Returns:
        The moving average, NaN for the first `n - 1` bars.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    return np.asarray(_rolling(_as_float(values), window).mean(), dtype=np.float64)


def wma(values: Array, window: int) -> Array:
    """Weighted moving average with linearly increasing weights.

    `WMA_t = sum(i * x_{t-n+i}) / sum(i)` for `i = 1..n`, so the most recent
    observation carries weight `n`.

    Args:
        values: Input series.
        window: Number of bars averaged.

    Returns:
        The weighted average, NaN for the first `n - 1` bars.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    weights = np.arange(1.0, window + 1.0)
    weights /= weights.sum()
    rolled = _rolling(_as_float(values), window).apply(
        lambda chunk: float(np.dot(chunk, weights)), raw=True
    )
    return np.asarray(rolled, dtype=np.float64)


def ema(values: Array, window: int) -> Array:
    """Exponential moving average, seeded with the SMA of the first `n` bars.

    `a = 2 / (n + 1)`; `EMA_{n-1} = SMA_{n-1}`;
    `EMA_t = a * x_t + (1 - a) * EMA_{t-1}` for `t >= n`.

    Args:
        values: Input series.
        window: Span of the average.

    Returns:
        The exponential average, NaN for the first `n - 1` bars.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    data = _as_float(values)
    out = np.full(data.shape, np.nan)
    if data.size < window:
        return out

    alpha = 2.0 / (window + 1.0)
    out[window - 1] = data[:window].mean()
    for t in range(window, data.size):
        out[t] = alpha * data[t] + (1.0 - alpha) * out[t - 1]
    return out


def hma(values: Array, window: int) -> Array:
    """Hull moving average.

    `HMA_t = WMA(2 * WMA(x, n/2) - WMA(x, n), sqrt(n))`

    Responds faster than an SMA of the same length while staying smooth, which
    is the whole point of the construction.

    Args:
        values: Input series.
        window: Nominal window length.

    Returns:
        The Hull average, NaN through the combined warm-up.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    half = max(1, window // 2)
    root = max(1, int(np.sqrt(window)))
    return wma(2.0 * wma(values, half) - wma(values, window), root)


# --------------------------------------------------------------------------
# Momentum
# --------------------------------------------------------------------------


def rsi(values: Array, window: int = 14) -> Array:
    """Wilder's relative strength index.

    `RSI_t = 100 - 100 / (1 + avg_gain_t / avg_loss_t)`, both averages
    Wilder-smoothed and seeded with the mean of the first `n` changes.

    A window with no losses gives 100 and one with no gains gives 0; a window
    with neither gives 50. Returning NaN there would discard a real
    observation, and returning 0 would claim maximum bearishness for a flat
    market.

    Args:
        values: Input series, normally close.
        window: Smoothing period.

    Returns:
        RSI in `[0, 100]`, NaN for the first `n` bars.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    data = _as_float(values)
    out = np.full(data.shape, np.nan)
    if data.size <= window:
        return out

    change = np.diff(data)
    gains = np.clip(change, 0.0, None)
    losses = np.clip(-change, 0.0, None)

    avg_gain = float(gains[:window].mean())
    avg_loss = float(losses[:window].mean())
    out[window] = _rsi_level(avg_gain, avg_loss)

    for t in range(window + 1, data.size):
        avg_gain = (avg_gain * (window - 1) + gains[t - 1]) / window
        avg_loss = (avg_loss * (window - 1) + losses[t - 1]) / window
        out[t] = _rsi_level(avg_gain, avg_loss)
    return out


def _rsi_level(avg_gain: float, avg_loss: float) -> float:
    """Convert smoothed average gain and loss into an RSI level.

    Args:
        avg_gain: Wilder-smoothed average gain.
        avg_loss: Wilder-smoothed average loss.

    Returns:
        RSI in `[0, 100]`.
    """
    if avg_loss == 0.0:
        return PERCENT if avg_gain > 0.0 else _RSI_NEUTRAL
    return PERCENT - PERCENT / (1.0 + avg_gain / avg_loss)


def macd(
    values: Array, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[Array, Array, Array]:
    """Moving average convergence/divergence.

    `line = EMA(x, fast) - EMA(x, slow)`;
    `signal = EMA(line, signal)`; `histogram = line - signal`.

    The signal EMA is seeded from the first `signal` *finite* values of the
    line, so the seed never averages across the line's own warm-up NaNs.

    Args:
        values: Input series.
        fast: Fast EMA span.
        slow: Slow EMA span.
        signal: Signal EMA span.

    Returns:
        `(line, signal, histogram)`.

    Raises:
        FeatureError: If any span is not positive, or `fast >= slow`.
    """
    fast = _check_window(fast, name="fast")
    slow = _check_window(slow, name="slow")
    signal = _check_window(signal, name="signal")
    if fast >= slow:
        msg = f"MACD requires fast < slow, got fast={fast} slow={slow}."
        raise FeatureError(msg)

    line = ema(values, fast) - ema(values, slow)
    return line, _ema_of_finite(line, signal), line - _ema_of_finite(line, signal)


def _ema_of_finite(values: Array, window: int) -> Array:
    """Apply `ema` starting at the first finite observation.

    Args:
        values: A series that may begin with NaN.
        window: Span of the average.

    Returns:
        The EMA, NaN until `window` finite observations have been seen.
    """
    out = np.full(values.shape, np.nan)
    finite = np.flatnonzero(~np.isnan(values))
    if finite.size < window:
        return out
    start = int(finite[0])
    out[start:] = ema(values[start:], window)
    return out


def roc(values: Array, window: int) -> Array:
    """Rate of change, as a percentage.

    `ROC_t = (x_t / x_{t-n} - 1) * 100`

    Args:
        values: Input series.
        window: Lookback in bars.

    Returns:
        Percentage change, NaN for the first `n` bars.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    data = _as_float(values)
    out = np.full(data.shape, np.nan)
    if data.size > window:
        out[window:] = (data[window:] / data[:-window] - 1.0) * PERCENT
    return out


def momentum(values: Array, window: int) -> Array:
    """Raw price momentum.

    `MOM_t = x_t - x_{t-n}`

    Args:
        values: Input series.
        window: Lookback in bars.

    Returns:
        The difference, NaN for the first `n` bars.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    data = _as_float(values)
    out = np.full(data.shape, np.nan)
    if data.size > window:
        out[window:] = data[window:] - data[:-window]
    return out


def ppo(values: Array, fast: int = 12, slow: int = 26) -> Array:
    """Percentage price oscillator — MACD normalised by price level.

    `PPO_t = (EMA(fast) - EMA(slow)) / EMA(slow) * 100`

    Being scale-free, it is comparable across instruments in a way the raw MACD
    line is not.

    Args:
        values: Input series.
        fast: Fast EMA span.
        slow: Slow EMA span.

    Returns:
        The oscillator in percent, NaN through the slow EMA's warm-up.

    Raises:
        FeatureError: If any span is not positive, or `fast >= slow`.
    """
    fast = _check_window(fast, name="fast")
    slow = _check_window(slow, name="slow")
    if fast >= slow:
        msg = f"PPO requires fast < slow, got fast={fast} slow={slow}."
        raise FeatureError(msg)
    slow_ema = ema(values, slow)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (ema(values, fast) - slow_ema) / slow_ema * PERCENT


# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------


def true_range(high: Array, low: Array, close: Array) -> Array:
    """True range.

    `TR_t = max(h_t - l_t, |h_t - c_{t-1}|, |l_t - c_{t-1}|)`

    Undefined on the first bar, which has no previous close — hence NaN rather
    than the naive `h - l`, which would understate the first bar's range
    whenever the session gapped.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.

    Returns:
        True range, NaN at index 0.
    """
    h, low_, c = _as_float(high), _as_float(low), _as_float(close)
    out = np.full(h.shape, np.nan)
    if h.size < _MIN_BARS_FOR_TRUE_RANGE:
        return out
    prev_close = c[:-1]
    out[1:] = np.maximum(
        h[1:] - low_[1:],
        np.maximum(np.abs(h[1:] - prev_close), np.abs(low_[1:] - prev_close)),
    )
    return out


def atr(high: Array, low: Array, close: Array, window: int = 14) -> Array:
    """Wilder's average true range.

    `ATR_n = mean(TR_1 .. TR_n)`;
    `ATR_t = (ATR_{t-1} * (n - 1) + TR_t) / n`

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        window: Smoothing period.

    Returns:
        ATR, NaN for the first `n` bars.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    ranges = true_range(high, low, close)
    out = np.full(ranges.shape, np.nan)
    if ranges.size <= window:
        return out

    out[window] = float(np.mean(ranges[1 : window + 1]))
    for t in range(window + 1, ranges.size):
        out[t] = (out[t - 1] * (window - 1) + ranges[t]) / window
    return out


def rolling_std(values: Array, window: int) -> Array:
    """Rolling population standard deviation.

    `sigma_t = sqrt((1/n) * sum((x_i - mean)^2))` over the trailing window.

    Args:
        values: Input series.
        window: Window length.

    Returns:
        Standard deviation, NaN for the first `n - 1` bars.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    return np.asarray(_rolling(_as_float(values), window).std(ddof=0), dtype=np.float64)


def realised_vol(values: Array, window: int, *, periods_per_year: float) -> Array:
    """Annualised realised volatility of log returns.

    `vol_t = sigma(log-returns over n bars) * sqrt(periods_per_year)`

    The annualisation factor is a required argument, never a default. §11.1 is
    explicit that 252 must not be hardcoded: a Sharpe with an undocumented
    annualisation is unfalsifiable, and so is a volatility.

    Args:
        values: Price series.
        window: Window length in bars.
        periods_per_year: Bars per year for this interval and calendar.

    Returns:
        Annualised volatility, NaN through the warm-up.

    Raises:
        FeatureError: If `window` is not positive or `periods_per_year` is not
            positive.
    """
    window = _check_window(window)
    if periods_per_year <= 0:
        msg = f"periods_per_year must be positive, got {periods_per_year!r}."
        raise FeatureError(msg)
    return rolling_std(log_return(values, 1), window) * float(np.sqrt(periods_per_year))


def bollinger(values: Array, window: int = 20, k: float = 2.0) -> tuple[Array, Array, Array, Array]:
    """Bollinger bands, %B, and bandwidth.

    `middle = SMA(n)`; `upper = middle + k * sigma`; `lower = middle - k * sigma`
    with `sigma` the population standard deviation over the same window.
    `%B = (x - lower) / (upper - lower)`; `bandwidth = (upper - lower) / middle`.

    Args:
        values: Input series.
        window: Window length.
        k: Band width in standard deviations.

    Returns:
        `(upper, lower, pct_b, bandwidth)`.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    data = _as_float(values)
    middle = sma(data, window)
    sigma = rolling_std(data, window)
    upper = middle + k * sigma
    lower = middle - k * sigma
    width = upper - lower
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_b = np.where(width != 0.0, (data - lower) / width, np.nan)
        bandwidth = np.where(middle != 0.0, width / middle, np.nan)
    return upper, lower, pct_b, bandwidth


def parkinson(high: Array, low: Array, window: int, *, periods_per_year: float) -> Array:
    """Parkinson high-low volatility estimator.

    `sigma^2 = (1 / (4 ln 2)) * mean(ln(h/l)^2)`, annualised by
    `sqrt(periods_per_year)`.

    Uses the intrabar range rather than close-to-close, so it extracts roughly
    five times more information per bar — at the cost of assuming no drift and
    no overnight gap.

    Args:
        high: High prices.
        low: Low prices.
        window: Window length in bars.
        periods_per_year: Bars per year.

    Returns:
        Annualised volatility, NaN through the warm-up.

    Raises:
        FeatureError: If `window` or `periods_per_year` is not positive.
    """
    window = _check_window(window)
    if periods_per_year <= 0:
        msg = f"periods_per_year must be positive, got {periods_per_year!r}."
        raise FeatureError(msg)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(_as_float(high) / _as_float(low)) ** 2
    mean_sq = _rolling(log_hl, window).mean().to_numpy(dtype=np.float64)
    return np.asarray(np.sqrt(_PARKINSON_FACTOR * mean_sq * periods_per_year), dtype=np.float64)


def garman_klass(
    open_: Array,
    high: Array,
    low: Array,
    close: Array,
    window: int,
    *,
    periods_per_year: float,
) -> Array:
    """Garman-Klass volatility estimator.

    `sigma^2 = mean(0.5 * ln(h/l)^2 - (2 ln 2 - 1) * ln(c/o)^2)`, annualised by
    `sqrt(periods_per_year)`.

    More efficient than Parkinson because it uses the open and the close as
    well as the range.

    Args:
        open_: Open prices.
        high: High prices.
        low: Low prices.
        close: Close prices.
        window: Window length in bars.
        periods_per_year: Bars per year.

    Returns:
        Annualised volatility, NaN through the warm-up. Also NaN where the
        estimator goes negative, which it can on a bar whose body exceeds its
        range — a data error, surfaced rather than square-rooted.

    Raises:
        FeatureError: If `window` or `periods_per_year` is not positive.
    """
    window = _check_window(window)
    if periods_per_year <= 0:
        msg = f"periods_per_year must be positive, got {periods_per_year!r}."
        raise FeatureError(msg)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(_as_float(high) / _as_float(low)) ** 2
        log_co = np.log(_as_float(close) / _as_float(open_)) ** 2
    per_bar = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    mean_sq = _rolling(per_bar, window).mean().to_numpy(dtype=np.float64)
    return np.sqrt(np.where(mean_sq >= 0.0, mean_sq, np.nan) * periods_per_year)


# --------------------------------------------------------------------------
# Volume
# --------------------------------------------------------------------------


def obv(close: Array, volume: Array) -> Array:
    """On-balance volume.

    `OBV_0 = 0`; `OBV_t = OBV_{t-1} + sign(c_t - c_{t-1}) * v_t`

    Args:
        close: Close prices.
        volume: Bar volumes.

    Returns:
        Cumulative signed volume. Defined from bar 0, so no warm-up.
    """
    c, v = _as_float(close), _as_float(volume)
    signed = np.zeros(c.shape)
    if c.size > 1:
        signed[1:] = np.sign(np.diff(c)) * v[1:]
    return np.cumsum(signed)


def rolling_vwap(high: Array, low: Array, close: Array, volume: Array, window: int) -> Array:
    """Rolling volume-weighted average price.

    `typical_t = (h_t + l_t + c_t) / 3`;
    `VWAP_t = sum(typical * v) / sum(v)` over the trailing window.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        volume: Bar volumes.
        window: Window length.

    Returns:
        VWAP, NaN for the first `n - 1` bars and wherever window volume is zero.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    typical = (_as_float(high) + _as_float(low) + _as_float(close)) / 3.0
    v = _as_float(volume)
    numerator = _rolling(typical * v, window).sum().to_numpy(dtype=np.float64)
    denominator = _rolling(v, window).sum().to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator != 0.0, numerator / denominator, np.nan)


def vwap_distance(close: Array, vwap: Array) -> Array:
    """Relative distance from price to VWAP.

    `dist_t = (c_t - VWAP_t) / VWAP_t`

    Args:
        close: Close prices.
        vwap: A VWAP series, normally from `rolling_vwap`.

    Returns:
        Fractional distance, NaN wherever VWAP is NaN or zero.
    """
    c, w = _as_float(close), _as_float(vwap)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(w != 0.0, (c - w) / w, np.nan)


def cmf(high: Array, low: Array, close: Array, volume: Array, window: int) -> Array:
    """Chaikin money flow.

    `mfm_t = ((c - l) - (h - c)) / (h - l)`;
    `CMF_t = sum(mfm * v) / sum(v)` over the trailing window.

    Bounded in `[-1, 1]`: +1 means every bar in the window closed at its high.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        volume: Bar volumes.
        window: Window length.

    Returns:
        CMF in `[-1, 1]`, NaN through the warm-up.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    h, low_, c, v = (_as_float(high), _as_float(low), _as_float(close), _as_float(volume))
    span = h - low_
    with np.errstate(divide="ignore", invalid="ignore"):
        # A zero-range bar has no directional information, so it contributes
        # nothing rather than dividing by zero.
        multiplier = np.where(span != 0.0, ((c - low_) - (h - c)) / span, 0.0)
    numerator = _rolling(multiplier * v, window).sum().to_numpy(dtype=np.float64)
    denominator = _rolling(v, window).sum().to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator != 0.0, numerator / denominator, np.nan)


def volume_ratio(volume: Array, window: int) -> Array:
    """Volume relative to its own trailing mean.

    `ratio_t = v_t / mean(v_{t-n+1} .. v_t)`

    Args:
        volume: Bar volumes.
        window: Window length.

    Returns:
        The ratio, NaN through the warm-up and wherever the mean is zero.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    v = _as_float(volume)
    mean = sma(v, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mean != 0.0, v / mean, np.nan)


def dollar_volume(close: Array, volume: Array) -> Array:
    """Traded notional per bar.

    `dv_t = c_t * v_t`

    Args:
        close: Close prices.
        volume: Bar volumes.

    Returns:
        Notional traded. Defined from bar 0.
    """
    return _as_float(close) * _as_float(volume)


# --------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------


def log_return(values: Array, window: int = 1) -> Array:
    """Log return over `n` bars.

    `r_t = ln(x_t / x_{t-n})`

    Log returns are used rather than simple ones because they add across time,
    which is what makes multi-horizon aggregation and the annualisation in
    §11.1 exact rather than approximate.

    Args:
        values: Price series.
        window: Horizon in bars.

    Returns:
        Log returns, NaN for the first `n` bars.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    data = _as_float(values)
    out = np.full(data.shape, np.nan)
    if data.size > window:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[window:] = np.log(data[window:] / data[:-window])
    return out


def simple_return(values: Array, window: int = 1) -> Array:
    """Simple return over `n` bars.

    `r_t = x_t / x_{t-n} - 1`

    Args:
        values: Price series.
        window: Horizon in bars.

    Returns:
        Simple returns, NaN for the first `n` bars.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    data = _as_float(values)
    out = np.full(data.shape, np.nan)
    if data.size > window:
        out[window:] = data[window:] / data[:-window] - 1.0
    return out


# --------------------------------------------------------------------------
# Statistical
# --------------------------------------------------------------------------


def rolling_stat(values: Array, window: int, stat: str) -> Array:
    """Rolling summary statistic.

    Args:
        values: Input series.
        window: Window length.
        stat: One of `mean`, `median`, `std`, `min`, `max`, `skew`, `kurt`.
            `std` uses the population denominator; `skew` and `kurt` are the
            sample-adjusted pandas definitions, and `kurt` is excess kurtosis.

    Returns:
        The statistic, NaN through the warm-up.

    Raises:
        FeatureError: If `window` is not positive or `stat` is unknown.
    """
    window = _check_window(window)
    if stat not in _ROLLING_STATS:
        msg = f"Unknown rolling statistic {stat!r}. Known: {list(_ROLLING_STATS)}."
        raise FeatureError(msg)
    roll = _rolling(_as_float(values), window)
    series = roll.std(ddof=0) if stat == "std" else getattr(roll, stat)()
    return np.asarray(series, dtype=np.float64)


def rolling_zscore(values: Array, window: int) -> Array:
    """Rolling z-score against the trailing window.

    `z_t = (x_t - mean_t) / sigma_t`, `sigma` the population deviation.

    Args:
        values: Input series.
        window: Window length.

    Returns:
        The z-score, NaN through the warm-up and NaN — not infinity — wherever
        the window has zero dispersion.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)
    data = _as_float(values)
    mean = sma(data, window)
    sigma = rolling_std(data, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sigma > 0.0, (data - mean) / sigma, np.nan)


def percentile_rank(values: Array, window: int) -> Array:
    """Rank of the current observation within its trailing window.

    `rank_t = #{x_i <= x_t, i in window} / n`, mapped to `[0, 1]`.

    Args:
        values: Input series.
        window: Window length.

    Returns:
        The rank in `[0, 1]`, NaN through the warm-up.

    Raises:
        FeatureError: If `window` is not a positive integer.
    """
    window = _check_window(window)

    def rank(chunk: NDArray[np.float64]) -> float:
        return float((chunk <= chunk[-1]).sum() - 1) / (len(chunk) - 1) if len(chunk) > 1 else 0.0

    ranked = _rolling(_as_float(values), window).apply(rank, raw=True)
    return np.asarray(ranked, dtype=np.float64)


def fracdiff_weights(d: float, thresh: float) -> Array:
    """Binomial weights for fractional differencing.

    `w_0 = 1`; `w_k = -w_{k-1} * (d - k + 1) / k`, truncated once `|w_k| < thresh`.

    Args:
        d: Differencing order, typically in `(0, 1)`.
        thresh: Weight magnitude below which the expansion is truncated.

    Returns:
        Weights, most recent observation first.

    Raises:
        FeatureError: If `thresh` is not positive.
    """
    if thresh <= 0:
        msg = f"fracdiff thresh must be positive, got {thresh!r}."
        raise FeatureError(msg)
    weights = [1.0]
    k = 1
    while True:
        nxt = -weights[-1] * (d - k + 1.0) / k
        if abs(nxt) < thresh:
            break
        weights.append(nxt)
        k += 1
    return np.array(weights, dtype=np.float64)


def fracdiff(values: Array, d: float, thresh: float = 1e-5) -> Array:
    """Fractional differencing with a fixed-width window.

    `x~_t = sum(w_k * x_{t-k})` over the truncated binomial expansion.

    This is the detail that separates a real feature set from a tutorial one.
    Prices are non-stationary and differencing them makes them stationary — but
    a first difference throws away all the memory in the level. Fractional
    differencing at `d ≈ 0.4` buys stationarity while keeping most of the
    autocorrelation structure a model can actually learn from.

    Args:
        values: Price series.
        d: Differencing order. `0` is the identity, `1` the first difference.
        thresh: Weight truncation threshold, which sets the window width.

    Returns:
        The differenced series, NaN for the first `len(weights) - 1` bars.

    Raises:
        FeatureError: If `thresh` is not positive.
    """
    weights = fracdiff_weights(d, thresh)
    data = _as_float(values)
    width = weights.size
    out = np.full(data.shape, np.nan)
    if data.size < width:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(data, width)
    out[width - 1 :] = windows @ weights[::-1]
    return out
