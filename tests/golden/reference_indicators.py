"""Naive, loop-based indicator implementations used as a test oracle.

These are transcribed directly from the published formulas, in the least clever
way possible: explicit loops, no vectorisation, no shared helpers. They are slow
and that is fine — they run on twenty-bar fixtures.

**Why an oracle rather than committed vendor values.** §7.2 asks for expected
values "from the original published definitions". Vendor exports are not that:
they bake in the vendor's own smoothing and seeding conventions, they are not
redistributable, and disagreeing with one tells you nothing about which of you
is right. A second implementation written from the formula, using a different
algorithm (iterative rather than vectorised), catches the errors that actually
happen — an off-by-one in a window, a wrong smoothing constant, a warm-up
region filled instead of left NaN.

The outputs are additionally frozen into `tests/golden/indicators/*.csv`, so
editing both implementations in the same way still fails the suite.

Conventions fixed here and matched by `alphacore.features.primitives`:

* Wilder smoothing for RSI and ATR: `avg_t = (avg_{t-1} * (n-1) + x_t) / n`
* EMA seeded with the SMA of the first `n` observations, then
  `y_t = a * x_t + (1 - a) * y_{t-1}` with `a = 2 / (n + 1)`
* Warm-up regions are NaN, never zero and never back-filled
"""

from __future__ import annotations

import math

__all__ = [
    "ref_atr",
    "ref_bollinger",
    "ref_ema",
    "ref_macd",
    "ref_rsi",
    "ref_sma",
    "ref_true_range",
    "ref_wma",
]

NAN = float("nan")


def ref_sma(values: list[float], window: int) -> list[float]:
    """Simple moving average: `mean(x[t-n+1 .. t])`."""
    out = [NAN] * len(values)
    for t in range(window - 1, len(values)):
        out[t] = sum(values[t - window + 1 : t + 1]) / window
    return out


def ref_wma(values: list[float], window: int) -> list[float]:
    """Weighted moving average with linearly increasing weights `1..n`."""
    out = [NAN] * len(values)
    denominator = window * (window + 1) / 2
    for t in range(window - 1, len(values)):
        total = 0.0
        for i in range(window):
            total += values[t - window + 1 + i] * (i + 1)
        out[t] = total / denominator
    return out


def ref_ema(values: list[float], window: int) -> list[float]:
    """Exponential moving average, seeded with the SMA of the first `n` values."""
    alpha = 2.0 / (window + 1.0)
    out = [NAN] * len(values)
    if len(values) < window:
        return out
    out[window - 1] = sum(values[:window]) / window
    for t in range(window, len(values)):
        out[t] = alpha * values[t] + (1.0 - alpha) * out[t - 1]
    return out


def ref_rsi(values: list[float], window: int) -> list[float]:
    """Wilder's relative strength index.

    `RSI = 100 - 100 / (1 + avg_gain / avg_loss)`, both averages Wilder-smoothed
    and seeded with the arithmetic mean of the first `n` changes.
    """
    out = [NAN] * len(values)
    if len(values) <= window:
        return out

    gains: list[float] = [NAN]
    losses: list[float] = [NAN]
    for t in range(1, len(values)):
        change = values[t] - values[t - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains[1 : window + 1]) / window
    avg_loss = sum(losses[1 : window + 1]) / window
    out[window] = _rsi_from(avg_gain, avg_loss)

    for t in range(window + 1, len(values)):
        avg_gain = (avg_gain * (window - 1) + gains[t]) / window
        avg_loss = (avg_loss * (window - 1) + losses[t]) / window
        out[t] = _rsi_from(avg_gain, avg_loss)
    return out


def _rsi_from(avg_gain: float, avg_loss: float) -> float:
    """Convert smoothed gain and loss into an RSI level.

    A zero average loss means an unbroken run of gains, which is RSI 100 by
    definition rather than a division by zero.
    """
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def ref_macd(
    values: list[float], fast: int, slow: int, signal: int
) -> tuple[list[float], list[float], list[float]]:
    """MACD line, signal line, and histogram.

    `line = EMA(fast) - EMA(slow)`, `signal = EMA(line, signal)`,
    `histogram = line - signal`. The signal EMA is seeded from the first `signal`
    finite values of the line, so the seed never averages across a NaN.
    """
    fast_ema = ref_ema(values, fast)
    slow_ema = ref_ema(values, slow)
    line = [
        f - s if not (math.isnan(f) or math.isnan(s)) else NAN
        for f, s in zip(fast_ema, slow_ema, strict=True)
    ]

    finite = [i for i, v in enumerate(line) if not math.isnan(v)]
    signal_line = [NAN] * len(values)
    if len(finite) >= signal:
        start = finite[signal - 1]
        alpha = 2.0 / (signal + 1.0)
        signal_line[start] = sum(line[i] for i in finite[:signal]) / signal
        for t in range(start + 1, len(values)):
            signal_line[t] = alpha * line[t] + (1.0 - alpha) * signal_line[t - 1]

    histogram = [
        line_value - signal_value
        if not (math.isnan(line_value) or math.isnan(signal_value))
        else NAN
        for line_value, signal_value in zip(line, signal_line, strict=True)
    ]
    return line, signal_line, histogram


def ref_true_range(high: list[float], low: list[float], close: list[float]) -> list[float]:
    """True range: `max(h-l, |h-c_prev|, |l-c_prev|)`.

    Undefined on the first bar, which has no previous close.
    """
    out = [NAN] * len(high)
    for t in range(1, len(high)):
        out[t] = max(
            high[t] - low[t],
            abs(high[t] - close[t - 1]),
            abs(low[t] - close[t - 1]),
        )
    return out


def ref_atr(high: list[float], low: list[float], close: list[float], window: int) -> list[float]:
    """Wilder's average true range, seeded with the mean of the first `n` ranges."""
    true_range = ref_true_range(high, low, close)
    out = [NAN] * len(high)
    if len(high) <= window:
        return out
    out[window] = sum(true_range[1 : window + 1]) / window
    for t in range(window + 1, len(high)):
        out[t] = (out[t - 1] * (window - 1) + true_range[t]) / window
    return out


def ref_bollinger(
    values: list[float], window: int, k: float
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Bollinger upper, lower, %B, and bandwidth.

    Bands are `SMA(n) +/- k * sigma(n)` with `sigma` the **population** standard
    deviation, which is the original definition. `%B = (x - lower) / (upper -
    lower)`; `bandwidth = (upper - lower) / SMA`.
    """
    middle = ref_sma(values, window)
    upper = [NAN] * len(values)
    lower = [NAN] * len(values)
    pct_b = [NAN] * len(values)
    bandwidth = [NAN] * len(values)

    for t in range(window - 1, len(values)):
        chunk = values[t - window + 1 : t + 1]
        mean = middle[t]
        variance = sum((v - mean) ** 2 for v in chunk) / window
        sigma = math.sqrt(variance)
        upper[t] = mean + k * sigma
        lower[t] = mean - k * sigma
        width = upper[t] - lower[t]
        pct_b[t] = (values[t] - lower[t]) / width if width != 0 else NAN
        bandwidth[t] = width / mean if mean != 0 else NAN
    return upper, lower, pct_b, bandwidth
