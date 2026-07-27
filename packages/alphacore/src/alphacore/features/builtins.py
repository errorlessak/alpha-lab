"""Registration of the built-in features (§7.2).

Thin adapters: each takes a bar frame plus concrete parameters and returns
named output columns, delegating the arithmetic to
`alphacore.features.primitives`. Keeping the maths separate from the
registration means the primitives stay unit-testable as pure array functions,
and the lookback declarations stay readable in one place.

Every `lookback` here is the number of leading bars the feature cannot produce.
Getting one wrong does not corrupt a value — the primitive still returns NaN —
but it does mean the pipeline drops the wrong number of rows, and a single
surviving NaN column will take an estimator down much later and much less
obviously.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from alphacore.features import primitives as p
from alphacore.features.registry import register

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__: list[str] = []

PRICE = frozenset({"close"})
HL = frozenset({"high", "low"})
HLC = frozenset({"high", "low", "close"})
OHLC = frozenset({"open", "high", "low", "close"})
HLCV = frozenset({"high", "low", "close", "volume"})
CV = frozenset({"close", "volume"})
V = frozenset({"volume"})


def _frame(index: pd.Index, **columns: np.ndarray) -> pd.DataFrame:
    """Assemble named output columns against a bar index.

    Args:
        index: The bar index to align to.
        **columns: Output name to values.

    Returns:
        A frame of float64 columns.
    """
    return pd.DataFrame(dict(columns), index=index)


def _close(bars: pd.DataFrame) -> np.ndarray:
    """Extract the close column as a float64 array."""
    return np.asarray(bars["close"], dtype=np.float64)


# --------------------------------------------------------------------------
# Trend
# --------------------------------------------------------------------------


@register(
    name="sma",
    family="trend",
    description="Simple moving average of close over `window` bars.",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]) - 1,
    defaults={"window": 20},
)
def _sma(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.sma(_close(bars), int(params["window"])))


@register(
    name="ema",
    family="trend",
    description="Exponential moving average of close, seeded with the SMA of the first `window`.",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]) - 1,
    defaults={"window": 20},
)
def _ema(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.ema(_close(bars), int(params["window"])))


@register(
    name="wma",
    family="trend",
    description="Linearly weighted moving average of close.",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]) - 1,
    defaults={"window": 20},
)
def _wma(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.wma(_close(bars), int(params["window"])))


@register(
    name="hma",
    family="trend",
    description="Hull moving average: WMA(2*WMA(n/2) - WMA(n), sqrt(n)).",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]) + int(np.sqrt(int(params["window"]))) - 2,
    defaults={"window": 20},
)
def _hma(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.hma(_close(bars), int(params["window"])))


# --------------------------------------------------------------------------
# Momentum
# --------------------------------------------------------------------------


@register(
    name="rsi",
    family="momentum",
    description="Wilder's relative strength index of close, in [0, 100].",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]),
    defaults={"window": 14},
)
def _rsi(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.rsi(_close(bars), int(params["window"])))


@register(
    name="macd",
    family="momentum",
    description="MACD line, signal line, and histogram from close.",
    inputs=PRICE,
    lookback=lambda params: int(params["slow"]) + int(params["signal"]) - 2,
    defaults={"fast": 12, "slow": 26, "signal": 9},
)
def _macd(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    line, signal, hist = p.macd(
        _close(bars), int(params["fast"]), int(params["slow"]), int(params["signal"])
    )
    return _frame(bars.index, line=line, signal=signal, hist=hist)


@register(
    name="roc",
    family="momentum",
    description="Rate of change of close over `window` bars, in percent.",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]),
    defaults={"window": 10},
)
def _roc(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.roc(_close(bars), int(params["window"])))


@register(
    name="momentum",
    family="momentum",
    description="Raw price difference of close over `window` bars.",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]),
    defaults={"window": 10},
)
def _momentum(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.momentum(_close(bars), int(params["window"])))


@register(
    name="ppo",
    family="momentum",
    description="Percentage price oscillator: the MACD line as a percentage of the slow EMA.",
    inputs=PRICE,
    lookback=lambda params: int(params["slow"]) - 1,
    defaults={"fast": 12, "slow": 26},
)
def _ppo(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.ppo(_close(bars), int(params["fast"]), int(params["slow"])))


# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------


@register(
    name="true_range",
    family="volatility",
    description="True range: max(h-l, |h-c_prev|, |l-c_prev|).",
    inputs=HLC,
    lookback=lambda _params: 1,
)
def _true_range(bars: pd.DataFrame, _params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(
        bars.index,
        value=p.true_range(
            bars["high"].to_numpy(np.float64),
            bars["low"].to_numpy(np.float64),
            _close(bars),
        ),
    )


@register(
    name="atr",
    family="volatility",
    description="Wilder's average true range.",
    inputs=HLC,
    lookback=lambda params: int(params["window"]),
    defaults={"window": 14},
)
def _atr(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(
        bars.index,
        value=p.atr(
            bars["high"].to_numpy(np.float64),
            bars["low"].to_numpy(np.float64),
            _close(bars),
            int(params["window"]),
        ),
    )


@register(
    name="realised_vol",
    family="volatility",
    description=(
        "Annualised realised volatility of close-to-close log returns. The "
        "annualisation factor comes from the interval and calendar, never a hardcoded 252."
    ),
    inputs=PRICE,
    lookback=lambda params: int(params["window"]),
    defaults={"window": 20},
    injects=frozenset({"periods_per_year"}),
)
def _realised_vol(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(
        bars.index,
        value=p.realised_vol(
            _close(bars), int(params["window"]), periods_per_year=params["periods_per_year"]
        ),
    )


@register(
    name="bollinger",
    family="volatility",
    description="Bollinger upper band, lower band, %B, and bandwidth.",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]) - 1,
    defaults={"window": 20, "k": 2.0},
)
def _bollinger(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    upper, lower, pct_b, bandwidth = p.bollinger(
        _close(bars), int(params["window"]), float(params["k"])
    )
    return _frame(bars.index, upper=upper, lower=lower, pct_b=pct_b, bandwidth=bandwidth)


@register(
    name="rolling_std",
    family="volatility",
    description="Rolling population standard deviation of close.",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]) - 1,
    defaults={"window": 20},
)
def _rolling_std(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.rolling_std(_close(bars), int(params["window"])))


@register(
    name="parkinson",
    family="volatility",
    description="Parkinson high-low range volatility estimator, annualised.",
    inputs=HL,
    lookback=lambda params: int(params["window"]) - 1,
    defaults={"window": 20},
    injects=frozenset({"periods_per_year"}),
)
def _parkinson(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(
        bars.index,
        value=p.parkinson(
            bars["high"].to_numpy(np.float64),
            bars["low"].to_numpy(np.float64),
            int(params["window"]),
            periods_per_year=params["periods_per_year"],
        ),
    )


@register(
    name="garman_klass",
    family="volatility",
    description="Garman-Klass OHLC volatility estimator, annualised.",
    inputs=OHLC,
    lookback=lambda params: int(params["window"]) - 1,
    defaults={"window": 20},
    injects=frozenset({"periods_per_year"}),
)
def _garman_klass(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(
        bars.index,
        value=p.garman_klass(
            bars["open"].to_numpy(np.float64),
            bars["high"].to_numpy(np.float64),
            bars["low"].to_numpy(np.float64),
            _close(bars),
            int(params["window"]),
            periods_per_year=params["periods_per_year"],
        ),
    )


# --------------------------------------------------------------------------
# Volume
# --------------------------------------------------------------------------


@register(
    name="obv",
    family="volume",
    description="On-balance volume: cumulative volume signed by the close-to-close direction.",
    inputs=CV,
    lookback=lambda _params: 1,
)
def _obv(bars: pd.DataFrame, _params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.obv(_close(bars), bars["volume"].to_numpy(np.float64)))


@register(
    name="vwap_dist",
    family="volume",
    description="Relative distance from close to the rolling volume-weighted average price.",
    inputs=HLCV,
    lookback=lambda params: int(params["window"]) - 1,
    defaults={"window": 20},
)
def _vwap_dist(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    vwap = p.rolling_vwap(
        bars["high"].to_numpy(np.float64),
        bars["low"].to_numpy(np.float64),
        _close(bars),
        bars["volume"].to_numpy(np.float64),
        int(params["window"]),
    )
    return _frame(bars.index, value=p.vwap_distance(_close(bars), vwap))


@register(
    name="cmf",
    family="volume",
    description="Chaikin money flow over `window` bars, in [-1, 1].",
    inputs=HLCV,
    lookback=lambda params: int(params["window"]) - 1,
    defaults={"window": 20},
)
def _cmf(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(
        bars.index,
        value=p.cmf(
            bars["high"].to_numpy(np.float64),
            bars["low"].to_numpy(np.float64),
            _close(bars),
            bars["volume"].to_numpy(np.float64),
            int(params["window"]),
        ),
    )


@register(
    name="volume_ratio",
    family="volume",
    description="Bar volume divided by its own rolling mean.",
    inputs=V,
    lookback=lambda params: int(params["window"]) - 1,
    defaults={"window": 20},
)
def _volume_ratio(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(
        bars.index,
        value=p.volume_ratio(bars["volume"].to_numpy(np.float64), int(params["window"])),
    )


@register(
    name="dollar_volume",
    family="volume",
    description="Notional traded per bar: close times volume.",
    inputs=CV,
    lookback=lambda _params: 0,
)
def _dollar_volume(bars: pd.DataFrame, _params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(
        bars.index,
        value=p.dollar_volume(_close(bars), bars["volume"].to_numpy(np.float64)),
    )


# --------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------


@register(
    name="log_return",
    family="returns",
    description="Backward-looking log return of close over `window` bars.",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]),
    defaults={"window": 1},
)
def _log_return(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.log_return(_close(bars), int(params["window"])))


@register(
    name="simple_return",
    family="returns",
    description="Backward-looking simple return of close over `window` bars.",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]),
    defaults={"window": 1},
)
def _simple_return(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.simple_return(_close(bars), int(params["window"])))


# --------------------------------------------------------------------------
# Statistical
# --------------------------------------------------------------------------


@register(
    name="rolling_stat",
    family="statistical",
    description="Rolling mean, median, std, min, max, skew, or kurtosis of close.",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]) - 1,
    defaults={"window": 20, "stat": "mean"},
)
def _rolling_stat(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(
        bars.index,
        value=p.rolling_stat(_close(bars), int(params["window"]), str(params["stat"])),
    )


@register(
    name="zscore",
    family="statistical",
    description="Rolling z-score of close against its own trailing window.",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]) - 1,
    defaults={"window": 20},
)
def _zscore(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.rolling_zscore(_close(bars), int(params["window"])))


@register(
    name="percentile_rank",
    family="statistical",
    description="Rank of the current close within its trailing window, in [0, 1].",
    inputs=PRICE,
    lookback=lambda params: int(params["window"]) - 1,
    defaults={"window": 20},
)
def _percentile_rank(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(bars.index, value=p.percentile_rank(_close(bars), int(params["window"])))


@register(
    name="fracdiff",
    family="statistical",
    description=(
        "Fractionally differenced close: stationary while retaining memory. The warm-up "
        "is the truncated weight count, which at d=0.4 and thresh=1e-4 is 282 bars."
    ),
    inputs=PRICE,
    lookback=lambda params: (
        p.fracdiff_weights(float(params["d"]), float(params["thresh"])).size - 1
    ),
    defaults={"d": 0.4, "thresh": 1e-4},
)
def _fracdiff(bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    return _frame(
        bars.index,
        value=p.fracdiff(_close(bars), float(params["d"]), float(params["thresh"])),
    )
