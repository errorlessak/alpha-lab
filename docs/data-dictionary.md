# Data dictionary

Generated from the feature registry. Do not edit by hand:

```bash
just docs
```

`lookback` is bars of warm-up required, as a function of the feature's
parameters. The pipeline drops `max(lookback)` rows rather than filling
them (§7.1), so this column is the cost in sample size of using a feature.

`inputs` lists every bar column the feature reads. No feature may read a
label column; the registry rejects one that tries (§4, I5).

## Trend

| feature | inputs | parameters | description |
|---|---|---|---|
| `ema` | `close` | `window=20` | Exponential moving average of close, seeded with the SMA of the first `window`. |
| `hma` | `close` | `window=20` | Hull moving average: WMA(2*WMA(n/2) - WMA(n), sqrt(n)). |
| `sma` | `close` | `window=20` | Simple moving average of close over `window` bars. |
| `wma` | `close` | `window=20` | Linearly weighted moving average of close. |

## Momentum

| feature | inputs | parameters | description |
|---|---|---|---|
| `macd` | `close` | `fast=12`, `signal=9`, `slow=26` | MACD line, signal line, and histogram from close. |
| `momentum` | `close` | `window=10` | Raw price difference of close over `window` bars. |
| `ppo` | `close` | `fast=12`, `slow=26` | Percentage price oscillator: the MACD line as a percentage of the slow EMA. |
| `roc` | `close` | `window=10` | Rate of change of close over `window` bars, in percent. |
| `rsi` | `close` | `window=14` | Wilder's relative strength index of close, in [0, 100]. |

## Volatility

| feature | inputs | parameters | description |
|---|---|---|---|
| `atr` | `close`, `high`, `low` | `window=14` | Wilder's average true range. |
| `bollinger` | `close` | `k=2.0`, `window=20` | Bollinger upper band, lower band, %B, and bandwidth. |
| `garman_klass` | `close`, `high`, `low`, `open` | `window=20` | Garman-Klass OHLC volatility estimator, annualised. |
| `parkinson` | `high`, `low` | `window=20` | Parkinson high-low range volatility estimator, annualised. |
| `realised_vol` | `close` | `window=20` | Annualised realised volatility of close-to-close log returns. The annualisation factor comes from the interval and calendar, never a hardcoded 252. |
| `rolling_std` | `close` | `window=20` | Rolling population standard deviation of close. |
| `true_range` | `close`, `high`, `low` | — | True range: max(h-l, |h-c_prev|, |l-c_prev|). |

## Volume

| feature | inputs | parameters | description |
|---|---|---|---|
| `cmf` | `close`, `high`, `low`, `volume` | `window=20` | Chaikin money flow over `window` bars, in [-1, 1]. |
| `dollar_volume` | `close`, `volume` | — | Notional traded per bar: close times volume. |
| `obv` | `close`, `volume` | — | On-balance volume: cumulative volume signed by the close-to-close direction. |
| `volume_ratio` | `volume` | `window=20` | Bar volume divided by its own rolling mean. |
| `vwap_dist` | `close`, `high`, `low`, `volume` | `window=20` | Relative distance from close to the rolling volume-weighted average price. |

## Returns

| feature | inputs | parameters | description |
|---|---|---|---|
| `log_return` | `close` | `window=1` | Backward-looking log return of close over `window` bars. |
| `simple_return` | `close` | `window=1` | Backward-looking simple return of close over `window` bars. |

## Statistical

| feature | inputs | parameters | description |
|---|---|---|---|
| `fracdiff` | `close` | `d=0.4`, `thresh=0.0001` | Fractionally differenced close: stationary while retaining memory. The warm-up is the truncated weight count, which at d=0.4 and thresh=1e-4 is 282 bars. |
| `percentile_rank` | `close` | `window=20` | Rank of the current close within its trailing window, in [0, 1]. |
| `rolling_stat` | `close` | `stat='mean'`, `window=20` | Rolling mean, median, std, min, max, skew, or kurtosis of close. |
| `zscore` | `close` | `window=20` | Rolling z-score of close against its own trailing window. |
