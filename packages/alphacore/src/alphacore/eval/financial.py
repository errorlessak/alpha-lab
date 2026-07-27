"""Risk-adjusted return and trade metrics (§11.2).

Every function here takes `periods_per_year` explicitly. §11.1 is blunt about
why: a Sharpe ratio with an undocumented annualisation factor is unfalsifiable,
and 252 is wrong for crypto, wrong for hourly bars, and wrong for any calendar
that is not the NYSE. The factor is recorded alongside the metric.

Nothing returns `NaN`. A metric that cannot be computed raises, because a `NaN`
propagating into a Sharpe ratio is a critical bug rather than a missing value
(§0.7).
"""

from __future__ import annotations

import contextlib
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from alphacore.errors import EvaluationError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

__all__ = [
    "annualised_return",
    "annualised_volatility",
    "break_even_cost_bps",
    "calmar_ratio",
    "cvar",
    "drawdown_series",
    "information_ratio",
    "max_drawdown",
    "max_drawdown_duration",
    "return_metrics",
    "sharpe_ratio",
    "sortino_ratio",
    "trade_metrics",
    "ulcer_index",
    "var",
]

_MIN_OBSERVATIONS: Final[int] = 2

_ZERO_TOLERANCE: Final[float] = 1e-12
"""Dispersion below this, relative to the data's own scale, counts as zero.

Exact comparison does not work here: `np.full(100, 0.001).std(ddof=1)` is
`2.18e-19`, not `0.0`, because the mean subtraction leaves floating-point dust.
An `== 0` guard silently never fires and the caller gets a Sharpe of 4.6e15
instead of an error."""


def _is_zero(value: float, *, scale: float) -> bool:
    """Whether a dispersion measure is zero to within floating-point dust.

    Args:
        value: The measured dispersion.
        scale: Magnitude of the data it came from.

    Returns:
        True if `value` is negligible relative to `scale`.
    """
    return bool(abs(value) <= _ZERO_TOLERANCE * max(1.0, abs(scale)))


def _check(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate a return series.

    Args:
        returns: Per-period simple returns.

    Returns:
        The array as float64.

    Raises:
        EvaluationError: If it is empty, too short, or contains non-finite
            values.
    """
    values = np.asarray(returns, dtype=float)
    if values.size < _MIN_OBSERVATIONS:
        msg = f"Need at least {_MIN_OBSERVATIONS} returns, got an empty or single-value series."
        raise EvaluationError(msg)
    if not np.isfinite(values).all():
        msg = "Return series contains non-finite values; a NaN Sharpe is a bug, not a result."
        raise EvaluationError(msg)
    return values


def annualised_return(returns: NDArray[np.float64], *, periods_per_year: float) -> float:
    """Geometric annualised return.

    `(prod(1 + r))^(periods_per_year / n) - 1`

    Geometric, not arithmetic: +50% then -50% is a 13.4% annualised loss over
    two periods, not break-even, and the arithmetic mean says otherwise.

    Args:
        returns: Per-period simple returns.
        periods_per_year: Bars per year for this interval and calendar.

    Returns:
        The annualised return.

    Raises:
        EvaluationError: If the series is invalid or the equity goes to zero.
    """
    values = _check(returns)
    growth = float(np.prod(1.0 + values))
    if growth <= 0:
        msg = "Cumulative growth reached zero or below; an annualised return is undefined."
        raise EvaluationError(msg)
    return float(growth ** (periods_per_year / values.size) - 1.0)


def annualised_volatility(returns: NDArray[np.float64], *, periods_per_year: float) -> float:
    """Annualised standard deviation of returns.

    Args:
        returns: Per-period simple returns.
        periods_per_year: Bars per year.

    Returns:
        The annualised volatility.

    Raises:
        EvaluationError: If the series is invalid.
    """
    values = _check(returns)
    return float(values.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: NDArray[np.float64], *, periods_per_year: float, risk_free: float = 0.0
) -> float:
    """Annualised Sharpe ratio.

    `mean(excess) / std(excess) * sqrt(periods_per_year)`

    Args:
        returns: Per-period simple returns.
        periods_per_year: Bars per year.
        risk_free: Per-period risk-free rate.

    Returns:
        The Sharpe ratio.

    Raises:
        EvaluationError: If the series is invalid, or has zero volatility —
            where the ratio is infinite rather than excellent.
    """
    values = _check(returns) - risk_free
    deviation = float(values.std(ddof=1))
    if _is_zero(deviation, scale=float(np.abs(values).max())):
        msg = (
            "Sharpe ratio is undefined for a series with zero volatility. A constant "
            "return is not an infinitely good strategy; it is usually a bug upstream."
        )
        raise EvaluationError(msg)
    return float(values.mean() / deviation * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: NDArray[np.float64], *, periods_per_year: float, mar: float = 0.0
) -> float:
    """Annualised Sortino ratio, penalising only downside deviation.

    `mean(r - mar) / downside_deviation * sqrt(periods_per_year)`, where the
    downside deviation is the root mean square of returns below `mar`.

    Args:
        returns: Per-period simple returns.
        periods_per_year: Bars per year.
        mar: Minimum acceptable return per period.

    Returns:
        The Sortino ratio.

    Raises:
        EvaluationError: If the series is invalid or never falls below `mar`.
    """
    values = _check(returns)
    downside = np.minimum(values - mar, 0.0)
    deviation = float(np.sqrt(np.mean(downside**2)))
    if _is_zero(deviation, scale=float(np.abs(values).max())):
        msg = (
            "Sortino ratio is undefined for a series with no downside. Over a real "
            "sample this means the window is too short, not that risk is absent."
        )
        raise EvaluationError(msg)
    return float((values.mean() - mar) / deviation * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak.

    Args:
        equity: Equity curve.

    Returns:
        A series of values in `[-1, 0]`.

    Raises:
        EvaluationError: If the curve is empty.
    """
    if equity.empty:
        msg = "Cannot compute drawdown on an empty equity curve."
        raise EvaluationError(msg)
    peak = equity.cummax()
    return pd.Series(np.where(peak > 0, equity / peak - 1.0, 0.0), index=equity.index)


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline.

    Args:
        equity: Equity curve.

    Returns:
        The maximum drawdown, negative or zero.

    Raises:
        EvaluationError: If the curve is empty.
    """
    return float(drawdown_series(equity).min())


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest run of bars spent below a previous peak.

    Duration matters as much as depth: a 20% drawdown recovered in a month and
    one that lasts three years are different experiences, and only one of them
    gets a strategy switched off.

    Args:
        equity: Equity curve.

    Returns:
        The longest underwater run, in bars.

    Raises:
        EvaluationError: If the curve is empty.
    """
    underwater = drawdown_series(equity).to_numpy() < 0
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def calmar_ratio(
    returns: NDArray[np.float64], equity: pd.Series, *, periods_per_year: float
) -> float:
    """Annualised return divided by the absolute maximum drawdown.

    Args:
        returns: Per-period simple returns.
        equity: Equity curve.
        periods_per_year: Bars per year.

    Returns:
        The Calmar ratio.

    Raises:
        EvaluationError: If the series is invalid or there is no drawdown to
            divide by.
    """
    worst = abs(max_drawdown(equity))
    if worst == 0:
        msg = "Calmar ratio is undefined when the equity curve never falls."
        raise EvaluationError(msg)
    return float(annualised_return(returns, periods_per_year=periods_per_year) / worst)


def ulcer_index(equity: pd.Series) -> float:
    """Root mean square drawdown.

    Penalises drawdowns that are both deep and *long*, which the maximum
    drawdown ignores entirely — it reports one number about one moment.

    Args:
        equity: Equity curve.

    Returns:
        The ulcer index, non-negative.

    Raises:
        EvaluationError: If the curve is empty.
    """
    return float(np.sqrt(np.mean(drawdown_series(equity).to_numpy() ** 2)))


def var(returns: NDArray[np.float64], *, level: float = 0.95) -> float:
    """Historical value at risk.

    Args:
        returns: Per-period simple returns.
        level: Confidence level, e.g. 0.95.

    Returns:
        The loss quantile, normally negative.

    Raises:
        EvaluationError: If the series is invalid.
    """
    values = _check(returns)
    return float(np.quantile(values, 1.0 - level))


def cvar(returns: NDArray[np.float64], *, level: float = 0.95) -> float:
    """Conditional value at risk: the mean loss beyond the VaR threshold.

    More informative than VaR, which says how bad a bad day is but nothing
    about how bad the bad days beyond it get.

    Args:
        returns: Per-period simple returns.
        level: Confidence level.

    Returns:
        The expected shortfall, normally negative.

    Raises:
        EvaluationError: If the series is invalid.
    """
    values = _check(returns)
    threshold = var(values, level=level)
    tail = values[values <= threshold]
    return float(tail.mean()) if tail.size else float(threshold)


def information_ratio(
    returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    *,
    periods_per_year: float,
) -> float:
    """Annualised active return divided by tracking error.

    Args:
        returns: Strategy returns.
        benchmark_returns: Benchmark returns, aligned.
        periods_per_year: Bars per year.

    Returns:
        The information ratio.

    Raises:
        EvaluationError: If the series disagree in length, or tracking error
            is zero.
    """
    strategy = _check(returns)
    benchmark = _check(benchmark_returns)
    if strategy.size != benchmark.size:
        msg = f"Got {strategy.size} strategy returns for {benchmark.size} benchmark returns."
        raise EvaluationError(msg)
    active = strategy - benchmark
    tracking = float(active.std(ddof=1))
    if _is_zero(tracking, scale=float(np.abs(active).max() or 1.0)):
        msg = "Information ratio is undefined when the strategy tracks the benchmark exactly."
        raise EvaluationError(msg)
    return float(active.mean() / tracking * np.sqrt(periods_per_year))


def return_metrics(
    returns: NDArray[np.float64],
    equity: pd.Series,
    *,
    periods_per_year: float,
    benchmark_returns: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """Compute the full return-based metric set (§11.2).

    Args:
        returns: Per-period simple returns.
        equity: Equity curve.
        periods_per_year: Bars per year.
        benchmark_returns: Optional benchmark for the information ratio.

    Returns:
        A mapping of metric name to value, every one finite. Metrics that are
        genuinely undefined for this sample are omitted rather than filled with
        a sentinel.
    """
    values = _check(returns)
    metrics: dict[str, float] = {
        "annualised_return": annualised_return(values, periods_per_year=periods_per_year),
        "annualised_volatility": annualised_volatility(values, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "max_drawdown_duration": float(max_drawdown_duration(equity)),
        "ulcer_index": ulcer_index(equity),
        "var_95": var(values, level=0.95),
        "cvar_95": cvar(values, level=0.95),
        "var_99": var(values, level=0.99),
        "cvar_99": cvar(values, level=0.99),
        "skew": float(stats.skew(values)),
        "kurtosis": float(stats.kurtosis(values, fisher=False)),
        "best_period": float(values.max()),
        "worst_period": float(values.min()),
        "periods_per_year": float(periods_per_year),
        "n_periods": float(values.size),
    }

    optional: tuple[tuple[str, Callable[[], float]], ...] = (
        ("sharpe", lambda: sharpe_ratio(values, periods_per_year=periods_per_year)),
        ("sortino", lambda: sortino_ratio(values, periods_per_year=periods_per_year)),
        ("calmar", lambda: calmar_ratio(values, equity, periods_per_year=periods_per_year)),
    )
    for name, fn in optional:
        try:
            metrics[name] = fn()
        except EvaluationError:
            # Genuinely undefined on this sample — recorded as absent, never as
            # a zero that would read as a real measurement.
            continue

    if benchmark_returns is not None:
        with contextlib.suppress(EvaluationError):
            metrics["information_ratio"] = information_ratio(
                values, benchmark_returns, periods_per_year=periods_per_year
            )
    return metrics


def trade_metrics(trades: pd.DataFrame) -> dict[str, float]:
    """Compute the trade-based metric set (§11.2).

    Args:
        trades: The blotter from `alphacore.backtest.Ledger.trades`.

    Returns:
        A mapping of metric name to value. An empty blotter gives zeros rather
        than NaNs — no trades is a fact, not a missing measurement.
    """
    if trades.empty:
        return {
            "n_trades": 0.0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "profit_factor": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "payoff_ratio": 0.0,
            "expectancy": 0.0,
            "largest_winner": 0.0,
            "largest_loser": 0.0,
            "average_bars_held": 0.0,
            "max_consecutive_losses": 0.0,
            "total_cost": 0.0,
        }

    pnl = trades["pnl"].to_numpy(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    average_win = float(wins.mean()) if wins.size else 0.0
    average_loss = float(losses.mean()) if losses.size else 0.0

    return {
        "n_trades": float(len(trades)),
        "win_rate": float(wins.size / len(trades)),
        "loss_rate": float(losses.size / len(trades)),
        # No losses is a finite, reportable outcome rather than infinity: a
        # profit factor of inf on a leaderboard is unreadable.
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else gross_profit,
        "average_win": average_win,
        "average_loss": average_loss,
        "payoff_ratio": float(abs(average_win / average_loss)) if average_loss else 0.0,
        "expectancy": float(pnl.mean()),
        "largest_winner": float(pnl.max()),
        "largest_loser": float(pnl.min()),
        "average_bars_held": float(trades["bars_held"].mean()),
        "max_consecutive_losses": float(_max_consecutive(pnl < 0)),
        "total_cost": float(trades["cost"].sum()),
    }


def _max_consecutive(flags: NDArray[np.bool_]) -> int:
    """Longest run of True values.

    Args:
        flags: A boolean array.

    Returns:
        The longest consecutive run.
    """
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def break_even_cost_bps(curve: Sequence[dict[str, Any]]) -> float:
    """The round-trip cost at which net Sharpe reaches zero (§11.2).

    Reported next to the headline Sharpe, always. It answers the only question
    that matters about an execution assumption: how much room is there between
    what you assumed and what would kill the strategy?

    Args:
        curve: Points from the cost sweep, each with `bps` and `sharpe`.

    Returns:
        The interpolated crossing in basis points. Zero when the strategy is
        already unprofitable at no cost; the highest level swept when it never
        crosses within the sweep — an honest lower bound rather than an
        extrapolation.

    Raises:
        EvaluationError: If the curve is empty or malformed.
    """
    if not curve:
        msg = "Cannot compute a break-even cost from an empty sensitivity curve."
        raise EvaluationError(msg)
    try:
        points = sorted(((float(p["bps"]), float(p["sharpe"])) for p in curve), key=lambda p: p[0])
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"Cost sensitivity points need numeric 'bps' and 'sharpe' keys: {exc}"
        raise EvaluationError(msg) from exc

    if points[0][1] <= 0:
        return 0.0
    for (bps_a, sharpe_a), (bps_b, sharpe_b) in pairwise(points):
        if sharpe_b <= 0:
            span = sharpe_a - sharpe_b
            fraction = sharpe_a / span if span else 0.0
            return float(bps_a + fraction * (bps_b - bps_a))
    return float(points[-1][0])
