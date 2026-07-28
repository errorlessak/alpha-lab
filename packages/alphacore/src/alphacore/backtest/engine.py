"""The event-driven backtest engine (§12.1).

Own engine rather than a vendored one (ADR 0002), because every interesting
question about a backtest is "what did you assume about fills and costs?" and a
library hides the answer.

**Per bar `t`, in this order:**

1. mark to market at bar `t`'s open, carrying the position from `t-1`
2. apply risk exits against `t`'s high and low, stop winning any tie
3. execute orders queued at `t - execution_lag`, at `t`'s open, plus costs
4. read the signal for `t`, using only what is known at `t`'s close
5. size the position
6. apply limits — daily loss, exposure, concurrency, drawdown kill
7. queue orders for `t + execution_lag`
8. append events to the ledger

**Why the engine physically cannot cheat.** Invariant I2 bans same-bar
execution. Rather than trusting the ordering above, the loop reads the signal
for bar `t` only *after* it has already executed and marked that bar, and
orders derived from it are placed in a queue keyed to a strictly later bar. The
close of bar `t` is not in scope at the moment bar `t`'s order is filled — it
has not been read yet. `test_no_same_bar_execution` proves this holds by giving
the engine a perfect close-of-bar predictor and requiring it to earn nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from alphacore.backtest.costs import CostModel
from alphacore.backtest.ledger import Ledger
from alphacore.backtest.portfolio import Portfolio
from alphacore.errors import BacktestError
from alphacore.features.primitives import atr as atr_series
from alphacore.risk import (
    ExitReason,
    KillSwitch,
    RiskLimiter,
    StopLevels,
    check_stop_exit,
    stop_levels,
    update_trailing_stop,
)
from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alphacore.config.schema import BacktestConfig, RiskConfig

__all__ = ["BacktestResult", "run_backtest"]

_ATR_WINDOW = 14
_TRADING_DAYS = 252.0


@dataclass(frozen=True)
class BacktestResult:
    """Everything one backtest produced.

    Attributes:
        equity: Per-bar equity, position and cash.
        drawdown: Per-bar underwater series.
        trades: Completed round-trip trades.
        total_costs: Currency charged in costs.
        n_fills: Fills executed.
        n_partial_fills: Fills truncated by the participation cap.
        kill_switch: The kill-switch record, if one tripped.
        warnings: Flags for the tearsheet, e.g. `UNCOSTED`.
        fills: The fill-by-fill event log, written to `ledger.parquet` (§16).
            Kept alongside `trades` so the round-trip pairing can be checked
            against the events it was derived from.
    """

    equity: pd.DataFrame
    drawdown: pd.DataFrame
    trades: pd.DataFrame
    total_costs: float
    n_fills: int
    n_partial_fills: int
    kill_switch: KillSwitch | None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    fills: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> dict[str, Any]:
        """Render a JSON-compatible summary for the tearsheet.

        Returns:
            A mapping of headline figures.
        """
        final = float(self.equity["equity"].iloc[-1]) if len(self.equity) else 0.0
        initial = float(self.equity["equity"].iloc[0]) if len(self.equity) else 0.0
        return {
            "n_bars": len(self.equity),
            "initial_equity": initial,
            "final_equity": final,
            "total_return": (final / initial - 1.0) if initial else 0.0,
            "max_drawdown": float(self.drawdown["drawdown"].min()) if len(self.drawdown) else 0.0,
            "total_costs": self.total_costs,
            "n_trades": len(self.trades),
            "n_fills": self.n_fills,
            "n_partial_fills": self.n_partial_fills,
            "kill_switch": self.kill_switch.to_dict() if self.kill_switch else None,
            "warnings": list(self.warnings),
        }


def run_backtest(  # noqa: PLR0912, PLR0915 - the event loop is one readable sequence
    bars: pd.DataFrame,
    *,
    signal: NDArray[np.int_],
    config: BacktestConfig,
    risk: RiskConfig,
    weights: NDArray[np.float64] | None = None,
    periods_per_year: float = _TRADING_DAYS,
    strict: bool = False,
) -> BacktestResult:
    """Run the event-driven backtest.

    Args:
        bars: Validated bars for one symbol.
        signal: Desired position per bar, in `{-1, 0, +1}`.
        config: The `backtest` section.
        risk: The `risk` section.
        weights: Optional per-bar target weights. Defaults to full exposure in
            the signal's direction.
        periods_per_year: Bars per year, for pro-rating carry costs.
        strict: Assert cash and position reconcile to equity every bar.

    Returns:
        The completed result.

    Raises:
        BacktestError: If the signal is misaligned, the execution lag is below
            one bar, or a reconciliation check fails under `strict`.
    """
    n = len(bars)
    signal = np.asarray(signal, dtype=int)
    if signal.size != n:
        msg = f"signal has {signal.size} entries for {n} bars."
        raise BacktestError(msg)
    lag = config.execution_lag_bars
    if lag < 1:
        msg = (
            f"execution_lag_bars must be at least 1, got {lag}. Same-bar execution is "
            "banned at the engine level (§4, I2)."
        )
        raise BacktestError(msg)

    target_weights = (
        np.sign(signal).astype(float) if weights is None else np.asarray(weights, dtype=float)
    )
    if target_weights.size != n:
        msg = f"weights has {target_weights.size} entries for {n} bars."
        raise BacktestError(msg)

    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    volume = bars["volume"].to_numpy(dtype=float)
    index = pd.DatetimeIndex(bars.index)

    atr = atr_series(high, low, close, _ATR_WINDOW)
    bar_vol = np.abs(np.diff(np.log(close), prepend=np.log(close[0])))

    model = CostModel(config.costs)
    portfolio = Portfolio(initial_capital=config.initial_capital)
    ledger = Ledger()
    limiter = RiskLimiter(risk, initial_equity=config.initial_capital)

    # Orders are keyed by the bar they may be executed on, which is what makes
    # same-bar execution unrepresentable rather than merely discouraged.
    pending: dict[int, tuple[float, str]] = {}
    levels: StopLevels | None = None
    entry_price: float | None = None
    unstopped_bars = 0

    for i in range(n):
        price_open = open_[i]

        # 1. mark to market at this bar's open
        equity = portfolio.equity(price=price_open)

        # 2. risk exits against this bar's range, stop wins ties
        if levels is not None and portfolio.quantity != 0.0:
            reason = check_stop_exit(levels, high=high[i], low=low[i])
            if reason is not None:
                exit_price = levels.stop if reason is ExitReason.STOP_LOSS else levels.target
                _fill(
                    portfolio,
                    ledger,
                    model,
                    ts=index[i],
                    quantity=-portfolio.quantity,
                    price=float(exit_price if exit_price is not None else price_open),
                    bar_volume=volume[i],
                    volatility=bar_vol[i],
                    reason=reason.value,
                    bar=i,
                )
                levels = None
                pending.pop(i, None)

        # 3. execute orders queued for this bar, at this bar's open
        order = pending.pop(i, None)
        if order is not None:
            delta, order_reason = order
            if delta != 0.0:
                filled = _fill(
                    portfolio,
                    ledger,
                    model,
                    ts=index[i],
                    quantity=delta,
                    price=price_open,
                    bar_volume=volume[i],
                    volatility=bar_vol[i],
                    reason=order_reason,
                    bar=i,
                )
                if portfolio.quantity == 0.0:
                    levels = None
                    entry_price = None
                elif filled != 0.0:
                    entry_price = price_open
                    levels = (
                        stop_levels(
                            entry_price=price_open,
                            side=int(np.sign(portfolio.quantity)),
                            atr=float(atr[i]),
                            config=risk,
                        )
                        if np.isfinite(atr[i])
                        else None
                    )

        # A position opened before the ATR warm-up completes would otherwise
        # run unstopped for the rest of the sample — silently, because nothing
        # errors. Attach levels as soon as an ATR exists.
        if levels is None and portfolio.quantity != 0.0 and entry_price is not None:
            if np.isfinite(atr[i]):
                levels = stop_levels(
                    entry_price=entry_price,
                    side=int(np.sign(portfolio.quantity)),
                    atr=float(atr[i]),
                    config=risk,
                )
            else:
                unstopped_bars += 1

        # carry costs accrue on whatever is held through the bar
        equity = portfolio.equity(price=close[i])
        carry = model.carry_cost(
            position_value=portfolio.position_value(price=close[i]),
            periods_per_year=periods_per_year,
            equity=equity,
        )
        if carry:
            portfolio.charge(carry)

        if levels is not None and np.isfinite(atr[i]):
            levels = update_trailing_stop(levels, price=close[i], atr=float(atr[i]), config=risk)

        # 4-6. read this bar's signal, size it, apply limits
        equity = portfolio.equity(price=close[i])
        breaches = limiter.check(
            equity=equity, open_positions=1 if portfolio.quantity != 0.0 else 0
        )
        desired_weight = 0.0 if breaches else float(target_weights[i])

        # 7. queue for a strictly later bar
        execute_at = i + lag
        if execute_at < n:
            target = portfolio.target_quantity(weight=desired_weight, price=close[i], equity=equity)
            delta = target - portfolio.quantity
            queued_reason = "kill_switch" if breaches else "signal"
            if abs(delta) > 0:
                pending[execute_at] = (delta, queued_reason)

        # 8. record
        if strict:
            portfolio.assert_reconciles(price=close[i], expected_equity=equity)
        ledger.record_equity(
            ts=index[i], equity=equity, position=portfolio.quantity, cash=portfolio.cash
        )

    warnings: list[str] = []
    if config.costs.is_frictionless:
        warnings.append("UNCOSTED")
    if limiter.killed:
        warnings.append("KILL_SWITCH")
    if unstopped_bars:
        # Held a position with stops configured but no ATR yet. Reported rather
        # than hidden, because those bars ran without the protection the config
        # asked for.
        warnings.append("UNSTOPPED_WARMUP")
        logger.bind(phase="backtest").warning(
            "{} bars held a position before the ATR warm-up completed, so the configured "
            "stops were not yet active",
            unstopped_bars,
        )

    logger.bind(phase="backtest").info(
        "backtest: {} bars, {} fills, {:.2f} in costs, final equity {:.2f}",
        n,
        portfolio.n_fills,
        portfolio.total_costs,
        portfolio.equity(price=close[-1]) if n else 0.0,
    )
    return BacktestResult(
        equity=ledger.equity_curve(),
        drawdown=ledger.drawdown(),
        trades=ledger.trades(),
        total_costs=portfolio.total_costs,
        n_fills=portfolio.n_fills,
        n_partial_fills=ledger.n_partial_fills,
        kill_switch=limiter.kill_switch,
        warnings=tuple(sorted(warnings)),
        fills=ledger.fill_log(),
    )


def _fill(  # noqa: PLR0913 - a fill genuinely has this many distinct inputs
    portfolio: Portfolio,
    ledger: Ledger,
    model: CostModel,
    *,
    ts: pd.Timestamp,
    quantity: float,
    price: float,
    bar_volume: float,
    volatility: float,
    reason: str,
    bar: int,
) -> float:
    """Cap, cost, execute, and record one order.

    Args:
        portfolio: The account.
        ledger: The event log.
        model: The cost model.
        ts: Bar the fill occurs on.
        quantity: Signed requested quantity.
        price: Fill price.
        bar_volume: Volume available in the bar.
        volatility: Per-bar volatility as a fraction of price.
        reason: Why the order existed.
        bar: Positional index of the bar, recorded for holding periods.

    Returns:
        The quantity actually filled, which may be less than requested when the
        participation cap binds.
    """
    filled = model.cap_quantity(quantity=quantity, bar_volume=bar_volume)
    if filled == 0.0:
        return 0.0
    cost = model.trade_cost(
        price=price, quantity=filled, bar_volume=bar_volume, volatility=volatility
    )
    portfolio.execute(quantity=filled, price=price, cost=cost)
    ledger.record_fill(
        ts=ts,
        quantity=filled,
        price=price,
        cost=cost,
        reason=reason,
        requested=quantity,
        bar=bar,
    )
    return filled
