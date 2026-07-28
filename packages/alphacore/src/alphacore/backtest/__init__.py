"""The event-driven backtester and its cost model (§12)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from alphacore.backtest.costs import BPS, CostModel, impact_cost, round_trip_bps
from alphacore.backtest.engine import BacktestResult, run_backtest
from alphacore.backtest.ledger import EquityPoint, Fill, Ledger
from alphacore.backtest.portfolio import Portfolio
from alphacore.backtest.reference import reference_backtest
from alphacore.eval.financial import net_sharpe

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

    from alphacore.config.schema import BacktestConfig, RiskConfig

__all__ = [
    "BPS",
    "BacktestResult",
    "CostModel",
    "EquityPoint",
    "Fill",
    "Ledger",
    "Portfolio",
    "cost_sensitivity",
    "impact_cost",
    "reference_backtest",
    "round_trip_bps",
    "run_backtest",
]


def cost_sensitivity(
    bars: pd.DataFrame,
    *,
    signal: NDArray[np.int_],
    config: BacktestConfig,
    risk: RiskConfig,
    weights: NDArray[np.float64] | None = None,
    periods_per_year: float = 252.0,
) -> list[dict[str, Any]]:
    """Re-run the backtest at each configured cost level (§12.2).

    The curve this produces is the most informative chart in the tearsheet. A
    strategy whose performance collapses between 0 and 2 basis points is not a
    strategy — it is an artifact of assuming free execution, and the shape of
    this curve says so at a glance.

    Args:
        bars: Validated bars.
        signal: Desired position per bar.
        config: The `backtest` section. Its `cost_sensitivity_bps` levels
            replace the commission, with the spread and impact zeroed so the
            sweep varies exactly one thing.
        risk: The `risk` section.
        weights: Per-bar target weights. **Pass the same weights the headline
            backtest used.** Sweeping an unsized signal while reporting a sized
            Sharpe would produce a break-even cost for a strategy nobody ran.
        periods_per_year: Bars per year, for carry costs.

    Returns:
        One point per level, each with `bps`, `sharpe`, `final_equity`,
        `total_return`, `max_drawdown`, `n_trades` and `total_costs`. The
        Sharpe is what `break_even_cost_bps` interpolates, so it is computed
        here rather than by re-running the sweep somewhere else.

    Raises:
        BacktestError: If the backtest cannot run at any level.
        EvaluationError: If a level produces constant non-zero returns, which
            would imply an infinite Sharpe.
    """
    points: list[dict[str, Any]] = []
    for level in config.cost_sensitivity_bps:
        # Half the round trip on each side, so `bps` reads as a round-trip cost.
        swept = config.costs.model_copy(
            update={
                "commission_bps": float(level) / 2.0,
                "half_spread_bps": 0.0,
                "min_commission": 0.0,
                "impact_model": "none",
            }
        )
        result = run_backtest(
            bars,
            signal=signal,
            config=config.model_copy(update={"costs": swept}),
            risk=risk,
            weights=weights,
            periods_per_year=periods_per_year,
        )
        summary = result.summary()
        returns = result.equity["equity"].pct_change().dropna().to_numpy(dtype=float)
        points.append(
            {
                "bps": float(level),
                "sharpe": net_sharpe(returns, periods_per_year=periods_per_year),
                "final_equity": summary["final_equity"],
                "total_return": summary["total_return"],
                "max_drawdown": summary["max_drawdown"],
                "n_trades": summary["n_trades"],
                "total_costs": summary["total_costs"],
            }
        )
    return points
