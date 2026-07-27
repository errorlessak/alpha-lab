"""The engine cannot trade on information it does not yet have (§12.3.3, §12.3.7).

Two tests that only mean anything together.

**The canary.** Give the engine a signal derived from bar `t`'s own close — a
*perfect* predictor of that bar's move — and require it to earn nothing.
Invariant I2 says a signal from bar `t` executes at bar `t + lag`'s open, so
by the time the order fills, the move it predicted has already happened. Any
profit at all means the engine executed at a price it could not have got.

**The inverse canary.** Give the engine the *forward* return as a signal, which
is genuine foresight it is entitled to act on, and require an absurd result. If
this fails, the engine is incapable of profiting from anything, and the canary
above would pass no matter how broken the execution logic was.

§12.3 is explicit that the second exists to prove the first is meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacore.backtest import run_backtest
from alphacore.config.schema import BacktestConfig, CostsConfig, RiskConfig
from alphacore.errors import BacktestError
from alphacore.schemas.bars import normalise_bars, validate_bars

N = 500
SEED = 20260727

NO_RISK = RiskConfig(
    stop_loss_atr=None,
    take_profit_atr=None,
    trailing_stop_atr=None,
    max_drawdown_kill=1.0,
    max_daily_loss=1.0,
    max_concurrent_positions=10,
)

FRICTIONLESS = CostsConfig(
    commission_bps=0.0,
    min_commission=0.0,
    half_spread_bps=0.0,
    impact_model="none",
    impact_coefficient=0.0,
    max_participation=1.0,
    borrow_bps_annual=0.0,
    financing_bps_annual=0.0,
)


@pytest.fixture
def bars() -> pd.DataFrame:
    """A random walk with real bar-to-bar variation to trade against."""
    rng = np.random.default_rng(SEED)
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(N) * 0.015))
    open_ = np.concatenate(([close[0]], close[:-1]))
    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2015-01-01", periods=N, freq="D", tz="UTC"),
            "open": open_,
            "high": body_high * 1.002,
            "low": body_low * 0.998,
            "close": close,
            "volume": np.full(N, 1e12),
            "adj_factor": 1.0,
            "symbol": "SYNTH",
        }
    )
    return validate_bars(normalise_bars(frame))


def config(lag: int = 1) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=100_000.0,
        execution_lag_bars=lag,
        fill_price="next_open",
        costs=FRICTIONLESS,
        cost_sensitivity_bps=(0.0,),
    )


def annualised_sharpe(equity: pd.Series, periods_per_year: float = 252.0) -> float:
    """Sharpe of an equity curve, or 0 when it never moves."""
    returns = equity.pct_change().dropna()
    if returns.empty or returns.std(ddof=0) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year))


def same_bar_signal(bars: pd.DataFrame) -> np.ndarray:
    """The direction of *this* bar's own open-to-close move.

    Perfect knowledge of a move that has already finished by the time the
    engine is allowed to act on it.
    """
    return np.sign(bars["close"].to_numpy() - bars["open"].to_numpy()).astype(int)


def forward_signal(bars: pd.DataFrame) -> np.ndarray:
    """The direction of the *next* bar's move — genuine, actionable foresight."""
    close = bars["close"].to_numpy()
    forward = np.zeros(len(close), dtype=int)
    forward[:-1] = np.sign(close[1:] - close[:-1]).astype(int)
    return forward


def test_no_same_bar_execution(bars):
    """A perfect close-of-bar predictor must earn approximately nothing."""
    result = run_backtest(
        bars, signal=same_bar_signal(bars), config=config(), risk=NO_RISK, strict=True
    )
    sharpe = annualised_sharpe(result.equity["equity"])

    assert abs(sharpe) < 1.0, (
        f"a signal built from bar t's own close produced a Sharpe of {sharpe:.2f}. "
        "The engine executed at a price it could not have had (§4, I2)."
    )


def test_perfect_foresight_sanity(bars):
    """The inverse canary: real foresight must produce an absurd Sharpe.

    Without this, an engine that simply never made money would pass the test
    above, and the whole check would be measuring nothing.
    """
    result = run_backtest(
        bars, signal=forward_signal(bars), config=config(), risk=NO_RISK, strict=True
    )
    sharpe = annualised_sharpe(result.equity["equity"])

    assert sharpe > 10.0, (
        f"a model handed the actual forward return produced a Sharpe of only {sharpe:.2f}. "
        "The engine cannot profit from foresight, so the same-bar test proves nothing."
    )
    assert result.equity["equity"].iloc[-1] > result.equity["equity"].iloc[0] * 2


def test_the_two_signals_are_genuinely_different(bars):
    """Neither canary means anything if the signals coincide."""
    same = same_bar_signal(bars)
    forward = forward_signal(bars)
    agreement = float((same == forward).mean())
    assert agreement < 0.7


def test_the_gap_between_the_canaries_is_large(bars):
    """The measurable statement: foresight pays, hindsight does not."""
    hindsight = annualised_sharpe(
        run_backtest(bars, signal=same_bar_signal(bars), config=config(), risk=NO_RISK).equity[
            "equity"
        ]
    )
    foresight = annualised_sharpe(
        run_backtest(bars, signal=forward_signal(bars), config=config(), risk=NO_RISK).equity[
            "equity"
        ]
    )
    assert foresight > hindsight + 10.0


@pytest.mark.parametrize("lag", [1, 2, 5])
def test_a_longer_lag_does_not_rescue_the_same_bar_signal(bars, lag):
    """Delaying execution further cannot make stale information profitable."""
    result = run_backtest(bars, signal=same_bar_signal(bars), config=config(lag), risk=NO_RISK)
    assert abs(annualised_sharpe(result.equity["equity"])) < 1.0


def test_the_engine_refuses_a_zero_lag(bars):
    """Same-bar execution is unrepresentable, not merely discouraged."""
    with pytest.raises(BacktestError, match="I2"):
        run_backtest(
            bars,
            signal=same_bar_signal(bars),
            config=config().model_copy(update={"execution_lag_bars": 0}),
            risk=NO_RISK,
        )
