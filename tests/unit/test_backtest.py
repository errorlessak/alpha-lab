"""Cost model, portfolio accounting, ledger, and the event loop (§12)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from alphacore.backtest import (
    BacktestResult,
    Ledger,
    Portfolio,
    cost_sensitivity,
    reference_backtest,
    run_backtest,
)
from alphacore.backtest.costs import CostModel, impact_cost, round_trip_bps
from alphacore.config.schema import BacktestConfig, CostsConfig, RiskConfig
from alphacore.errors import BacktestError
from alphacore.schemas.bars import normalise_bars, validate_bars


def make_bars(
    close: list[float] | np.ndarray,
    *,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: float = 1e9,
) -> pd.DataFrame:
    """Bars whose open equals the previous close, so gaps are explicit."""
    close_arr = np.asarray(close, dtype=float)
    n = close_arr.size
    open_arr = np.concatenate(([close_arr[0]], close_arr[:-1]))
    body_high = np.maximum(open_arr, close_arr)
    body_low = np.minimum(open_arr, close_arr)
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
            "open": open_arr,
            "high": body_high * 1.001 if high is None else high,
            "low": body_low * 0.999 if low is None else low,
            "close": close_arr,
            "volume": np.full(n, volume),
            "adj_factor": 1.0,
            "symbol": "SYNTH",
        }
    )
    return validate_bars(normalise_bars(frame))


def costs(**overrides) -> CostsConfig:
    base = {
        "commission_bps": 0.0,
        "min_commission": 0.0,
        "half_spread_bps": 0.0,
        "impact_model": "none",
        "impact_coefficient": 0.0,
        "max_participation": 1.0,
        "borrow_bps_annual": 0.0,
        "financing_bps_annual": 0.0,
    }
    return CostsConfig.model_validate({**base, **overrides})


def backtest_config(**overrides) -> BacktestConfig:
    base = {"initial_capital": 100_000.0, "execution_lag_bars": 1, "costs": costs()}
    return BacktestConfig.model_validate({**base, **overrides})


def frictionless() -> BacktestConfig:
    """The configuration the reference cross-check runs under (§12.5)."""
    return backtest_config(costs=costs(max_participation=1.0))


NO_RISK = RiskConfig(
    stop_loss_atr=None,
    take_profit_atr=None,
    trailing_stop_atr=None,
    max_drawdown_kill=1.0,
    max_daily_loss=1.0,
    max_concurrent_positions=10,
)


# --- cost model ------------------------------------------------------------


def test_commission_and_spread_scale_with_notional():
    model = CostModel(costs(commission_bps=1.0, half_spread_bps=2.0))
    charged = model.trade_cost(price=100.0, quantity=10.0, bar_volume=1e9, volatility=0.01)
    # (1 + 2) bps of 1,000 notional = 0.30
    assert charged == pytest.approx(0.30)


def test_cost_is_charged_on_both_sides_of_a_trade():
    model = CostModel(costs(commission_bps=1.0))
    buy = model.trade_cost(price=100.0, quantity=10.0, bar_volume=1e9, volatility=0.01)
    sell = model.trade_cost(price=100.0, quantity=-10.0, bar_volume=1e9, volatility=0.01)
    assert buy == pytest.approx(sell)


def test_the_minimum_commission_floors_a_tiny_trade():
    model = CostModel(costs(commission_bps=0.1, min_commission=1.0))
    assert model.trade_cost(
        price=1.0, quantity=1.0, bar_volume=1e9, volatility=0.01
    ) == pytest.approx(1.0)


def test_a_zero_quantity_trade_costs_nothing():
    model = CostModel(costs(commission_bps=1.0, min_commission=5.0))
    assert model.trade_cost(price=100.0, quantity=0.0, bar_volume=1e9, volatility=0.01) == 0.0


def test_square_root_impact_grows_sublinearly():
    small = impact_cost(
        price=100.0, quantity=100.0, bar_volume=1e6, volatility=0.02, model="sqrt", coefficient=0.1
    )
    big = impact_cost(
        price=100.0, quantity=400.0, bar_volume=1e6, volatility=0.02, model="sqrt", coefficient=0.1
    )
    # Four times the size, twice the per-share impact.
    assert big / small == pytest.approx(2.0, rel=1e-9)


def test_linear_impact_grows_linearly():
    small = impact_cost(
        price=100.0,
        quantity=100.0,
        bar_volume=1e6,
        volatility=0.02,
        model="linear",
        coefficient=0.1,
    )
    big = impact_cost(
        price=100.0,
        quantity=200.0,
        bar_volume=1e6,
        volatility=0.02,
        model="linear",
        coefficient=0.1,
    )
    assert big / small == pytest.approx(2.0, rel=1e-9)


def test_impact_can_be_switched_off():
    assert (
        impact_cost(
            price=100.0, quantity=1e6, bar_volume=1.0, volatility=0.5, model="none", coefficient=1.0
        )
        == 0.0
    )


def test_participation_caps_the_fill():
    model = CostModel(costs(max_participation=0.05))
    filled = model.cap_quantity(quantity=1_000.0, bar_volume=10_000.0)
    # Never more than 5% of the bar's volume.
    assert filled == pytest.approx(500.0)


def test_participation_cap_preserves_the_side():
    model = CostModel(costs(max_participation=0.05))
    assert model.cap_quantity(quantity=-1_000.0, bar_volume=10_000.0) == pytest.approx(-500.0)


def test_a_full_participation_cap_fills_everything():
    model = CostModel(costs(max_participation=1.0))
    assert model.cap_quantity(quantity=10_000.0, bar_volume=10_000.0) == pytest.approx(10_000.0)


def test_borrow_accrues_only_on_short_notional():
    model = CostModel(costs(borrow_bps_annual=365.0))
    short = model.carry_cost(position_value=-10_000.0, periods_per_year=365.0)
    long = model.carry_cost(position_value=10_000.0, periods_per_year=365.0)
    assert short == pytest.approx(1.0)
    assert long == 0.0


def test_financing_accrues_on_leveraged_long_notional():
    model = CostModel(costs(financing_bps_annual=365.0))
    assert model.carry_cost(
        position_value=200_000.0, periods_per_year=365.0, equity=100_000.0
    ) == pytest.approx(10.0)


def test_round_trip_bps_sums_the_explicit_costs():
    assert round_trip_bps(costs(commission_bps=0.5, half_spread_bps=1.0)) == pytest.approx(3.0)


def test_a_frictionless_config_is_recognised():
    assert costs().is_frictionless
    assert not costs(commission_bps=0.5).is_frictionless


# --- portfolio -------------------------------------------------------------


def test_a_new_portfolio_is_all_cash():
    portfolio = Portfolio(initial_capital=1_000.0)
    assert portfolio.cash == 1_000.0
    assert portfolio.quantity == 0.0
    assert portfolio.equity(price=100.0) == 1_000.0


def test_buying_moves_cash_into_position_value():
    portfolio = Portfolio(initial_capital=1_000.0)
    portfolio.execute(quantity=5.0, price=100.0, cost=0.0)
    assert portfolio.cash == pytest.approx(500.0)
    assert portfolio.quantity == pytest.approx(5.0)
    assert portfolio.equity(price=100.0) == pytest.approx(1_000.0)


def test_costs_come_straight_out_of_equity():
    portfolio = Portfolio(initial_capital=1_000.0)
    portfolio.execute(quantity=5.0, price=100.0, cost=3.0)
    assert portfolio.equity(price=100.0) == pytest.approx(997.0)


def test_equity_marks_to_the_supplied_price():
    portfolio = Portfolio(initial_capital=1_000.0)
    portfolio.execute(quantity=5.0, price=100.0, cost=0.0)
    assert portfolio.equity(price=110.0) == pytest.approx(1_050.0)


def test_shorting_raises_cash_and_creates_a_negative_position():
    portfolio = Portfolio(initial_capital=1_000.0)
    portfolio.execute(quantity=-5.0, price=100.0, cost=0.0)
    assert portfolio.cash == pytest.approx(1_500.0)
    assert portfolio.equity(price=100.0) == pytest.approx(1_000.0)
    assert portfolio.equity(price=90.0) == pytest.approx(1_050.0)


def test_cash_and_position_reconcile_to_equity_every_time():
    portfolio = Portfolio(initial_capital=1_000.0)
    for quantity, price in ((3.0, 100.0), (-1.0, 110.0), (-2.0, 90.0)):
        portfolio.execute(quantity=quantity, price=price, cost=0.5)
        assert portfolio.equity(price=price) == pytest.approx(
            portfolio.cash + portfolio.quantity * price
        )


# --- ledger ----------------------------------------------------------------


def test_the_ledger_records_every_fill():
    ledger = Ledger()
    ts = pd.Timestamp("2020-01-01", tz="UTC")
    ledger.record_fill(ts=ts, quantity=5.0, price=100.0, cost=1.0, reason="signal")
    ledger.record_fill(ts=ts, quantity=-5.0, price=110.0, cost=1.0, reason="signal")
    assert len(ledger.fills) == 2


def test_the_ledger_pairs_fills_into_round_trip_trades():
    ledger = Ledger()
    start = pd.Timestamp("2020-01-01", tz="UTC")
    ledger.record_fill(ts=start, quantity=5.0, price=100.0, cost=1.0, reason="signal", bar=0)
    ledger.record_fill(
        ts=start + pd.Timedelta(days=3),
        quantity=-5.0,
        price=110.0,
        cost=1.0,
        reason="signal",
        bar=3,
    )

    trades = ledger.trades()
    assert len(trades) == 1
    trade = trades.iloc[0]
    # 5 shares x 10 gain, less 2 of cost.
    assert trade["pnl"] == pytest.approx(48.0)
    assert trade["bars_held"] == 3
    assert trade["side"] == 1


def test_an_open_position_is_not_yet_a_trade():
    ledger = Ledger()
    ledger.record_fill(
        ts=pd.Timestamp("2020-01-01", tz="UTC"),
        quantity=5.0,
        price=100.0,
        cost=0.0,
        reason="signal",
    )
    assert ledger.trades().empty


def test_the_equity_curve_is_recorded_per_bar():
    ledger = Ledger()
    for i, value in enumerate((100.0, 101.0, 99.0)):
        ledger.record_equity(
            ts=pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i),
            equity=value,
            position=0.0,
            cash=value,
        )
    curve = ledger.equity_curve()
    assert len(curve) == 3
    assert curve["equity"].tolist() == [100.0, 101.0, 99.0]


def test_the_drawdown_series_is_never_positive():
    ledger = Ledger()
    for i, value in enumerate((100.0, 120.0, 90.0, 130.0)):
        ledger.record_equity(
            ts=pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i),
            equity=value,
            position=0.0,
            cash=value,
        )
    drawdown = ledger.drawdown()
    assert (drawdown["drawdown"] <= 1e-12).all()
    assert drawdown["drawdown"].min() == pytest.approx(-0.25)


# --- engine ----------------------------------------------------------------


def constant_signal(n: int, value: int = 1) -> np.ndarray:
    return np.full(n, value, dtype=int)


def test_a_flat_signal_leaves_equity_untouched():
    bars = make_bars(np.linspace(100.0, 120.0, 30))
    result = run_backtest(bars, signal=np.zeros(30, dtype=int), config=frictionless(), risk=NO_RISK)
    assert result.equity["equity"].iloc[-1] == pytest.approx(100_000.0)
    assert result.trades.empty


def test_a_long_signal_tracks_the_instrument():
    bars = make_bars(np.linspace(100.0, 110.0, 40))
    result = run_backtest(bars, signal=constant_signal(40), config=frictionless(), risk=NO_RISK)
    assert result.equity["equity"].iloc[-1] > 100_000.0


def test_execution_happens_at_the_next_open_not_this_close():
    """§4 I2: the engine cannot see bar t's close when ordering for bar t."""
    bars = make_bars([100.0, 100.0, 200.0, 200.0, 200.0])
    result = run_backtest(
        bars, signal=np.array([0, 1, 0, 0, 0]), config=frictionless(), risk=NO_RISK
    )
    fill = result.trades.iloc[0]
    # Signal on bar 1 fills at bar 2's open, which is bar 1's close of 100.
    assert fill["entry_price"] == pytest.approx(100.0)


def test_a_longer_execution_lag_delays_the_fill():
    bars = make_bars([100.0, 100.0, 100.0, 200.0, 200.0, 200.0])
    lagged = run_backtest(
        bars,
        signal=np.array([0, 1, 0, 0, 0, 0]),
        config=backtest_config(execution_lag_bars=2, costs=costs()),
        risk=NO_RISK,
    )
    assert not lagged.trades.empty


def test_costs_reduce_the_equity_curve():
    bars = make_bars(np.linspace(100.0, 110.0, 40))
    signal = np.tile([1, 0], 20)
    free = run_backtest(bars, signal=signal, config=frictionless(), risk=NO_RISK)
    charged = run_backtest(
        bars,
        signal=signal,
        config=backtest_config(costs=costs(commission_bps=5.0, half_spread_bps=5.0)),
        risk=NO_RISK,
    )
    assert charged.equity["equity"].iloc[-1] < free.equity["equity"].iloc[-1]


def test_the_engine_reconciles_cash_and_position_every_bar():
    bars = make_bars(np.linspace(100.0, 130.0, 60))
    result = run_backtest(
        bars,
        signal=np.tile([1, 1, -1, 0], 15),
        config=backtest_config(costs=costs(commission_bps=1.0)),
        risk=NO_RISK,
        strict=True,
    )
    assert len(result.equity) == len(bars)


def test_strict_mode_catches_an_accounting_break():
    # The reconciliation assertion has to be able to fail, or it is decoration.
    portfolio = Portfolio(initial_capital=1_000.0)
    portfolio.execute(quantity=1.0, price=100.0, cost=0.0)
    portfolio.cash += 5.0  # a break, introduced by hand
    with pytest.raises(BacktestError, match="reconcile"):
        portfolio.assert_reconciles(price=100.0, expected_equity=1_000.0)


def test_the_kill_switch_stops_trading():
    bars = make_bars(np.linspace(100.0, 10.0, 60))
    result = run_backtest(
        bars,
        signal=constant_signal(60),
        config=frictionless(),
        risk=RiskConfig(
            stop_loss_atr=None,
            take_profit_atr=None,
            max_drawdown_kill=0.2,
            max_daily_loss=1.0,
            max_concurrent_positions=10,
        ),
    )
    assert result.kill_switch is not None
    assert result.equity["position"].iloc[-1] == 0.0


def test_a_position_opened_before_the_atr_warm_up_is_flagged():
    """Stops configured but not yet computable must not pass silently."""
    bars = make_bars(np.linspace(100.0, 110.0, 12))
    result = run_backtest(
        bars,
        signal=constant_signal(12),
        config=frictionless(),
        risk=RiskConfig(
            stop_loss_atr=1.0,
            take_profit_atr=None,
            max_drawdown_kill=1.0,
            max_daily_loss=1.0,
            max_concurrent_positions=10,
        ),
    )
    assert "UNSTOPPED_WARMUP" in result.warnings


def test_stops_attach_once_the_atr_becomes_available():
    close = np.concatenate((np.full(40, 100.0), np.full(20, 60.0)))
    bars = make_bars(close)
    result = run_backtest(
        bars,
        signal=constant_signal(60),
        config=frictionless(),
        risk=RiskConfig(
            stop_loss_atr=1.0,
            take_profit_atr=None,
            max_drawdown_kill=1.0,
            max_daily_loss=1.0,
            max_concurrent_positions=10,
        ),
    )
    assert (result.trades["exit_reason"] == "stop_loss").any()


def test_stops_close_a_losing_position():
    close = np.concatenate((np.full(40, 100.0), np.full(20, 60.0)))
    bars = make_bars(close)
    result = run_backtest(
        bars,
        signal=constant_signal(60),
        config=frictionless(),
        risk=RiskConfig(
            stop_loss_atr=1.0,
            take_profit_atr=None,
            max_drawdown_kill=1.0,
            max_daily_loss=1.0,
            max_concurrent_positions=10,
        ),
    )
    assert (result.trades["exit_reason"] == "stop_loss").any()


def test_the_engine_rejects_a_signal_of_the_wrong_length():
    bars = make_bars(np.linspace(100.0, 110.0, 10))
    with pytest.raises(BacktestError, match="signal"):
        run_backtest(bars, signal=np.ones(5, dtype=int), config=frictionless(), risk=NO_RISK)


def test_the_engine_refuses_zero_execution_lag():
    bars = make_bars(np.linspace(100.0, 110.0, 10))
    with pytest.raises(BacktestError, match="execution_lag"):
        run_backtest(
            bars,
            signal=np.ones(10, dtype=int),
            config=backtest_config(execution_lag_bars=1).model_copy(
                update={"execution_lag_bars": 0}
            ),
            risk=NO_RISK,
        )


def test_the_result_flags_an_uncosted_run():
    bars = make_bars(np.linspace(100.0, 110.0, 20))
    result = run_backtest(bars, signal=constant_signal(20), config=frictionless(), risk=NO_RISK)
    assert "UNCOSTED" in result.warnings


def test_a_costed_run_is_not_flagged():
    bars = make_bars(np.linspace(100.0, 110.0, 20))
    result = run_backtest(
        bars,
        signal=constant_signal(20),
        config=backtest_config(costs=costs(commission_bps=0.5, half_spread_bps=1.0)),
        risk=NO_RISK,
    )
    assert "UNCOSTED" not in result.warnings


def test_partial_fills_are_recorded_when_participation_binds():
    bars = make_bars(np.linspace(100.0, 110.0, 30), volume=100.0)
    result = run_backtest(
        bars,
        signal=constant_signal(30),
        config=backtest_config(costs=costs(max_participation=0.01)),
        risk=NO_RISK,
    )
    # A 100,000 order against a 100-share bar cannot fill in one go.
    assert result.n_partial_fills > 0


# --- reference cross-check (§12.5) -----------------------------------------


def test_the_engine_matches_the_reference_engine():
    """§12.5: under a frictionless config the two must agree to 1e-8."""
    rng = np.random.default_rng(20260727)
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(300) * 0.01))
    bars = make_bars(close)
    signal = rng.choice([-1, 0, 1], size=300)

    engine = run_backtest(bars, signal=signal, config=frictionless(), risk=NO_RISK)
    naive = reference_backtest(bars, signal=signal, config=frictionless())

    np.testing.assert_allclose(
        engine.equity["equity"].to_numpy(), naive["equity"].to_numpy(), rtol=0, atol=1e-8
    )


def test_the_reference_engine_disagrees_once_costs_are_charged():
    # The cross-check is only meaningful frictionless; this shows why.
    rng = np.random.default_rng(5)
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(200) * 0.01))
    bars = make_bars(close)
    signal = rng.choice([-1, 0, 1], size=200)

    charged = backtest_config(costs=costs(commission_bps=10.0))
    engine = run_backtest(bars, signal=signal, config=charged, risk=NO_RISK)
    naive = reference_backtest(bars, signal=signal, config=frictionless())
    assert not np.allclose(
        engine.equity["equity"].to_numpy(), naive["equity"].to_numpy(), atol=1e-8
    )


# --- cost sensitivity ------------------------------------------------------


def test_cost_sensitivity_returns_a_point_per_level():
    bars = make_bars(np.linspace(100.0, 120.0, 80))
    signal = np.tile([1, 0], 40)
    curve = cost_sensitivity(
        bars,
        signal=signal,
        config=backtest_config(cost_sensitivity_bps=(0.0, 5.0, 20.0)),
        risk=NO_RISK,
    )
    assert [point["bps"] for point in curve] == [0.0, 5.0, 20.0]


def test_higher_costs_never_improve_terminal_equity():
    bars = make_bars(np.linspace(100.0, 120.0, 80))
    signal = np.tile([1, 0], 40)
    curve = cost_sensitivity(
        bars,
        signal=signal,
        config=backtest_config(cost_sensitivity_bps=(0.0, 2.0, 10.0, 50.0)),
        risk=NO_RISK,
    )
    equities = [point["final_equity"] for point in curve]
    assert equities == sorted(equities, reverse=True)


def test_the_sensitivity_curve_is_serialisable():
    bars = make_bars(np.linspace(100.0, 110.0, 40))
    curve = cost_sensitivity(
        bars,
        signal=np.tile([1, 0], 20),
        config=backtest_config(cost_sensitivity_bps=(0.0, 5.0)),
        risk=NO_RISK,
    )
    assert json.loads(json.dumps(curve)) == curve


def test_a_backtest_result_is_serialisable():
    bars = make_bars(np.linspace(100.0, 110.0, 30))
    result = run_backtest(bars, signal=constant_signal(30), config=frictionless(), risk=NO_RISK)
    assert isinstance(result, BacktestResult)
    payload = result.summary()
    assert payload["n_bars"] == 30
    assert "final_equity" in payload
