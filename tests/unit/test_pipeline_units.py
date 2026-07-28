"""The pieces Phase 10 added to the library, tested away from a full run.

The end-to-end behaviour lives in `tests/integration/test_full_run.py`. These
are the units that run wrote, each with a case that would have caught the bug
it was written for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacore.backtest.ledger import Fill, Ledger
from alphacore.config import load_config
from alphacore.errors import EvaluationError, ModelError, PipelineError
from alphacore.eval import net_sharpe, rolling_sharpe, sharpe_ratio
from alphacore.models import ECONOMIC_COLUMNS, attach_economics
from alphacore.pipelines.backtest import _VOL_WINDOW, _signal_and_weights, sharpe_variance
from alphacore.pipelines.build_features import primary_symbol


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")


# --------------------------------------------------------------------------
# net_sharpe
# --------------------------------------------------------------------------


def test_a_strategy_that_never_moved_scores_zero_rather_than_raising():
    assert net_sharpe(np.zeros(100), periods_per_year=252.0) == 0.0


def test_a_constant_non_zero_return_still_raises():
    # An infinite Sharpe is a bug upstream. The all-zero shortcut must not
    # swallow this case with it.
    with pytest.raises(EvaluationError, match="zero volatility"):
        net_sharpe(np.full(100, 0.001), periods_per_year=252.0)


def test_net_sharpe_matches_sharpe_ratio_on_a_moving_series():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, 500)
    assert net_sharpe(returns, periods_per_year=252.0) == pytest.approx(
        sharpe_ratio(returns, periods_per_year=252.0)
    )


# --------------------------------------------------------------------------
# rolling_sharpe
# --------------------------------------------------------------------------


def test_rolling_sharpe_drops_the_warm_up_rather_than_padding_it():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0005, 0.01, 300), index=_index(300))
    rolling = rolling_sharpe(returns, window=63, periods_per_year=252.0)
    assert len(rolling) == 300 - 63 + 1
    assert rolling.index[0] == returns.index[62]
    assert rolling.notna().all()


def test_a_flat_stretch_scores_zero_not_nan():
    # A window spent flat earned nothing and risked nothing. Before this, the
    # whole window was dropped and a smoke run logged 671 warnings.
    returns = pd.Series(np.zeros(100), index=_index(100))
    returns.iloc[:20] = 0.01
    returns.iloc[1:20:2] = -0.008
    rolling = rolling_sharpe(returns, window=30, periods_per_year=252.0)
    assert rolling.notna().all()
    assert rolling.iloc[-1] == 0.0


def test_a_constant_non_zero_window_raises():
    returns = pd.Series(np.full(100, 0.002), index=_index(100))
    with pytest.raises(EvaluationError, match="constant non-zero"):
        rolling_sharpe(returns, window=30, periods_per_year=252.0)


def test_a_window_longer_than_the_sample_raises():
    returns = pd.Series(np.linspace(-0.01, 0.01, 40), index=_index(40))
    with pytest.raises(EvaluationError, match="Cannot compute"):
        rolling_sharpe(returns, window=252, periods_per_year=252.0)


# --------------------------------------------------------------------------
# The ledger's round-trip pairing
# --------------------------------------------------------------------------


def _ledger(quantities: list[float], prices: list[float]) -> Ledger:
    ledger = Ledger()
    index = _index(len(quantities))
    for i, (quantity, price) in enumerate(zip(quantities, prices, strict=True)):
        ledger.fills.append(
            Fill(ts=index[i], quantity=quantity, price=price, cost=0.0, reason="signal", bar=i)
        )
    return ledger


def test_scaling_into_a_position_is_not_a_round_trip():
    # Vol-target sizing rebalances on most bars. Counting a same-direction
    # top-up as a completed trade reported 744 trades for a position that was
    # opened once and never closed.
    ledger = _ledger([100.0, 50.0, 25.0], [10.0, 11.0, 12.0])
    assert ledger.trades().empty


def test_scaling_in_blends_the_entry_price():
    ledger = _ledger([100.0, 100.0, -200.0], [10.0, 12.0, 13.0])
    trades = ledger.trades()
    assert len(trades) == 1
    assert trades.iloc[0]["entry_price"] == pytest.approx(11.0)
    assert trades.iloc[0]["quantity"] == pytest.approx(200.0)
    assert trades.iloc[0]["pnl"] == pytest.approx(200.0 * (13.0 - 11.0))


def test_a_partial_exit_closes_only_what_it_sold():
    ledger = _ledger([100.0, -40.0], [10.0, 12.0])
    trades = ledger.trades()
    assert len(trades) == 1
    assert trades.iloc[0]["quantity"] == pytest.approx(40.0)
    assert trades.iloc[0]["pnl"] == pytest.approx(40.0 * 2.0)


def test_the_remainder_of_a_partial_exit_keeps_its_original_entry_price():
    # The unsold half was bought at 10, not at the 12 it was partly sold at.
    ledger = _ledger([100.0, -50.0, -50.0], [10.0, 12.0, 14.0])
    trades = ledger.trades()
    assert len(trades) == 2
    assert trades.iloc[1]["entry_price"] == pytest.approx(10.0)
    assert trades.iloc[1]["pnl"] == pytest.approx(50.0 * 4.0)


def test_a_flip_opens_a_fresh_trade_at_the_fill_price():
    ledger = _ledger([100.0, -150.0, 50.0], [10.0, 12.0, 11.0])
    trades = ledger.trades()
    assert len(trades) == 2
    assert trades.iloc[0]["side"] == 1
    assert trades.iloc[1]["side"] == -1
    assert trades.iloc[1]["entry_price"] == pytest.approx(12.0)


def test_the_fill_log_records_every_execution():
    ledger = _ledger([100.0, -100.0], [10.0, 11.0])
    log = ledger.fill_log()
    assert list(log.columns) == [
        "quantity",
        "requested",
        "price",
        "cost",
        "reason",
        "bar",
        "partial",
    ]
    assert len(log) == 2
    assert log.index.name == "ts"


def test_an_empty_fill_log_still_has_its_columns():
    assert Ledger().fill_log().empty
    assert "quantity" in Ledger().fill_log().columns


# --------------------------------------------------------------------------
# attach_economics
# --------------------------------------------------------------------------


def _row(model: str, *, tstat: float) -> dict:
    return {
        "model": model,
        "rank": 0,
        "n_predictions": 100,
        "n_folds": 3,
        "n_trials": 1,
        "flags": ["NO_BACKTEST"],
        "ic_tstat": tstat,
        **dict.fromkeys(ECONOMIC_COLUMNS, None),
    }


def test_ranking_switches_from_ic_to_net_sharpe():
    # Before the backtest exists the leaderboard ranks on the IC t-stat. Once
    # it does, net Sharpe decides — a model can rank first on prediction
    # quality and last on money (ADR 0007).
    rows = [_row("a", tstat=5.0), _row("b", tstat=0.5)]
    attached = attach_economics(rows, {"a": {"oos_net_sharpe": -0.5}, "b": {"oos_net_sharpe": 1.2}})
    assert [row["model"] for row in attached] == ["b", "a"]
    assert [row["rank"] for row in attached] == [1, 2]


def test_a_backtested_row_loses_its_no_backtest_flag():
    attached = attach_economics([_row("a", tstat=1.0)], {"a": {"oos_net_sharpe": 0.4}})
    assert "NO_BACKTEST" not in attached[0]["flags"]


def test_a_row_without_a_backtest_keeps_its_flag_and_is_not_dropped():
    attached = attach_economics(
        [_row("a", tstat=1.0), _row("b", tstat=2.0)], {"a": {"oos_net_sharpe": 0.4}}
    )
    kept = {row["model"]: row for row in attached}
    assert kept["b"]["flags"] == ["NO_BACKTEST"]
    assert kept["b"]["oos_net_sharpe"] is None


def test_a_weak_deflated_sharpe_is_flagged():
    attached = attach_economics([_row("a", tstat=1.0)], {"a": {"oos_net_sharpe": 2.0, "dsr": 0.2}})
    assert "LOW_DSR" in attached[0]["flags"]


def test_a_strong_deflated_sharpe_is_not_flagged():
    attached = attach_economics([_row("a", tstat=1.0)], {"a": {"oos_net_sharpe": 2.0, "dsr": 0.9}})
    assert "LOW_DSR" not in attached[0]["flags"]


def test_economics_for_an_unknown_model_raises():
    with pytest.raises(ModelError, match="not on the leaderboard"):
        attach_economics([_row("a", tstat=1.0)], {"ghost": {"oos_net_sharpe": 1.0}})


def test_an_unknown_economic_column_raises_rather_than_being_ignored():
    # A misspelled key would otherwise leave the column empty and the row
    # flagged NO_BACKTEST, with nothing saying why.
    with pytest.raises(ModelError, match="unknown columns"):
        attach_economics([_row("a", tstat=1.0)], {"a": {"oos_sharpe": 1.0}})


# --------------------------------------------------------------------------
# sharpe_variance
# --------------------------------------------------------------------------


def test_a_single_configuration_gets_a_positive_variance_floor():
    # A zero variance makes the expected maximum zero, which turns the
    # deflated Sharpe into the plain one and silently overstates the result.
    assert sharpe_variance([1.0]) > 0.0
    assert sharpe_variance([]) > 0.0


def test_identical_sharpes_still_get_the_floor():
    assert sharpe_variance([0.5, 0.5, 0.5]) > 0.0


def test_the_spread_of_a_search_is_measured_not_assumed():
    assert sharpe_variance([0.0, 1.0, 2.0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Sizing during the volatility warm-up
# --------------------------------------------------------------------------


def test_vol_target_sizing_holds_flat_until_its_window_has_warmed_up(smoke_config_file):
    """The warm-up must not be back-filled from the future (§4, I1).

    Back-filling the first computed volatility into the bars before it sizes a
    position at bar 1 using the volatility of bars 1 to 20 — information that
    does not exist at bar 1. The bars are held flat instead.
    """
    config = load_config(smoke_config_file, use_env=False)
    assert config.signal.sizing.method == "vol_target"

    n = 200
    rng = np.random.default_rng(3)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=_index(n),
    )
    proba = np.full(n, 0.99)

    signal, weights = _signal_and_weights(proba, bars, config, periods_per_year=252.0)
    window = _VOL_WINDOW

    assert (signal[:window] == 0).all(), "no position may be taken before a volatility exists"
    assert (weights[:window] == 0.0).all()
    assert np.isfinite(weights).all(), "sizing must stay defined across the warm-up"
    assert (signal[window:] == 1).all(), "and normal sizing must resume once it does"


# --------------------------------------------------------------------------
# primary_symbol
# --------------------------------------------------------------------------


def test_a_multi_symbol_config_is_refused_rather_than_silently_truncated(smoke_config_file):
    config = load_config(smoke_config_file, use_env=False)
    multi = config.model_copy(
        update={"data": config.data.model_copy(update={"symbols": ("SYNTH", "OTHER")})}
    )
    with pytest.raises(PipelineError, match="single-symbol"):
        primary_symbol(multi)


@pytest.fixture
def smoke_config_file(repo_root):
    return repo_root / "configs" / "experiments" / "smoke.yaml"
