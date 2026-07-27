"""Financial metrics, deflated Sharpe, PBO, and bootstrap intervals (§11)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from alphacore.errors import EvaluationError
from alphacore.eval.bootstrap import block_bootstrap_ci, politis_white_block_length
from alphacore.eval.deflated import (
    EULER_MASCHERONI,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)
from alphacore.eval.financial import (
    annualised_return,
    annualised_volatility,
    break_even_cost_bps,
    calmar_ratio,
    cvar,
    drawdown_series,
    max_drawdown,
    return_metrics,
    sharpe_ratio,
    sortino_ratio,
    trade_metrics,
    ulcer_index,
    var,
)
from alphacore.eval.pbo import cscv_pbo

PERIODS = 252.0


def equity_from_returns(returns: np.ndarray, start: float = 100.0) -> pd.Series:
    index = pd.date_range("2015-01-01", periods=len(returns) + 1, freq="D", tz="UTC")
    return pd.Series(start * np.cumprod(np.concatenate(([1.0], 1.0 + returns))), index=index)


# --- return metrics --------------------------------------------------------


def test_a_constant_return_series_has_no_defined_sharpe():
    # numpy's std of a constant array is 2.18e-19, not 0.0, so this also
    # guards the tolerance that makes the check actually fire.
    with pytest.raises(EvaluationError, match="zero volatility"):
        sharpe_ratio(np.full(100, 0.001), periods_per_year=PERIODS)


def test_sharpe_matches_the_hand_computed_definition():
    rng = np.random.default_rng(1)
    returns = rng.standard_normal(500) * 0.01 + 0.0005
    expected = returns.mean() / returns.std(ddof=1) * math.sqrt(PERIODS)
    assert sharpe_ratio(returns, periods_per_year=PERIODS) == pytest.approx(expected)


def test_sharpe_scales_with_the_square_root_of_the_annualisation():
    rng = np.random.default_rng(2)
    returns = rng.standard_normal(500) * 0.01
    low = sharpe_ratio(returns, periods_per_year=252.0)
    high = sharpe_ratio(returns, periods_per_year=252.0 * 4)
    assert high == pytest.approx(low * 2.0)


def test_annualised_return_is_geometric_not_arithmetic():
    # +50% then -50% is a 13.4% loss, not break-even.
    returns = np.array([0.5, -0.5])
    assert annualised_return(returns, periods_per_year=2.0) == pytest.approx(-0.25)


def test_annualised_volatility_uses_the_sample_deviation():
    rng = np.random.default_rng(3)
    returns = rng.standard_normal(400) * 0.02
    expected = returns.std(ddof=1) * math.sqrt(PERIODS)
    assert annualised_volatility(returns, periods_per_year=PERIODS) == pytest.approx(expected)


def test_sortino_only_penalises_downside():
    returns = np.array([0.02, 0.02, -0.01, 0.03, -0.01])
    assert sortino_ratio(returns, periods_per_year=PERIODS) > sharpe_ratio(
        returns, periods_per_year=PERIODS
    )


def test_sortino_of_an_all_positive_series_is_refused():
    with pytest.raises(EvaluationError, match="no downside"):
        sortino_ratio(np.array([0.01, 0.02, 0.03]), periods_per_year=PERIODS)


def test_max_drawdown_is_hand_computable():
    equity = pd.Series([100.0, 120.0, 90.0, 130.0])
    # Peak 120 to trough 90 is -25%.
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_drawdown_series_is_never_positive():
    equity = pd.Series([100.0, 120.0, 90.0, 130.0, 125.0])
    series = drawdown_series(equity)
    assert (series <= 1e-12).all()
    assert series.iloc[0] == pytest.approx(0.0)


def test_calmar_divides_return_by_the_worst_drawdown():
    rng = np.random.default_rng(4)
    returns = rng.standard_normal(500) * 0.01 + 0.001
    equity = equity_from_returns(returns)
    calmar = calmar_ratio(returns, equity, periods_per_year=PERIODS)
    expected = annualised_return(returns, periods_per_year=PERIODS) / abs(max_drawdown(equity))
    assert calmar == pytest.approx(expected)


def test_ulcer_index_accumulates_time_spent_underwater():
    # Drawdowns enter squared, so a long shallow one only outweighs a deep
    # short one once it lasts more than (0.40 / 0.05)^2 = 64 bars.
    shallow_long = pd.Series([100.0] + [95.0] * 200 + [100.0])
    deep_short = pd.Series([100.0, 60.0] + [100.0] * 200)
    assert ulcer_index(shallow_long) > ulcer_index(deep_short)


def test_max_drawdown_alone_cannot_tell_those_two_apart():
    # Which is exactly why the ulcer index is reported alongside it.
    shallow_long = pd.Series([100.0] + [95.0] * 200 + [100.0])
    deep_short = pd.Series([100.0, 60.0] + [100.0] * 200)
    assert max_drawdown(deep_short) < max_drawdown(shallow_long)


def test_var_is_the_loss_quantile():
    returns = np.linspace(-0.10, 0.10, 101)
    # The 5th percentile of a uniform grid from -10% to +10%.
    assert var(returns, level=0.95) == pytest.approx(-0.09, abs=1e-9)


def test_cvar_is_at_least_as_bad_as_var():
    rng = np.random.default_rng(5)
    returns = rng.standard_normal(2_000) * 0.02
    assert cvar(returns, level=0.95) <= var(returns, level=0.95)


def test_return_metrics_reports_the_full_set():
    rng = np.random.default_rng(6)
    returns = rng.standard_normal(800) * 0.01 + 0.0004
    metrics = return_metrics(returns, equity_from_returns(returns), periods_per_year=PERIODS)
    for key in (
        "annualised_return",
        "annualised_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "max_drawdown_duration",
        "ulcer_index",
        "var_95",
        "cvar_95",
        "var_99",
        "cvar_99",
        "skew",
        "kurtosis",
        "best_period",
        "worst_period",
        "periods_per_year",
    ):
        assert key in metrics, key
        assert np.isfinite(metrics[key]), key


def test_return_metrics_records_the_annualisation_factor():
    # §11.1: a Sharpe with an undocumented annualisation is unfalsifiable.
    rng = np.random.default_rng(7)
    returns = rng.standard_normal(300) * 0.01
    metrics = return_metrics(returns, equity_from_returns(returns), periods_per_year=365.0)
    assert metrics["periods_per_year"] == 365.0


def test_metrics_refuse_an_empty_series():
    with pytest.raises(EvaluationError, match="empty"):
        sharpe_ratio(np.array([]), periods_per_year=PERIODS)


# --- trade metrics ---------------------------------------------------------


def make_trades(pnls: list[float]) -> pd.DataFrame:
    n = len(pnls)
    return pd.DataFrame(
        {
            "entry_ts": pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
            "exit_ts": pd.date_range("2020-01-02", periods=n, freq="D", tz="UTC"),
            "side": [1] * n,
            "quantity": [10.0] * n,
            "entry_price": [100.0] * n,
            "exit_price": [100.0 + p for p in pnls],
            "pnl": pnls,
            "return_pct": [p / 1000.0 for p in pnls],
            "cost": [0.0] * n,
            "bars_held": [2] * n,
            "exit_reason": ["signal"] * n,
        }
    )


def test_trade_metrics_are_hand_computable():
    metrics = trade_metrics(make_trades([10.0, -5.0, 20.0, -5.0]))
    assert metrics["n_trades"] == 4
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["profit_factor"] == pytest.approx(30.0 / 10.0)
    assert metrics["average_win"] == pytest.approx(15.0)
    assert metrics["average_loss"] == pytest.approx(-5.0)
    assert metrics["payoff_ratio"] == pytest.approx(3.0)
    assert metrics["expectancy"] == pytest.approx(5.0)
    assert metrics["largest_winner"] == pytest.approx(20.0)
    assert metrics["largest_loser"] == pytest.approx(-5.0)


def test_max_consecutive_losses_counts_the_worst_run():
    metrics = trade_metrics(make_trades([1.0, -1.0, -1.0, -1.0, 1.0, -1.0]))
    assert metrics["max_consecutive_losses"] == 3


def test_trade_metrics_on_an_empty_blotter_are_zero_not_nan():
    metrics = trade_metrics(make_trades([]).iloc[:0])
    assert metrics["n_trades"] == 0
    assert all(np.isfinite(v) for v in metrics.values())


def test_profit_factor_with_no_losses_is_reported_not_infinite():
    metrics = trade_metrics(make_trades([1.0, 2.0]))
    assert np.isfinite(metrics["profit_factor"])


# --- break-even cost -------------------------------------------------------


def test_break_even_cost_is_zero_when_the_strategy_already_loses():
    curve = [{"bps": 0.0, "sharpe": -0.2}, {"bps": 5.0, "sharpe": -0.5}]
    assert break_even_cost_bps(curve) == pytest.approx(0.0)


def test_break_even_cost_interpolates_the_crossing():
    curve = [{"bps": 0.0, "sharpe": 1.0}, {"bps": 10.0, "sharpe": -1.0}]
    assert break_even_cost_bps(curve) == pytest.approx(5.0)


def test_break_even_cost_beyond_the_sweep_is_reported_as_the_last_level():
    curve = [{"bps": 0.0, "sharpe": 2.0}, {"bps": 20.0, "sharpe": 1.5}]
    # Never crosses zero within the sweep; the honest answer is "at least 20".
    assert break_even_cost_bps(curve) == pytest.approx(20.0)


# --- PSR / DSR / minTRL ----------------------------------------------------


def psr_by_hand(sr: float, benchmark: float, n: int, skew: float, kurt: float) -> float:
    """The §11.2 formula, transcribed independently for the test."""
    numerator = (sr - benchmark) * math.sqrt(n - 1)
    denominator = math.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    return float(norm.cdf(numerator / denominator))


def test_psr_matches_the_formula_for_normal_returns():
    value = probabilistic_sharpe_ratio(
        sharpe=0.5, n_observations=250, benchmark=0.0, skew=0.0, kurtosis=3.0
    )
    assert value == pytest.approx(psr_by_hand(0.5, 0.0, 250, 0.0, 3.0))


def test_psr_of_a_sharpe_equal_to_the_benchmark_is_one_half():
    assert probabilistic_sharpe_ratio(
        sharpe=0.7, n_observations=500, benchmark=0.7, skew=0.0, kurtosis=3.0
    ) == pytest.approx(0.5)


def test_psr_rises_with_sample_size():
    small = probabilistic_sharpe_ratio(sharpe=0.4, n_observations=60, benchmark=0.0)
    large = probabilistic_sharpe_ratio(sharpe=0.4, n_observations=2_000, benchmark=0.0)
    assert large > small


def test_negative_skew_lowers_psr():
    # A strategy that occasionally blows up deserves less confidence.
    # Sharpe here is *per observation*, so a realistic daily value is small —
    # 0.5 per day would saturate the normal CDF at 1.0 and hide the effect.
    symmetric = probabilistic_sharpe_ratio(sharpe=0.05, n_observations=500, skew=0.0, kurtosis=3.0)
    skewed = probabilistic_sharpe_ratio(sharpe=0.05, n_observations=500, skew=-1.5, kurtosis=3.0)
    assert skewed < symmetric


def test_fat_tails_lower_psr():
    normal = probabilistic_sharpe_ratio(sharpe=0.05, n_observations=500, kurtosis=3.0)
    fat = probabilistic_sharpe_ratio(sharpe=0.05, n_observations=500, kurtosis=30.0)
    assert fat < normal


def test_expected_max_sharpe_grows_with_the_trial_count():
    few = expected_max_sharpe(n_trials=5, variance=1.0)
    many = expected_max_sharpe(n_trials=1_000, variance=1.0)
    assert many > few > 0


def test_expected_max_sharpe_uses_the_euler_mascheroni_constant():
    assert pytest.approx(0.5772156649, abs=1e-9) == EULER_MASCHERONI


def test_a_single_trial_needs_no_haircut():
    assert expected_max_sharpe(n_trials=1, variance=1.0) == pytest.approx(0.0)


def test_dsr_is_never_above_psr_when_more_than_one_trial_was_run():
    kwargs = {"sharpe": 0.8, "n_observations": 500, "skew": 0.0, "kurtosis": 3.0}
    psr = probabilistic_sharpe_ratio(benchmark=0.0, **kwargs)
    dsr = deflated_sharpe_ratio(n_trials=50, sharpe_variance=1.0, **kwargs)
    assert dsr <= psr


def test_dsr_falls_as_the_trial_count_rises():
    kwargs = {"sharpe": 1.0, "n_observations": 1_000, "sharpe_variance": 1.0}
    assert deflated_sharpe_ratio(n_trials=1_000, **kwargs) < deflated_sharpe_ratio(
        n_trials=10, **kwargs
    )


def test_dsr_punishes_a_wide_spread_of_trial_sharpes():
    kwargs = {"sharpe": 1.0, "n_observations": 1_000, "n_trials": 100}
    assert deflated_sharpe_ratio(sharpe_variance=4.0, **kwargs) < deflated_sharpe_ratio(
        sharpe_variance=0.25, **kwargs
    )


def test_min_track_record_length_falls_as_sharpe_rises():
    weak = min_track_record_length(sharpe=0.3)
    strong = min_track_record_length(sharpe=1.5)
    assert strong < weak


def test_min_track_record_length_of_a_zero_sharpe_is_unreachable():
    with pytest.raises(EvaluationError, match="does not exceed the benchmark"):
        min_track_record_length(sharpe=0.0)


def test_deflated_metrics_reject_a_nonsense_trial_count():
    with pytest.raises(EvaluationError, match="n_trials"):
        deflated_sharpe_ratio(sharpe=1.0, n_observations=100, n_trials=0, sharpe_variance=1.0)


# --- bootstrap -------------------------------------------------------------


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(8)
    returns = rng.standard_normal(1_000) * 0.01 + 0.0006
    point = sharpe_ratio(returns, periods_per_year=PERIODS)
    low, high = block_bootstrap_ci(
        returns, lambda r: sharpe_ratio(r, periods_per_year=PERIODS), seed=1, n_samples=300
    )
    assert low < point < high


def test_a_pure_noise_series_has_an_interval_containing_zero():
    rng = np.random.default_rng(9)
    returns = rng.standard_normal(1_500) * 0.01
    low, high = block_bootstrap_ci(
        returns, lambda r: sharpe_ratio(r, periods_per_year=PERIODS), seed=2, n_samples=300
    )
    assert low <= 0.0 <= high


def test_the_bootstrap_is_reproducible():
    rng = np.random.default_rng(10)
    returns = rng.standard_normal(600) * 0.01
    stat = lambda r: sharpe_ratio(r, periods_per_year=PERIODS)  # noqa: E731
    assert block_bootstrap_ci(returns, stat, seed=3, n_samples=200) == block_bootstrap_ci(
        returns, stat, seed=3, n_samples=200
    )


def test_block_length_grows_with_sample_size():
    rng = np.random.default_rng(11)
    short = politis_white_block_length(rng.standard_normal(200))
    long = politis_white_block_length(rng.standard_normal(5_000))
    assert long >= short >= 1


def test_block_length_is_larger_for_autocorrelated_data():
    rng = np.random.default_rng(12)
    noise = rng.standard_normal(2_000)
    autocorrelated = np.convolve(noise, np.ones(20) / 20, mode="same")
    assert politis_white_block_length(autocorrelated) > politis_white_block_length(noise)


# --- PBO via CSCV ----------------------------------------------------------


def test_pbo_of_a_genuinely_best_strategy_is_low():
    rng = np.random.default_rng(13)
    n, n_configs = 1_200, 8
    returns = rng.standard_normal((n, n_configs)) * 0.01
    # Config 0 has a real, persistent edge.
    returns[:, 0] += 0.004
    result = cscv_pbo(returns, n_chunks=8)
    assert result["pbo"] < 0.3
    assert result["n_configs"] == n_configs


def test_pbo_of_pure_noise_is_near_a_half():
    rng = np.random.default_rng(14)
    returns = rng.standard_normal((1_200, 10)) * 0.01
    result = cscv_pbo(returns, n_chunks=8)
    # With no real edge, picking the in-sample best is a coin flip out of sample.
    assert 0.25 < result["pbo"] < 0.75


def test_pbo_reports_the_number_of_splits_evaluated():
    rng = np.random.default_rng(15)
    returns = rng.standard_normal((600, 5)) * 0.01
    result = cscv_pbo(returns, n_chunks=6)
    # C(6, 3) = 20 splits.
    assert result["n_splits"] == 20


def test_pbo_needs_more_than_one_configuration():
    rng = np.random.default_rng(16)
    with pytest.raises(EvaluationError, match="configurations"):
        cscv_pbo(rng.standard_normal((500, 1)), n_chunks=6)


def test_pbo_needs_an_even_chunk_count():
    rng = np.random.default_rng(17)
    with pytest.raises(EvaluationError, match="even"):
        cscv_pbo(rng.standard_normal((500, 4)), n_chunks=7)


def test_pbo_is_deterministic():
    rng = np.random.default_rng(18)
    returns = rng.standard_normal((800, 6)) * 0.01
    assert cscv_pbo(returns, n_chunks=6) == cscv_pbo(returns, n_chunks=6)
