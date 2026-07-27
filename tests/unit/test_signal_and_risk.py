"""Signal generation, sizing, stops, and risk limits (§6, Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from alphacore.config.schema import RiskConfig, SignalConfig, SizingConfig
from alphacore.errors import BacktestError
from alphacore.risk import (
    ExitReason,
    KillSwitch,
    LimitBreach,
    RiskLimiter,
    StopLevels,
    check_stop_exit,
    stop_levels,
    update_trailing_stop,
)
from alphacore.signal import (
    apply_hysteresis,
    count_flips,
    generate_signal,
    kelly_capped,
    size_positions,
    threshold_signal,
    vol_target_weight,
)


def config(**overrides) -> SignalConfig:
    base = {"long_threshold": 0.6, "short_threshold": 0.4, "hysteresis": 0.0}
    return SignalConfig.model_validate({**base, **overrides})


# --- thresholds ------------------------------------------------------------


def test_threshold_signal_maps_probabilities_to_sides():
    proba = np.array([0.9, 0.5, 0.1])
    np.testing.assert_array_equal(
        threshold_signal(proba, long_threshold=0.6, short_threshold=0.4, allow_short=True),
        [1, 0, -1],
    )


def test_the_dead_zone_between_thresholds_is_flat():
    proba = np.array([0.41, 0.5, 0.59])
    signal = threshold_signal(proba, long_threshold=0.6, short_threshold=0.4, allow_short=True)
    np.testing.assert_array_equal(signal, [0, 0, 0])


def test_shorting_can_be_disabled():
    proba = np.array([0.1, 0.9])
    signal = threshold_signal(proba, long_threshold=0.6, short_threshold=0.4, allow_short=False)
    np.testing.assert_array_equal(signal, [0, 1])


def test_thresholds_are_inclusive_at_the_boundary():
    proba = np.array([0.6, 0.4])
    np.testing.assert_array_equal(
        threshold_signal(proba, long_threshold=0.6, short_threshold=0.4, allow_short=True),
        [1, -1],
    )


def test_threshold_signal_rejects_probabilities_outside_the_unit_interval():
    with pytest.raises(BacktestError, match=r"\[0, 1\]"):
        threshold_signal(np.array([1.5]), long_threshold=0.6, short_threshold=0.4, allow_short=True)


# --- hysteresis ------------------------------------------------------------


def test_hysteresis_holds_a_position_through_a_marginal_reading():
    # Crosses above 0.6 then dips to 0.58 — inside the band, so it holds.
    proba = np.array([0.70, 0.58, 0.70])
    held = apply_hysteresis(
        proba, long_threshold=0.6, short_threshold=0.4, hysteresis=0.05, allow_short=True
    )
    np.testing.assert_array_equal(held, [1, 1, 1])


def test_hysteresis_still_exits_on_a_decisive_move():
    proba = np.array([0.70, 0.30])
    held = apply_hysteresis(
        proba, long_threshold=0.6, short_threshold=0.4, hysteresis=0.05, allow_short=True
    )
    np.testing.assert_array_equal(held, [1, -1])


def test_entering_requires_clearing_the_upper_band():
    # 0.62 clears the raw threshold but not threshold + hysteresis.
    proba = np.array([0.62])
    held = apply_hysteresis(
        proba, long_threshold=0.6, short_threshold=0.4, hysteresis=0.05, allow_short=True
    )
    np.testing.assert_array_equal(held, [0])


def test_zero_hysteresis_reproduces_plain_thresholding():
    rng = np.random.default_rng(1)
    proba = rng.uniform(size=500)
    plain = threshold_signal(proba, long_threshold=0.6, short_threshold=0.4, allow_short=True)
    held = apply_hysteresis(
        proba, long_threshold=0.6, short_threshold=0.4, hysteresis=0.0, allow_short=True
    )
    np.testing.assert_array_equal(held, plain)


def walk_proba(seed: int, n: int = 1_000) -> np.ndarray:
    """A probability path that wanders across the thresholds."""
    rng = np.random.default_rng(seed)
    return np.clip(0.5 + np.cumsum(rng.standard_normal(n) * 0.02) * 0.3, 0.01, 0.99)


def flip_pair(proba: np.ndarray, hysteresis: float) -> tuple[int, int]:
    """Flip counts with plain thresholding and with hysteresis."""
    plain = threshold_signal(proba, long_threshold=0.55, short_threshold=0.45, allow_short=True)
    held = apply_hysteresis(
        proba, long_threshold=0.55, short_threshold=0.45, hysteresis=hysteresis, allow_short=True
    )
    return count_flips(plain), count_flips(held)


@pytest.mark.parametrize("seed", range(20))
def test_hysteresis_never_increases_flips_on_any_path(seed):
    """Per path, hysteresis can only help. This must hold everywhere."""
    plain, held = flip_pair(walk_proba(seed), 0.03)
    assert held <= plain


def test_hysteresis_strictly_reduces_flips_across_random_walks():
    """The Phase 6 exit criterion, stated over paths rather than one path.

    Turnover is what the cost model charges for, so a device claiming to reduce
    flipping has to be shown to reduce it. Per-path strictness is the wrong
    claim: a path that crosses a threshold once has one flip either way. The
    honest statement is that it strictly reduces flips in aggregate, and helps
    on most individual paths.
    """
    pairs = [flip_pair(walk_proba(seed), 0.03) for seed in range(20)]
    total_plain = sum(p for p, _ in pairs)
    total_held = sum(h for _, h in pairs)

    assert total_held < total_plain
    improved = sum(1 for p, h in pairs if h < p)
    assert improved >= len(pairs) // 2


def test_more_hysteresis_never_increases_flips():
    rng = np.random.default_rng(11)
    proba = np.clip(0.5 + np.cumsum(rng.standard_normal(2_000) * 0.02) * 0.3, 0.01, 0.99)
    counts = [
        count_flips(
            apply_hysteresis(
                proba, long_threshold=0.55, short_threshold=0.45, hysteresis=h, allow_short=True
            )
        )
        for h in (0.0, 0.02, 0.05, 0.10)
    ]
    assert counts == sorted(counts, reverse=True)


def test_count_flips_counts_changes_not_positions():
    assert count_flips(np.array([1, 1, 1])) == 0
    assert count_flips(np.array([1, -1, 1])) == 2
    assert count_flips(np.array([0, 1, 0, -1])) == 3


def test_generate_signal_applies_the_configured_hysteresis():
    proba = np.array([0.70, 0.58, 0.70])
    np.testing.assert_array_equal(generate_signal(proba, config(hysteresis=0.05)), [1, 1, 1])


# --- sizing ----------------------------------------------------------------


def sizing(**overrides) -> SizingConfig:
    base = {"method": "fixed_fraction", "fixed_fraction": 0.5, "max_leverage": 1.0}
    return SizingConfig.model_validate({**base, **overrides})


def test_fixed_fraction_scales_the_signal():
    weights = size_positions(np.array([1, 0, -1]), sizing())
    np.testing.assert_allclose(weights, [0.5, 0.0, -0.5])


def test_vol_target_scales_inversely_with_volatility():
    signal = np.array([1, 1])
    weights = size_positions(
        signal,
        sizing(method="vol_target", target_annual_vol=0.10),
        volatility=np.array([0.10, 0.20]),
    )
    # Twice the volatility, half the position.
    assert weights[1] == pytest.approx(weights[0] / 2)


def test_vol_target_hits_its_target_when_unclipped():
    weights = size_positions(
        np.array([1]),
        sizing(method="vol_target", target_annual_vol=0.10, max_leverage=10.0),
        volatility=np.array([0.20]),
    )
    assert weights[0] == pytest.approx(0.5)


def test_vol_target_needs_a_volatility_series():
    with pytest.raises(BacktestError, match="volatility"):
        size_positions(np.array([1]), sizing(method="vol_target"))


def test_zero_volatility_sizes_to_zero_not_infinity():
    weights = size_positions(
        np.array([1]),
        sizing(method="vol_target", max_leverage=10.0),
        volatility=np.array([0.0]),
    )
    assert weights[0] == 0.0


def test_kelly_is_capped():
    # A near-certain edge would size enormously without the cap.
    weights = size_positions(
        np.array([1]),
        sizing(method="kelly_capped", kelly_cap=0.25, max_leverage=10.0),
        proba=np.array([0.99]),
    )
    assert weights[0] == pytest.approx(0.25)


def test_kelly_scales_with_confidence():
    weights = size_positions(
        np.array([1, 1]),
        sizing(method="kelly_capped", kelly_cap=1.0, max_leverage=10.0),
        proba=np.array([0.55, 0.75]),
    )
    assert weights[1] > weights[0]


def test_kelly_of_a_coin_flip_is_zero():
    weights = size_positions(
        np.array([1]),
        sizing(method="kelly_capped", kelly_cap=1.0, max_leverage=10.0),
        proba=np.array([0.5]),
    )
    assert weights[0] == pytest.approx(0.0)


def test_kelly_never_goes_negative_for_a_long_signal():
    # A long signal with an unfavourable probability sizes to zero, not short.
    weights = size_positions(
        np.array([1]),
        sizing(method="kelly_capped", kelly_cap=1.0, max_leverage=10.0),
        proba=np.array([0.2]),
    )
    assert weights[0] == 0.0


@pytest.mark.parametrize("method", ["fixed_fraction", "vol_target", "kelly_capped"])
@pytest.mark.parametrize("seed", range(5))
def test_sizing_never_exceeds_max_leverage(method, seed):
    """The Phase 6 exit criterion, over every method and randomised inputs.

    Every knob is set above the leverage cap, and the volatility series runs
    down to near zero — the case where vol targeting would size to infinity if
    the clip were missing.
    """
    rng = np.random.default_rng(seed)
    n = 500
    cap = 0.5
    signal = rng.choice([-1, 0, 1], size=n)
    weights = size_positions(
        signal,
        sizing(
            method=method,
            max_leverage=cap,
            fixed_fraction=1.0,
            kelly_cap=1.0,
            target_annual_vol=2.0,
        ),
        volatility=np.abs(rng.standard_normal(n)) * 0.05,
        proba=rng.uniform(size=n),
    )
    assert np.isfinite(weights).all()
    assert np.abs(weights).max() <= cap + 1e-12


def test_the_leverage_cap_actually_binds():
    # A property test that never approaches the cap proves nothing.
    weights = size_positions(
        np.ones(10, dtype=int),
        sizing(method="vol_target", max_leverage=0.5, target_annual_vol=2.0),
        volatility=np.full(10, 0.1),
    )
    assert np.abs(weights).max() == pytest.approx(0.5)


def test_sizing_preserves_the_signal_direction():
    rng = np.random.default_rng(3)
    signal = rng.choice([-1, 0, 1], size=200)
    weights = size_positions(signal, sizing(method="vol_target"), volatility=np.full(200, 0.2))
    assert (np.sign(weights) == np.sign(signal)).all()


def test_vol_target_weight_is_exposed_for_the_engine():
    assert vol_target_weight(0.2, target_annual_vol=0.1, max_leverage=10.0) == pytest.approx(0.5)


def test_kelly_capped_is_exposed_for_the_engine():
    assert kelly_capped(0.99, cap=0.25) == pytest.approx(0.25)


# --- stops -----------------------------------------------------------------


def risk(**overrides) -> RiskConfig:
    base = {"stop_loss_atr": 2.0, "take_profit_atr": 4.0, "trailing_stop_atr": None}
    return RiskConfig.model_validate({**base, **overrides})


def test_long_stop_sits_below_and_target_above():
    levels = stop_levels(entry_price=100.0, side=1, atr=1.0, config=risk())
    assert levels.stop == pytest.approx(98.0)
    assert levels.target == pytest.approx(104.0)


def test_short_stop_sits_above_and_target_below():
    levels = stop_levels(entry_price=100.0, side=-1, atr=1.0, config=risk())
    assert levels.stop == pytest.approx(102.0)
    assert levels.target == pytest.approx(96.0)


def test_stops_can_be_disabled():
    levels = stop_levels(
        entry_price=100.0, side=1, atr=1.0, config=risk(stop_loss_atr=None, take_profit_atr=None)
    )
    assert levels.stop is None
    assert levels.target is None


def test_a_long_stop_triggers_on_the_bar_low():
    levels = StopLevels(stop=98.0, target=104.0, side=1)
    assert check_stop_exit(levels, high=101.0, low=97.5) is ExitReason.STOP_LOSS


def test_a_long_target_triggers_on_the_bar_high():
    levels = StopLevels(stop=98.0, target=104.0, side=1)
    assert check_stop_exit(levels, high=104.5, low=99.0) is ExitReason.TAKE_PROFIT


def test_no_touch_means_no_exit():
    levels = StopLevels(stop=98.0, target=104.0, side=1)
    assert check_stop_exit(levels, high=103.0, low=99.0) is None


def test_the_stop_wins_a_same_bar_tie():
    """§12.1: never assume the favourable fill came first."""
    levels = StopLevels(stop=98.0, target=104.0, side=1)
    assert check_stop_exit(levels, high=105.0, low=97.0) is ExitReason.STOP_LOSS


def test_the_stop_wins_a_same_bar_tie_when_short():
    levels = StopLevels(stop=102.0, target=96.0, side=-1)
    assert check_stop_exit(levels, high=103.0, low=95.0) is ExitReason.STOP_LOSS


def test_a_trailing_stop_ratchets_up_but_never_down():
    config = risk(trailing_stop_atr=2.0)
    levels = StopLevels(stop=98.0, target=None, side=1)
    raised = update_trailing_stop(levels, price=110.0, atr=1.0, config=config)
    assert raised.stop == pytest.approx(108.0)

    lowered = update_trailing_stop(raised, price=100.0, atr=1.0, config=config)
    assert lowered.stop == pytest.approx(108.0)


def test_a_short_trailing_stop_ratchets_down():
    config = risk(trailing_stop_atr=2.0)
    levels = StopLevels(stop=102.0, target=None, side=-1)
    lowered = update_trailing_stop(levels, price=90.0, atr=1.0, config=config)
    assert lowered.stop == pytest.approx(92.0)


def test_trailing_is_a_no_op_when_unconfigured():
    levels = StopLevels(stop=98.0, target=None, side=1)
    assert update_trailing_stop(levels, price=200.0, atr=1.0, config=risk()) == levels


# --- limits ----------------------------------------------------------------


def limits(**overrides) -> RiskConfig:
    base = {
        "max_drawdown_kill": 0.25,
        "max_daily_loss": 0.03,
        "max_concurrent_positions": 2,
    }
    return RiskConfig.model_validate({**base, **overrides})


def test_a_fresh_limiter_permits_trading():
    limiter = RiskLimiter(limits(), initial_equity=1_000.0)
    assert limiter.check(equity=1_000.0, open_positions=0) == []


def test_the_drawdown_kill_switch_trips_and_latches():
    limiter = RiskLimiter(limits(), initial_equity=1_000.0)
    limiter.check(equity=1_200.0, open_positions=0)

    breaches = limiter.check(equity=800.0, open_positions=0)
    assert LimitBreach.MAX_DRAWDOWN in breaches
    assert limiter.killed

    # Recovering does not un-kill: the run is over (§risk).
    assert LimitBreach.MAX_DRAWDOWN in limiter.check(equity=1_300.0, open_positions=0)


def test_the_kill_switch_reports_why():
    limiter = RiskLimiter(limits(), initial_equity=1_000.0)
    limiter.check(equity=700.0, open_positions=0)
    assert limiter.kill_switch is not None
    assert limiter.kill_switch.reason is LimitBreach.MAX_DRAWDOWN
    assert limiter.kill_switch.drawdown == pytest.approx(-0.3)


def test_the_daily_loss_limit_trips_within_a_session():
    limiter = RiskLimiter(limits(), initial_equity=1_000.0)
    limiter.start_session(equity=1_000.0)
    assert LimitBreach.MAX_DAILY_LOSS in limiter.check(equity=950.0, open_positions=0)


def test_the_daily_loss_limit_resets_each_session():
    limiter = RiskLimiter(limits(), initial_equity=1_000.0)
    limiter.start_session(equity=1_000.0)
    limiter.check(equity=960.0, open_positions=0)

    limiter.start_session(equity=960.0)
    assert LimitBreach.MAX_DAILY_LOSS not in limiter.check(equity=955.0, open_positions=0)


def test_the_concurrency_limit_blocks_a_further_position():
    limiter = RiskLimiter(limits(max_concurrent_positions=2), initial_equity=1_000.0)
    assert limiter.check(equity=1_000.0, open_positions=2) == []
    assert LimitBreach.MAX_CONCURRENT in limiter.check(equity=1_000.0, open_positions=3)


def test_breaches_accumulate_rather_than_masking_each_other():
    limiter = RiskLimiter(limits(), initial_equity=1_000.0)
    limiter.start_session(equity=1_000.0)
    breaches = limiter.check(equity=700.0, open_positions=9)
    assert LimitBreach.MAX_DRAWDOWN in breaches
    assert LimitBreach.MAX_DAILY_LOSS in breaches
    assert LimitBreach.MAX_CONCURRENT in breaches


def test_the_equity_peak_tracks_the_high_water_mark():
    limiter = RiskLimiter(limits(), initial_equity=1_000.0)
    limiter.check(equity=1_500.0, open_positions=0)
    limiter.check(equity=1_400.0, open_positions=0)
    assert limiter.peak_equity == pytest.approx(1_500.0)


def test_a_kill_switch_is_serialisable():
    limiter = RiskLimiter(limits(), initial_equity=1_000.0)
    limiter.check(equity=100.0, open_positions=0)
    payload = limiter.kill_switch.to_dict()
    assert payload["reason"] == "max_drawdown"
    assert payload["drawdown"] < 0
    assert isinstance(KillSwitch.from_dict(payload), KillSwitch)
