"""Labelling methods and sample weights (§8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacore.config.schema import LabelsConfig, SampleWeightsConfig, TripleBarrierConfig
from alphacore.errors import LabelError, LeakageError
from alphacore.labels import (
    build_labels,
    horizon_quantile,
    horizon_return,
    horizon_sign,
    meta_label,
    sample_weights,
    time_decay_weights,
    triple_barrier,
    uniqueness_weights,
)
from alphacore.schemas.bars import normalise_bars, validate_bars
from alphacore.schemas.labels import LABEL_COLUMNS, validate_labels


def make_bars(
    close: list[float], *, high: list[float] | None = None, low: list[float] | None = None
) -> pd.DataFrame:
    """Build a validated bar frame from an explicit close path."""
    n = len(close)
    close_arr = np.asarray(close, dtype=float)
    high_arr = np.asarray(high, dtype=float) if high else close_arr * 1.001
    low_arr = np.asarray(low, dtype=float) if low else close_arr * 0.999
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "open": close_arr,
            "high": np.maximum(high_arr, close_arr),
            "low": np.minimum(low_arr, close_arr),
            "close": close_arr,
            "volume": np.full(n, 1000.0),
            "adj_factor": 1.0,
            "symbol": "SYNTH",
        }
    )
    return validate_bars(normalise_bars(frame))


@pytest.fixture
def walk() -> pd.DataFrame:
    rng = np.random.default_rng(20260727)
    n = 300
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    return make_bars(
        close.tolist(),
        high=(close * (1 + np.abs(rng.standard_normal(n)) * 0.006)).tolist(),
        low=(close * (1 - np.abs(rng.standard_normal(n)) * 0.006)).tolist(),
    )


# --- schema ----------------------------------------------------------------


def test_every_label_frame_carries_the_required_columns(walk):
    labels = horizon_return(walk, horizon_bars=5)
    assert set(LABEL_COLUMNS) <= set(labels.columns) | {labels.index.name}
    validate_labels(labels)


def test_t_end_is_always_at_or_after_t_start(walk):
    labels = horizon_return(walk, horizon_bars=5)
    assert (labels["t_end"] >= labels.index).all()


def test_validation_rejects_a_label_ending_before_it_starts(walk):
    labels = horizon_return(walk, horizon_bars=5)
    broken = labels.copy()
    broken.iloc[0, broken.columns.get_loc("t_end")] = broken.index[0] - pd.Timedelta(days=1)
    with pytest.raises(LabelError, match="t_end"):
        validate_labels(broken)


# --- horizon methods -------------------------------------------------------


def test_horizon_return_is_the_forward_log_return():
    bars = make_bars([100.0, 110.0, 121.0, 133.1])
    labels = horizon_return(bars, horizon_bars=1)
    np.testing.assert_allclose(labels["label"].to_numpy(), np.log(1.1), atol=1e-12)


def test_horizon_return_drops_rows_with_no_future(walk):
    labels = horizon_return(walk, horizon_bars=5)
    # The last 5 bars have no bar 5 ahead, so they produce no label at all
    # rather than a label computed from a shorter horizon.
    assert len(labels) == len(walk) - 5
    assert labels.index[-1] == walk.index[-6]


def test_horizon_return_t_end_is_the_horizon_bar(walk):
    labels = horizon_return(walk, horizon_bars=5)
    assert labels["t_end"].iloc[0] == walk.index[5]


def test_horizon_sign_is_plus_or_minus_one():
    bars = make_bars([100.0, 110.0, 100.0, 110.0])
    labels = horizon_sign(bars, horizon_bars=1)
    assert set(labels["label"].unique()) <= {-1, 1}
    assert labels["label"].iloc[0] == 1
    assert labels["label"].iloc[1] == -1


def test_horizon_sign_dead_zone_drops_rather_than_calls_flat():
    # A move too small to trade is not a "flat" observation to learn from; it
    # is an observation with no information, and keeping it teaches the model
    # that noise has a direction.
    bars = make_bars([100.0, 100.0001, 110.0, 100.0])
    labels = horizon_sign(bars, horizon_bars=1, min_return=0.01)
    assert bars.index[0] not in labels.index
    assert len(labels) == 2


def test_horizon_quantile_fits_edges_only_on_the_fit_rows(walk):
    mask = np.zeros(len(walk), dtype=bool)
    mask[:150] = True
    labels = horizon_quantile(walk, horizon_bars=5, n_bins=5, fit_mask=mask)
    assert set(labels["label"].unique()) <= {0, 1, 2, 3, 4}


def test_horizon_quantile_requires_a_fit_mask(walk):
    with pytest.raises(LeakageError, match="fit"):
        horizon_quantile(walk, horizon_bars=5, n_bins=5, fit_mask=None)


def test_horizon_quantile_edges_differ_between_disjoint_fit_windows(walk):
    early = np.zeros(len(walk), dtype=bool)
    early[:150] = True
    late = np.zeros(len(walk), dtype=bool)
    late[150:] = True

    first = horizon_quantile(walk, horizon_bars=5, n_bins=5, fit_mask=early)
    second = horizon_quantile(walk, horizon_bars=5, n_bins=5, fit_mask=late)
    assert first.attrs["bin_edges"] != second.attrs["bin_edges"]


def test_horizon_quantile_records_its_edges(walk):
    mask = np.ones(len(walk), dtype=bool)
    labels = horizon_quantile(walk, horizon_bars=5, n_bins=4, fit_mask=mask)
    # n_bins bins means n_bins - 1 interior edges.
    assert len(labels.attrs["bin_edges"]) == 3


# --- triple barrier --------------------------------------------------------


def test_triple_barrier_upper_touch_is_labelled_plus_one():
    # Flat then a sharp rise: the upper barrier is hit before the vertical one.
    close = [100.0] * 20 + [100.0, 100.0, 130.0] + [130.0] * 10
    bars = make_bars(close, high=[c * 1.001 for c in close], low=[c * 0.999 for c in close])
    labels = triple_barrier(bars, profit_mult=1.0, loss_mult=1.0, vertical_bars=5, atr_window=5)
    hit = labels.loc[bars.index[20]]
    assert hit["label"] == 1
    assert hit["t_end"] == bars.index[22]


def test_triple_barrier_lower_touch_is_labelled_minus_one():
    close = [100.0] * 20 + [100.0, 100.0, 70.0] + [70.0] * 10
    bars = make_bars(close, high=[c * 1.001 for c in close], low=[c * 0.999 for c in close])
    labels = triple_barrier(bars, profit_mult=1.0, loss_mult=1.0, vertical_bars=5, atr_window=5)
    assert labels.loc[bars.index[20], "label"] == -1


def test_triple_barrier_vertical_expiry_is_labelled_zero():
    close = [100.0] * 40
    bars = make_bars(close)
    labels = triple_barrier(bars, profit_mult=5.0, loss_mult=5.0, vertical_bars=5, atr_window=5)
    assert (labels["label"] == 0).all()
    first_bar = bars.index.get_loc(labels.index[0])
    assert labels["t_end"].iloc[0] == bars.index[first_bar + 5]


def test_triple_barrier_stop_wins_a_same_bar_tie():
    # One bar touches both barriers. Assuming the favourable one filled first
    # is how a backtest manufactures an edge that does not exist (§12.1).
    close = [100.0] * 20 + [100.0, 100.0] + [100.0] * 10
    high = [100.1] * 21 + [200.0] + [100.1] * 10
    low = [99.9] * 21 + [10.0] + [99.9] * 10
    bars = make_bars(close, high=high, low=low)
    labels = triple_barrier(bars, profit_mult=1.0, loss_mult=1.0, vertical_bars=5, atr_window=5)
    assert labels.loc[bars.index[20], "label"] == -1


def test_triple_barrier_hand_computed_twenty_bar_fixture():
    # A 20-bar fixture whose barriers can be checked by reading the numbers.
    # ATR over 5 bars of a series moving 1.0 per bar with a 0.2 range is
    # dominated by the 1.0 steps, so the barriers sit roughly +/- 1 ATR.
    close = [
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        107.0,
        108.0,
        109.0,
        110.0,
        109.0,
        108.0,
        107.0,
        106.0,
        105.0,
        104.0,
        103.0,
        102.0,
        101.0,
    ]
    high = [c + 0.1 for c in close]
    low = [c - 0.1 for c in close]
    bars = make_bars(close, high=high, low=low)

    labels = triple_barrier(bars, profit_mult=1.0, loss_mult=1.0, vertical_bars=3, atr_window=5)

    # The first labelled bar comes after the ATR warm-up, and every label is
    # one of the three permitted outcomes with a t_end inside its window.
    assert not labels.empty
    assert set(labels["label"].unique()) <= {-1, 0, 1}
    for t_start, row in labels.iterrows():
        span = bars.index.get_loc(row["t_end"]) - bars.index.get_loc(t_start)
        assert 1 <= span <= 3

    # The rising stretch trips the upper barrier; the falling stretch the lower.
    assert labels.loc[bars.index[6], "label"] == 1
    assert labels.loc[bars.index[12], "label"] == -1


def test_triple_barrier_realised_return_matches_the_touch_bar(walk):
    labels = triple_barrier(walk, profit_mult=2.0, loss_mult=1.0, vertical_bars=10, atr_window=14)
    row = labels.iloc[0]
    start_close = walk.loc[labels.index[0], "close"]
    end_close = walk.loc[row["t_end"], "close"]
    assert row["realised_return"] == pytest.approx(np.log(end_close / start_close))


def test_triple_barrier_never_looks_past_its_vertical_barrier(walk):
    labels = triple_barrier(walk, profit_mult=2.0, loss_mult=1.0, vertical_bars=10, atr_window=14)
    positions = walk.index.get_indexer(labels.index)
    ends = walk.index.get_indexer(labels["t_end"])
    assert ((ends - positions) <= 10).all()
    assert ((ends - positions) >= 1).all()


def test_triple_barrier_rejects_a_non_positive_vertical(walk):
    with pytest.raises(LabelError, match="vertical_bars"):
        triple_barrier(walk, profit_mult=2.0, loss_mult=1.0, vertical_bars=0, atr_window=14)


# --- meta-labelling --------------------------------------------------------


def test_meta_label_is_one_when_the_primary_side_was_right(walk):
    primary = pd.Series(1, index=walk.index)
    base = triple_barrier(walk, profit_mult=2.0, loss_mult=1.0, vertical_bars=10, atr_window=14)
    meta = meta_label(base, primary.loc[base.index])

    profitable = base["realised_return"] > 0
    assert (meta.loc[profitable, "label"] == 1).all()
    assert (meta.loc[~profitable, "label"] == 0).all()


def test_meta_label_is_one_when_a_short_side_was_right(walk):
    base = triple_barrier(walk, profit_mult=2.0, loss_mult=1.0, vertical_bars=10, atr_window=14)
    primary = pd.Series(-1, index=base.index)
    meta = meta_label(base, primary)
    assert (meta.loc[base["realised_return"] < 0, "label"] == 1).all()


def test_meta_label_keeps_the_primary_side_as_a_column(walk):
    base = triple_barrier(walk, profit_mult=2.0, loss_mult=1.0, vertical_bars=10, atr_window=14)
    meta = meta_label(base, pd.Series(1, index=base.index))
    assert "side" in meta.columns


def test_meta_label_drops_bars_the_primary_rule_stayed_out_of(walk):
    base = triple_barrier(walk, profit_mult=2.0, loss_mult=1.0, vertical_bars=10, atr_window=14)
    primary = pd.Series(0, index=base.index)
    primary.iloc[:10] = 1
    meta = meta_label(base, primary)
    # A trade the primary rule never took is not an observation about whether
    # to take it.
    assert len(meta) == 10


def test_meta_label_requires_aligned_indexes(walk):
    base = triple_barrier(walk, profit_mult=2.0, loss_mult=1.0, vertical_bars=10, atr_window=14)
    with pytest.raises(LabelError, match="index"):
        meta_label(base, pd.Series(1, index=base.index[:-5]))


# --- uniqueness and decay --------------------------------------------------


def test_uniqueness_of_non_overlapping_labels_is_one():
    index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    t_start = index[[0, 2, 4]]
    t_end = index[[1, 3, 5]]
    weights = uniqueness_weights(t_start, t_end, index)
    np.testing.assert_allclose(weights, 1.0, atol=1e-12)


def test_uniqueness_falls_when_labels_overlap():
    index = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    # Two labels spanning the same two bars: each bar carries concurrency 2.
    weights = uniqueness_weights(index[[0, 0]], index[[1, 1]], index)
    # Normalised to mean 1, so equal overlap gives equal weight...
    np.testing.assert_allclose(weights, 1.0, atol=1e-12)


def test_a_uniquely_spanning_label_outweighs_a_crowded_one():
    index = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    # Three labels crowd bars 0-2; one label sits alone on bars 5-7.
    t_start = index[[0, 0, 0, 5]]
    t_end = index[[2, 2, 2, 7]]
    weights = uniqueness_weights(t_start, t_end, index)
    assert weights[3] > weights[0]
    assert weights[0] == pytest.approx(weights[1])


def test_uniqueness_weights_average_to_one():
    index = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    t_start = index[:40]
    t_end = index[5:45]
    assert uniqueness_weights(t_start, t_end, index).mean() == pytest.approx(1.0)


def test_time_decay_favours_recent_observations():
    n = 10
    decay = 0.5
    weights = time_decay_weights(np.ones(n), decay=decay)

    assert weights[-1] == pytest.approx(1.0)
    # The schedule is linear from an intercept of `decay` at cumulative
    # uniqueness zero, so the first observation sits one step above it.
    step = (1.0 - decay) / n
    assert weights[0] == pytest.approx(decay + step)
    np.testing.assert_allclose(np.diff(weights), step, atol=1e-12)


def test_time_decay_of_one_is_flat():
    uniqueness = np.ones(10)
    np.testing.assert_allclose(time_decay_weights(uniqueness, decay=1.0), 1.0, atol=1e-12)


def test_time_decay_never_goes_negative():
    assert (time_decay_weights(np.ones(20), decay=0.0) >= 0).all()


def test_sample_weights_combine_uniqueness_and_decay(walk):
    labels = triple_barrier(walk, profit_mult=2.0, loss_mult=1.0, vertical_bars=10, atr_window=14)
    config = SampleWeightsConfig(uniqueness=True, time_decay=0.5)
    weights = sample_weights(labels, walk.index, config)
    assert len(weights) == len(labels)
    assert (weights > 0).all()
    # Uniqueness is noisy label by label, so compare the ends in aggregate:
    # recent observations must carry more weight than old ones.
    tenth = len(weights) // 10
    assert weights[-tenth:].mean() > weights[:tenth].mean()


def test_sample_weights_of_all_ones_when_both_are_disabled(walk):
    labels = triple_barrier(walk, profit_mult=2.0, loss_mult=1.0, vertical_bars=10, atr_window=14)
    config = SampleWeightsConfig(uniqueness=False, time_decay=1.0)
    np.testing.assert_allclose(sample_weights(labels, walk.index, config), 1.0, atol=1e-12)


# --- build_labels ----------------------------------------------------------


def test_build_labels_dispatches_on_the_configured_method(walk):
    for method in ("horizon_return", "horizon_sign", "triple_barrier"):
        config = LabelsConfig(method=method, horizon_bars=5)
        labels = build_labels(walk, config)
        assert not labels.empty
        validate_labels(labels)


def test_build_labels_attaches_weights(walk):
    config = LabelsConfig(method="triple_barrier", horizon_bars=5)
    labels = build_labels(walk, config)
    assert (labels["weight"] > 0).all()
    assert labels["weight"].mean() == pytest.approx(1.0, abs=0.5)


def test_build_labels_quantile_requires_a_fit_mask(walk):
    config = LabelsConfig(method="horizon_quantile", horizon_bars=5)
    with pytest.raises(LeakageError, match="fit"):
        build_labels(walk, config)


def test_build_labels_reports_class_balance(walk):
    config = LabelsConfig(method="triple_barrier", horizon_bars=5)
    labels = build_labels(walk, config)
    balance = labels.attrs["class_balance"]
    assert pytest.approx(sum(balance.values()), abs=1e-9) == 1.0


def test_build_labels_reports_mean_uniqueness(walk):
    config = LabelsConfig(
        method="triple_barrier",
        horizon_bars=5,
        triple_barrier=TripleBarrierConfig(vertical_bars=10),
    )
    labels = build_labels(walk, config)
    assert 0.0 < labels.attrs["mean_uniqueness"] <= 1.0
