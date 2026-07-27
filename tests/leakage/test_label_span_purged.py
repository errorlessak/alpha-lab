"""No training label may overlap its test window (§12.3.2).

This is invariant I4 made executable, and it is the test that decides whether
any out-of-sample number this system reports means anything.

The failure it guards against is quiet. With a ten-bar label horizon, the ten
training samples on each side of every test block have outcomes determined by
bars inside that block. Nothing in the code looks like a look-ahead; the model
simply learns the answer to part of its own exam. Reported accuracy rises a
little, reported Sharpe rises a lot, and neither survives contact with reality.

Hypothesis drives the horizons and geometries, because the off-by-one that
matters is rarely the one anybody chose to write a test for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from alphacore.cv import CPCV, PurgedKFold, Split, WalkForward

SLOW = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def make_labels(n: int, horizon: int) -> pd.DataFrame:
    """A label frame whose spans overlap by `horizon - 1` bars."""
    index = pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC")
    usable = n - horizon
    return pd.DataFrame(
        {
            "t_end": index[horizon : horizon + usable],
            "label": np.zeros(usable, dtype=int),
            "realised_return": np.zeros(usable),
            "weight": np.ones(usable),
        },
        index=pd.DatetimeIndex(index[:usable], name="ts"),
    )


def assert_purged(split: Split, labels: pd.DataFrame) -> None:
    """Assert no training span intersects the test window."""
    if split.train_idx.size == 0:
        return
    test_start = labels.index[split.test_idx].min()
    test_end = max(
        labels.index[split.test_idx].max(),
        pd.DatetimeIndex(labels["t_end"])[split.test_idx].max(),
    )
    train_start = labels.index[split.train_idx]
    train_end = pd.DatetimeIndex(labels["t_end"])[split.train_idx]

    overlapping = (train_end >= test_start) & (train_start <= test_end)
    assert not overlapping.any(), (
        f"fold {split.fold}: {int(overlapping.sum())} training labels overlap "
        f"[{test_start}, {test_end}] (§4, I4)"
    )


def all_schemes(embargo: float = 0.02):
    """The three schemes, configured identically enough to compare."""
    return (
        PurgedKFold(n_splits=5, embargo_pct=embargo),
        WalkForward(train_bars=200, test_bars=100, step_bars=100, embargo_pct=embargo),
        WalkForward(
            train_bars=200, test_bars=100, step_bars=100, anchored=True, embargo_pct=embargo
        ),
        CPCV(n_groups=6, n_test_groups=2, embargo_pct=embargo),
    )


def test_label_span_purged():
    """Every scheme purges every fold, on a realistic ten-bar horizon (§4, I4).

    The named test §12.3 requires. The hypothesis-driven variants below widen
    it across horizons and geometries.
    """
    labels = make_labels(600, horizon=10)
    for scheme in all_schemes():
        splits = scheme.split(labels)
        assert splits, f"{scheme!r} produced no folds"
        for split in splits:
            assert_purged(split, labels)
        assert sum(s.purged for s in splits) > 0, (
            f"{scheme!r} purged nothing on a ten-bar horizon, which cannot be right"
        )


# --- the three schemes, over randomised geometry ---------------------------


@SLOW
@given(
    horizon=st.integers(min_value=1, max_value=30),
    n_splits=st.integers(min_value=2, max_value=8),
    embargo=st.floats(min_value=0.0, max_value=0.1),
)
def test_label_span_purged_kfold(horizon: int, n_splits: int, embargo: float):
    labels = make_labels(600, horizon)
    for split in PurgedKFold(n_splits=n_splits, embargo_pct=embargo).split(labels):
        assert_purged(split, labels)


@SLOW
@given(
    horizon=st.integers(min_value=1, max_value=30),
    train_bars=st.integers(min_value=50, max_value=250),
    test_bars=st.integers(min_value=20, max_value=120),
    anchored=st.booleans(),
    embargo=st.floats(min_value=0.0, max_value=0.1),
)
def test_label_span_purged_walkforward(
    horizon: int, train_bars: int, test_bars: int, anchored: bool, embargo: float
):
    labels = make_labels(600, horizon)
    scheme = WalkForward(
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=test_bars,
        anchored=anchored,
        embargo_pct=embargo,
    )
    for split in scheme.split(labels):
        assert_purged(split, labels)


@SLOW
@given(
    horizon=st.integers(min_value=1, max_value=30),
    n_groups=st.integers(min_value=3, max_value=7),
    embargo=st.floats(min_value=0.0, max_value=0.1),
)
def test_label_span_purged_cpcv(horizon: int, n_groups: int, embargo: float):
    labels = make_labels(600, horizon)
    for split in CPCV(n_groups=n_groups, n_test_groups=2, embargo_pct=embargo).split(labels):
        assert_purged(split, labels)


# --- the embargo is present, not merely claimed ----------------------------


def test_embargo_leaves_a_measurable_gap_after_the_test_block():
    labels = make_labels(600, horizon=1)
    embargoed = PurgedKFold(n_splits=5, embargo_pct=0.05).split(labels)
    unembargoed = PurgedKFold(n_splits=5, embargo_pct=0.0).split(labels)

    assert sum(s.embargoed for s in embargoed) > 0
    assert sum(s.embargoed for s in unembargoed) == 0
    for split in embargoed:
        assert split.embargoed == split.embargoed_idx.size


def test_embargoed_samples_all_start_after_the_test_window():
    labels = make_labels(600, horizon=5)
    for split in PurgedKFold(n_splits=4, embargo_pct=0.05).split(labels):
        if split.embargoed_idx.size == 0:
            continue
        test_end = pd.DatetimeIndex(labels["t_end"])[split.test_idx].max()
        assert (labels.index[split.embargoed_idx] > test_end).all()


def test_purged_and_embargoed_partition_the_candidate_training_set():
    labels = make_labels(600, horizon=10)
    for split in PurgedKFold(n_splits=5, embargo_pct=0.03).split(labels):
        recovered = np.concatenate((split.train_idx, split.purged_idx, split.embargoed_idx))
        expected = np.setdiff1d(np.arange(len(labels)), split.test_idx)
        np.testing.assert_array_equal(np.sort(recovered), expected)


# --- the control -----------------------------------------------------------


def test_the_purge_check_catches_an_unpurged_split():
    """Without this, the tests above could be asserting nothing.

    An unpurged split is exactly what a naive implementation produces: train on
    everything that is not in the test block. On a ten-bar horizon that leaves
    ten contaminated samples on each boundary.
    """
    labels = make_labels(600, horizon=10)
    n = len(labels)
    test_idx = np.arange(200, 300, dtype=np.int64)
    unpurged = Split(
        fold=0,
        train_idx=np.setdiff1d(np.arange(n, dtype=np.int64), test_idx),
        test_idx=test_idx,
        train_span=(labels.index[0], labels.index[-1]),
        test_span=(labels.index[200], labels.index[299]),
        purged=0,
        embargoed=0,
        purged_idx=np.empty(0, dtype=np.int64),
        embargoed_idx=np.empty(0, dtype=np.int64),
    )

    with pytest.raises(AssertionError, match="overlap"):
        assert_purged(unpurged, labels)


def test_a_longer_horizon_purges_more():
    """Purging must scale with the label span, not be a fixed fudge."""
    counts = []
    for horizon in (1, 10, 30):
        labels = make_labels(600, horizon)
        splits = PurgedKFold(n_splits=5, embargo_pct=0.0).split(labels)
        counts.append(sum(s.purged for s in splits))
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]
