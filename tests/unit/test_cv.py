"""Purged cross-validation schemes (§9)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from alphacore.config.schema import CPCVConfig, CVConfig, WalkForwardConfig
from alphacore.cv import CPCV, PurgedKFold, Split, WalkForward, make_splits, splits_to_json
from alphacore.cv.splits import splits_from_json
from alphacore.errors import CVError

N_BARS = 600
HORIZON = 10


def make_labels(n: int = N_BARS, horizon: int = HORIZON) -> pd.DataFrame:
    """A label frame with overlapping spans, which is the whole difficulty."""
    index = pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC")
    usable = n - horizon
    return pd.DataFrame(
        {
            "t_end": index[horizon : horizon + usable],
            "label": np.tile([1, -1], usable // 2 + 1)[:usable],
            "realised_return": np.zeros(usable),
            "weight": np.ones(usable),
        },
        index=pd.DatetimeIndex(index[:usable], name="ts"),
    )


@pytest.fixture
def labels() -> pd.DataFrame:
    return make_labels()


def assert_no_overlap(split: Split, labels: pd.DataFrame) -> None:
    """No training label's span may intersect the test window (§4, I4)."""
    if split.train_idx.size == 0:
        return
    test_start = labels.index[split.test_idx].min()
    test_end = labels["t_end"].iloc[split.test_idx].max()
    train_start = labels.index[split.train_idx]
    train_end = labels["t_end"].iloc[split.train_idx]
    overlapping = (train_end >= test_start) & (train_start <= test_end)
    assert not overlapping.any(), (
        f"fold {split.fold}: {int(overlapping.sum())} training labels overlap the test window"
    )


# --- Split -----------------------------------------------------------------


def test_split_reports_its_spans(labels):
    split = WalkForward(train_bars=200, test_bars=100, step_bars=100).split(labels)[0]
    assert split.train_span[0] == labels.index[split.train_idx].min()
    assert split.test_span[1] == labels.index[split.test_idx].max()


def test_split_is_frozen(labels):
    split = WalkForward(train_bars=200, test_bars=100, step_bars=100).split(labels)[0]
    with pytest.raises(AttributeError):
        split.fold = 99


def test_split_train_and_test_never_intersect(labels):
    for split in WalkForward(train_bars=200, test_bars=100, step_bars=100).split(labels):
        assert not set(split.train_idx.tolist()) & set(split.test_idx.tolist())


def test_splits_round_trip_through_json(labels):
    splits = WalkForward(train_bars=200, test_bars=100, step_bars=100).split(labels)
    restored = splits_from_json(json.loads(splits_to_json(splits)))

    assert len(restored) == len(splits)
    for before, after in zip(splits, restored, strict=True):
        assert after.fold == before.fold
        np.testing.assert_array_equal(after.train_idx, before.train_idx)
        np.testing.assert_array_equal(after.test_idx, before.test_idx)
        np.testing.assert_array_equal(after.purged_idx, before.purged_idx)
        np.testing.assert_array_equal(after.embargoed_idx, before.embargoed_idx)
        assert after.train_span == before.train_span
        assert after.test_span == before.test_span
        assert after.purged == before.purged
        assert after.embargoed == before.embargoed


def test_serialised_splits_are_canonical_json(labels):
    splits = WalkForward(train_bars=200, test_bars=100, step_bars=100).split(labels)
    # Byte-identical across calls, so folds.json is comparable between runs.
    assert splits_to_json(splits) == splits_to_json(splits)


def test_serialised_splits_carry_the_gaps_for_drawing(labels):
    splits = WalkForward(train_bars=200, test_bars=100, step_bars=100).split(labels)
    payload = json.loads(splits_to_json(splits))
    assert "purged_idx" in payload[0]
    assert "embargoed_idx" in payload[0]


# --- WalkForward -----------------------------------------------------------


def test_walkforward_folds_are_chronological(labels):
    splits = WalkForward(train_bars=200, test_bars=100, step_bars=100).split(labels)
    starts = [s.test_span[0] for s in splits]
    assert starts == sorted(starts)
    assert [s.fold for s in splits] == list(range(len(splits)))


def test_walkforward_test_blocks_do_not_overlap(labels):
    splits = WalkForward(train_bars=200, test_bars=100, step_bars=100).split(labels)
    seen: set[int] = set()
    for split in splits:
        assert not seen & set(split.test_idx.tolist())
        seen |= set(split.test_idx.tolist())


def test_rolling_walkforward_keeps_the_train_window_bounded(labels):
    splits = WalkForward(train_bars=200, test_bars=100, step_bars=100, anchored=False).split(labels)
    assert all(split.train_idx.size <= 200 for split in splits)


def test_anchored_walkforward_grows_the_train_window(labels):
    splits = WalkForward(train_bars=200, test_bars=100, step_bars=100, anchored=True).split(labels)
    sizes = [split.train_idx.size for split in splits]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_walkforward_training_always_precedes_testing(labels):
    for split in WalkForward(train_bars=200, test_bars=100, step_bars=100).split(labels):
        assert split.train_span[1] < split.test_span[0]


def test_walkforward_purges_and_embargoes(labels):
    splits = WalkForward(train_bars=200, test_bars=100, step_bars=100, embargo_pct=0.02).split(
        labels
    )
    for split in splits:
        assert_no_overlap(split, labels)
    # Purging removes the overlapping tail of the training window, and the
    # embargo removes bars after the test block — which for a walk-forward fold
    # only bites on the anchored variant or the next fold's training set.
    assert sum(s.purged for s in splits) > 0


def test_walkforward_skips_folds_with_too_little_training_data(labels):
    scheme = WalkForward(train_bars=200, test_bars=100, step_bars=100, min_train_bars=10_000)
    with pytest.raises(CVError, match="min_train_bars"):
        scheme.split(labels)


def test_walkforward_rejects_a_series_shorter_than_one_fold():
    short = make_labels(n=50)
    with pytest.raises(CVError, match="too short"):
        WalkForward(train_bars=200, test_bars=100, step_bars=100).split(short)


# --- PurgedKFold -----------------------------------------------------------


def test_purged_kfold_produces_the_requested_number_of_folds(labels):
    assert len(PurgedKFold(n_splits=5).split(labels)) == 5


def test_purged_kfold_test_blocks_are_contiguous_and_disjoint(labels):
    splits = PurgedKFold(n_splits=5).split(labels)
    seen: set[int] = set()
    for split in splits:
        positions = split.test_idx
        assert (np.diff(positions) == 1).all()
        assert not seen & set(positions.tolist())
        seen |= set(positions.tolist())
    assert len(seen) == len(labels)


def test_purged_kfold_purges_overlapping_training_labels(labels):
    splits = PurgedKFold(n_splits=5, embargo_pct=0.01).split(labels)
    for split in splits:
        assert_no_overlap(split, labels)
    assert sum(s.purged for s in splits) > 0


def test_purged_kfold_embargo_removes_samples_after_the_test_block(labels):
    without = PurgedKFold(n_splits=5, embargo_pct=0.0).split(labels)
    with_embargo = PurgedKFold(n_splits=5, embargo_pct=0.05).split(labels)
    assert sum(s.embargoed for s in with_embargo) > 0
    assert sum(s.train_idx.size for s in with_embargo) < sum(s.train_idx.size for s in without)


def test_purged_kfold_rejects_too_few_splits(labels):
    with pytest.raises(CVError, match="n_splits"):
        PurgedKFold(n_splits=1).split(labels)


# --- CPCV ------------------------------------------------------------------


def test_cpcv_produces_every_combination(labels):
    splits = CPCV(n_groups=6, n_test_groups=2).split(labels)
    # C(6, 2) = 15
    assert len(splits) == 15


def test_cpcv_test_sets_are_unions_of_whole_groups(labels):
    splits = CPCV(n_groups=6, n_test_groups=2).split(labels)
    sizes = {split.test_idx.size for split in splits}
    # Two groups per split; group sizes differ by at most one from rounding.
    assert max(sizes) - min(sizes) <= 2


def test_cpcv_purges_every_combination(labels):
    for split in CPCV(n_groups=6, n_test_groups=2, embargo_pct=0.01).split(labels):
        assert_no_overlap(split, labels)


def test_cpcv_covers_each_group_the_same_number_of_times(labels):
    splits = CPCV(n_groups=6, n_test_groups=2).split(labels)
    counts = np.zeros(len(labels), dtype=int)
    for split in splits:
        counts[split.test_idx] += 1
    # Each bar is tested C(5, 1) = 5 times, which is what gives CPCV its
    # multiple backtest paths for PBO (§11.3).
    assert set(counts.tolist()) == {5}


def test_cpcv_rejects_holding_out_every_group(labels):
    with pytest.raises(CVError, match="n_test_groups"):
        CPCV(n_groups=4, n_test_groups=4).split(labels)


# --- config dispatch -------------------------------------------------------


def test_make_splits_dispatches_on_the_configured_scheme(labels):
    for scheme, expected in (("walkforward", 3), ("purged_kfold", 5), ("cpcv", 15)):
        config = CVConfig(
            scheme=scheme,
            n_splits=5,
            walkforward=WalkForwardConfig(train_bars=200, test_bars=100, step_bars=100),
            cpcv=CPCVConfig(n_groups=6, n_test_groups=2),
            min_train_bars=50,
        )
        splits = make_splits(labels, config)
        assert len(splits) >= 1
        if scheme != "walkforward":
            assert len(splits) == expected


def test_make_splits_is_deterministic(labels):
    config = CVConfig(scheme="purged_kfold", n_splits=5, min_train_bars=50)
    first = splits_to_json(make_splits(labels, config))
    second = splits_to_json(make_splits(labels, config))
    assert first == second


def test_every_scheme_reports_purge_and_embargo_counts(labels):
    for config in (
        CVConfig(
            scheme="walkforward",
            walkforward=WalkForwardConfig(train_bars=200, test_bars=100, step_bars=100),
            min_train_bars=50,
        ),
        CVConfig(scheme="purged_kfold", n_splits=5, min_train_bars=50),
        CVConfig(scheme="cpcv", cpcv=CPCVConfig(n_groups=6, n_test_groups=2), min_train_bars=50),
    ):
        for split in make_splits(labels, config):
            assert split.purged == split.purged_idx.size
            assert split.embargoed == split.embargoed_idx.size


def test_an_empty_label_frame_is_rejected():
    empty = make_labels(n=30).iloc[:0]
    with pytest.raises(CVError, match="empty"):
        PurgedKFold(n_splits=3).split(empty)
