"""A sequence window must not reach across a purged block (§10.3, I4).

Phase 4 purges label spans out of every training fold, so `fit` receives a frame
with a hole in it. A rolling window laid over that frame positionally stitches
the two sides of the hole together, and the model reads bars from after the test
period while training on bars from before it. The fold is still called purged.
The leak is invisible in the fold metadata, because the fold metadata is
correct — it is the tensor that is wrong.

Like every canary in this suite, this one carries its control: `_naive_windows`
is the obvious implementation, and the same assertion run against it must fail.
A test that only ever sees the fixed code proves the fix compiles, not that the
check works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacore.cv import PurgedKFold
from alphacore.models.windows import build_windows, grid_positions

SEQ_LEN = 8
N_BARS = 400
HORIZON = 5


@pytest.fixture
def frame():
    """Features and labels on a clean daily grid, with spanning labels."""
    index = pd.date_range("2022-01-03", periods=N_BARS, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
    features = pd.DataFrame(
        {"f0": rng.normal(size=N_BARS), "f1": np.arange(N_BARS, dtype=float)},
        index=index,
    )
    labels = pd.DataFrame(
        {
            "label": rng.integers(0, 2, size=N_BARS),
            "t_end": index[np.minimum(np.arange(N_BARS) + HORIZON, N_BARS - 1)],
        },
        index=index,
    )
    return features, labels


@pytest.fixture
def middle_fold(frame):
    """One fold whose test block sits in the middle of the training rows."""
    _, labels = frame
    folds = PurgedKFold(n_splits=4, embargo_pct=0.01).split(labels)
    splits = [s for s in folds if s.fold == 1]
    assert splits, "expected a fold numbered 1 from a 4-fold purged split"
    return splits[0]


def _naive_windows(matrix, seq_len):
    """The wrong implementation: stack windows by row position, ignoring gaps.

    Args:
        matrix: The fold's training rows, purged rows already removed.
        seq_len: Window length.

    Returns:
        `(n, seq_len, n_features)` windows, some of which straddle the hole the
        purge left behind.
    """
    view = np.lib.stride_tricks.sliding_window_view(matrix, seq_len, axis=0)
    return np.transpose(view, (0, 2, 1))


def _spanned_positions(windows, positions, rows):
    """Grid positions each window drew its bars from.

    Args:
        windows: Windows whose last feature column carries the row's grid
            position, so the source of every bar can be recovered.
        positions: Grid positions of the fold's rows.
        rows: Row each window describes.

    Returns:
        An `(n_windows, seq_len)` array of grid positions.
    """
    assert len(windows) == len(rows)
    return windows[:, :, 1].astype(np.int64) if positions.size else np.empty((0, 0), np.int64)


def test_window_purge(frame, middle_fold):
    """No training window may contain a bar from inside the test block."""
    features, _ = frame
    train = features.iloc[middle_fold.train_idx]
    positions = grid_positions(train.index, features.index)
    test_positions = set(middle_fold.test_idx.tolist())

    windows, rows = build_windows(train.to_numpy(), positions, seq_len=SEQ_LEN, pad=False)

    assert len(windows) > 0
    spanned = _spanned_positions(windows, positions, rows)
    assert not (set(spanned.ravel().tolist()) & test_positions)
    assert np.all(np.diff(spanned, axis=1) == 1)


def test_the_naive_windower_this_guards_against_really_does_straddle(frame, middle_fold):
    """The control: the obvious implementation fails the assertion above."""
    features, _ = frame
    train = features.iloc[middle_fold.train_idx]
    test_positions = set(middle_fold.test_idx.tolist())

    windows = _naive_windows(train.to_numpy(), SEQ_LEN)
    spanned = windows[:, :, 1].astype(np.int64)

    # It reads no test bar — those rows are gone — but it welds the two sides of
    # the gap into one window, which is the same information path.
    assert not (set(spanned.ravel().tolist()) & test_positions)
    assert np.any(np.diff(spanned, axis=1) != 1)


def test_padded_windows_are_purged_too(frame, middle_fold):
    """Padding for prediction must not reach across the gap either."""
    features, _ = frame
    train = features.iloc[middle_fold.train_idx]
    positions = grid_positions(train.index, features.index)

    windows, rows = build_windows(train.to_numpy(), positions, seq_len=SEQ_LEN, pad=True)

    assert len(rows) == len(train)
    spanned = windows[:, :, 1].astype(np.int64)
    steps = np.diff(spanned, axis=1)
    # Every step is either the next bar or a repeat of the run's first bar.
    assert np.all((steps == 1) | (steps == 0))
    for window in spanned:
        real = window[window != window[0]]
        assert real.size == 0 or np.all(np.diff(real) == 1)


def test_a_window_never_reaches_past_the_row_it_scores(frame, middle_fold):
    """The last bar of a window is the row itself; nothing later gets in."""
    features, _ = frame
    train = features.iloc[middle_fold.train_idx]
    positions = grid_positions(train.index, features.index)

    windows, rows = build_windows(train.to_numpy(), positions, seq_len=SEQ_LEN, pad=True)

    spanned = windows[:, :, 1].astype(np.int64)
    for window, row in zip(spanned, rows, strict=True):
        assert window.max() == positions[row]
