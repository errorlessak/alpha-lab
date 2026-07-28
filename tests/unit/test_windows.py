"""Fold-aware windowing for sequence models (§10.3).

The window is where a sequence model can leak without anyone noticing. A fold's
training rows arrive already purged, which means the frame handed to `fit` has a
*hole* in it where the test block used to be. Stacking a rolling window over
that frame naively produces windows whose first half sits before the test period
and whose second half sits after it — a model reading bars from both sides of
the block it is about to be scored on.

These tests pin the rule that prevents it: a window is built only from rows that
are consecutive **on the original bar grid**, never across a gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacore.errors import ModelError
from alphacore.models.windows import build_windows, contiguous_runs, grid_positions


def _values(n, n_features=2):
    """Rows whose every cell encodes its own row number, so mixing shows up."""
    base = np.arange(n, dtype=float).reshape(-1, 1)
    return np.repeat(base, n_features, axis=1)


def test_contiguous_runs_returns_one_run_for_an_unbroken_block():
    assert contiguous_runs(np.arange(5, dtype=np.int64)) == ((0, 5),)


def test_contiguous_runs_splits_where_the_grid_skips():
    positions = np.array([0, 1, 2, 9, 10], dtype=np.int64)
    assert contiguous_runs(positions) == ((0, 3), (3, 5))


def test_contiguous_runs_of_nothing_is_nothing():
    assert contiguous_runs(np.empty(0, dtype=np.int64)) == ()


def test_a_window_never_spans_a_purge_gap():
    # Rows 0..4 and 20..24 survive; 5..19 were purged out of the fold.
    positions = np.array([*range(5), *range(20, 25)], dtype=np.int64)
    windows, rows = build_windows(_values(10), positions, seq_len=3, pad=False)

    for window, row in zip(windows, rows, strict=True):
        source_rows = window[:, 0].astype(int)
        assert positions[source_rows[-1]] == positions[row]
        assert np.all(np.diff(positions[source_rows]) == 1)


def test_unpadded_windows_appear_only_where_the_history_is_complete():
    positions = np.array([*range(5), *range(20, 25)], dtype=np.int64)
    _, rows = build_windows(_values(10), positions, seq_len=3, pad=False)

    # Two runs of five rows each, three of which have two predecessors.
    assert rows.tolist() == [2, 3, 4, 7, 8, 9]


def test_a_run_shorter_than_the_window_contributes_nothing_unpadded():
    positions = np.array([0, 1, 50, 51, 52, 53], dtype=np.int64)
    _, rows = build_windows(_values(6), positions, seq_len=4, pad=False)

    assert rows.tolist() == [5]


def test_padded_windows_cover_every_row():
    positions = np.array([*range(3), *range(30, 33)], dtype=np.int64)
    windows, rows = build_windows(_values(6), positions, seq_len=4, pad=True)

    assert rows.tolist() == [0, 1, 2, 3, 4, 5]
    assert windows.shape == (6, 4, 2)


def test_padding_repeats_the_first_bar_of_its_own_run():
    positions = np.array([*range(3), *range(30, 33)], dtype=np.int64)
    windows, rows = build_windows(_values(6), positions, seq_len=4, pad=True)

    # Row 3 opens the second run; its padding must be row 3 itself, never row 2,
    # which sits on the far side of the gap.
    first_of_second_run = windows[rows.tolist().index(3)]
    assert first_of_second_run[:, 0].tolist() == [3.0, 3.0, 3.0, 3.0]


def test_the_window_ends_on_the_row_it_describes():
    positions = np.arange(6, dtype=np.int64)
    windows, rows = build_windows(_values(6), positions, seq_len=3, pad=True)

    for window, row in zip(windows, rows, strict=True):
        assert window[-1, 0] == float(row)


def test_the_window_is_ordered_oldest_first():
    positions = np.arange(6, dtype=np.int64)
    windows, _ = build_windows(_values(6), positions, seq_len=3, pad=False)

    for window in windows:
        assert np.all(np.diff(window[:, 0]) >= 0)


def test_a_sequence_length_below_one_is_rejected():
    with pytest.raises(ModelError, match="seq_len"):
        build_windows(_values(4), np.arange(4, dtype=np.int64), seq_len=0, pad=True)


def test_positions_that_do_not_increase_are_rejected():
    positions = np.array([0, 2, 1, 3], dtype=np.int64)
    with pytest.raises(ModelError, match="increasing"):
        build_windows(_values(4), positions, seq_len=2, pad=True)


def test_values_and_positions_must_describe_the_same_rows():
    with pytest.raises(ModelError, match="rows"):
        build_windows(_values(4), np.arange(5, dtype=np.int64), seq_len=2, pad=True)


def test_grid_positions_locates_each_timestamp_on_the_full_bar_grid():
    grid = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    subset = grid[[0, 1, 7, 8]]

    assert grid_positions(subset, grid).tolist() == [0, 1, 7, 8]


def test_grid_positions_rejects_a_timestamp_that_is_not_on_the_grid():
    grid = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    stray = pd.DatetimeIndex(["2024-01-02T12:00:00Z"])

    with pytest.raises(ModelError, match="grid"):
        grid_positions(stray, grid)
