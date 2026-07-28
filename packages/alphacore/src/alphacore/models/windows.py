"""Fold-aware windowing for sequence models (§10.3).

A sequence model reads `(n_samples, seq_len, n_features)` tensors, and the
tensor is where a purged fold quietly stops being purged. `fit` receives the
fold's training rows *after* purge and embargo, so the frame it sees has a hole
in it exactly where the test block was removed. Rolling a window over that frame
positionally produces windows whose early bars sit before the test period and
whose late bars sit after it — the model reads across the block it is about to
be scored on, which is the leak §9 spent the whole cross-validation layer
preventing.

The rule enforced here is the same purge logic applied one level down: **a
window is built only from rows that are consecutive on the original bar grid.**
Rows are located on that grid by timestamp (`grid_positions`), the grid is cut
into maximal unbroken runs (`contiguous_runs`), and every window lives inside
one run.

Two window modes, because training and prediction want different things:

* `pad=False` — used for training. A row contributes a window only if it has a
  full `seq_len` of real history behind it *in its own run*. Rows near a fold
  boundary simply do not become training samples. Dropping them costs a handful
  of rows and fabricates nothing.
* `pad=True` — used for prediction, where the caller needs one score per row and
  a missing row would misalign every downstream metric. The short history at the
  head of a run is filled by repeating that run's first bar. The padding is
  always a bar the model may legitimately see: it is inside the run, at or
  before the row being scored, so it can carry neither future information nor
  information from the far side of a purge gap.

Nothing here imports torch. The leak this module prevents is a property of the
indexing, not of the network, and `tests/leakage/test_window_purge.py` has to be
able to prove it on a machine with no deep-learning stack installed.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from alphacore.errors import ModelError

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["build_windows", "contiguous_runs", "grid_positions"]

_MATRIX_NDIM: Final[int] = 2


def grid_positions(index: pd.Index, grid: pd.Index) -> NDArray[np.int64]:
    """Locate each row of `index` on the full bar grid.

    Positional distance on the grid — not elapsed time — is what "consecutive"
    means for a bar series. A weekend is not a gap: Friday and Monday are
    adjacent bars in a daily equity index. A purged block *is* a gap, because
    those bars exist in the grid and were removed from the fold.

    Args:
        index: The rows to locate, typically one fold's training index.
        grid: The complete feature index the fold was cut from.

    Returns:
        Positions into `grid`, one per row of `index`.

    Raises:
        ModelError: If any timestamp is not on the grid, which would mean the
            caller passed an index the windows cannot be reasoned about.
    """
    positions = np.asarray(pd.Index(grid).get_indexer(pd.Index(index)), dtype=np.int64)
    missing = np.flatnonzero(positions < 0)
    if missing.size:
        first = pd.Index(index)[int(missing[0])]
        msg = (
            f"{missing.size} timestamp(s) are not on the bar grid, the first being {first}. "
            "Sequence windows are cut on grid positions, so every row must come from the "
            "same feature index the folds were built on."
        )
        raise ModelError(msg)
    return positions


def contiguous_runs(positions: NDArray[np.int64]) -> tuple[tuple[int, int], ...]:
    """Split grid positions into maximal runs of consecutive bars.

    Args:
        positions: Strictly increasing grid positions.

    Returns:
        `(start, stop)` half-open slices **into `positions`**, one per run. A
        fold with one purged block in the middle yields two runs.

    Raises:
        ModelError: If the positions do not strictly increase.
    """
    if positions.size == 0:
        return ()
    steps = np.diff(positions)
    if np.any(steps <= 0):
        msg = (
            "Window positions must be strictly increasing; the rows arrived out of order "
            "or contain duplicates."
        )
        raise ModelError(msg)
    breaks = np.flatnonzero(steps != 1) + 1
    bounds = [0, *breaks.tolist(), positions.size]
    return tuple((int(a), int(b)) for a, b in itertools.pairwise(bounds))


def build_windows(
    values: NDArray[np.float64],
    positions: NDArray[np.int64],
    *,
    seq_len: int,
    pad: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Stack trailing windows that never cross a gap in the bar grid.

    Args:
        values: `(n_rows, n_features)` feature matrix for the rows in hand.
        positions: Grid positions of those rows, from `grid_positions`.
        seq_len: Window length in bars.
        pad: Whether rows with incomplete history still produce a window, by
            repeating the first bar of their run. False drops them.

    Returns:
        `(windows, rows)` where `windows` is `(n_windows, seq_len, n_features)`
        ordered oldest bar first, and `rows` gives the row of `values` each
        window describes — always the window's **last** bar, so a window never
        contains anything dated after the row it is used to score.

    Raises:
        ModelError: If `seq_len` is below one, the shapes disagree, or the
            positions do not strictly increase.
    """
    if seq_len < 1:
        msg = f"seq_len must be at least 1 bar, got {seq_len}."
        raise ModelError(msg)
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != _MATRIX_NDIM:  # pragma: no cover - defensive
        msg = f"Window values must be a 2-D matrix, got shape {matrix.shape}."
        raise ModelError(msg)
    if matrix.shape[0] != positions.size:
        msg = f"Got {matrix.shape[0]} rows of features for {positions.size} grid positions."
        raise ModelError(msg)

    offsets: list[NDArray[np.int64]] = []
    for start, stop in contiguous_runs(positions):
        run = np.arange(start, stop, dtype=np.int64)
        for row in run:
            history = np.arange(row - seq_len + 1, row + 1, dtype=np.int64)
            if history[0] < start:
                if not pad:
                    continue
                # Clamp into the run: the missing head becomes copies of the
                # run's first bar, never a bar from the other side of the gap.
                history = np.clip(history, start, None)
            offsets.append(history)

    n_features = matrix.shape[1]
    if not offsets:
        empty = np.empty((0, seq_len, n_features), dtype=float)
        return empty, np.empty(0, dtype=np.int64)

    taken = np.stack(offsets)
    return matrix[taken], taken[:, -1].astype(np.int64)
