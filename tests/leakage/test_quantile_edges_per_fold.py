"""Quantile label edges are fitted per fold, never globally (§12.3.6).

Global quantile labels are on §19's list of things that sink a project, and
they are hard to spot by reading code: nothing looks like a look-ahead. The
mechanism is subtler. If the edge between "hold" and "strong buy" is the 80th
percentile of the *whole sample*, then during training the model already knows
how large returns get in the test period. Every class boundary carries a little
of the future in it.

These tests check three things: that the edges actually move between folds,
that the labels move with them, and — the control — that a deliberately global
fit is detectably different from a per-fold one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacore.errors import LeakageError
from alphacore.labels import horizon_quantile
from alphacore.schemas.bars import normalise_bars, validate_bars

N_BARS = 600
N_FOLDS = 4
HORIZON = 5
N_BINS = 5


@pytest.fixture
def bars() -> pd.DataFrame:
    """A walk whose volatility regime changes, so fold edges genuinely differ."""
    rng = np.random.default_rng(20260727)
    # Volatility doubles halfway through: any honest per-fold fit must notice.
    scale = np.concatenate((np.full(N_BARS // 2, 0.005), np.full(N_BARS // 2, 0.02)))
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(N_BARS) * scale))
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2015-01-01", periods=N_BARS, freq="D", tz="UTC"),
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(N_BARS, 1e6),
            "adj_factor": 1.0,
            "symbol": "SYNTH",
        }
    )
    return validate_bars(normalise_bars(frame))


def fold_masks(n: int, n_folds: int) -> list[np.ndarray]:
    """Contiguous, disjoint training masks standing in for walk-forward folds."""
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    masks = []
    for i in range(n_folds):
        mask = np.zeros(n, dtype=bool)
        mask[bounds[i] : bounds[i + 1]] = True
        masks.append(mask)
    return masks


def test_quantile_edges_per_fold(bars):
    """Each fold must produce its own bin edges (§8.1)."""
    edges = [
        tuple(
            horizon_quantile(bars, horizon_bars=HORIZON, n_bins=N_BINS, fit_mask=mask).attrs[
                "bin_edges"
            ]
        )
        for mask in fold_masks(len(bars), N_FOLDS)
    ]

    assert len(edges) == N_FOLDS
    assert len(set(edges)) == N_FOLDS, (
        "Two folds produced identical quantile edges, which means the edges were not "
        "fitted on the fold's own training rows."
    )


def test_edges_track_the_volatility_regime(bars):
    """A calmer training window must produce tighter class boundaries."""
    masks = fold_masks(len(bars), 2)
    calm = horizon_quantile(bars, horizon_bars=HORIZON, n_bins=N_BINS, fit_mask=masks[0])
    volatile = horizon_quantile(bars, horizon_bars=HORIZON, n_bins=N_BINS, fit_mask=masks[1])

    calm_width = calm.attrs["bin_edges"][-1] - calm.attrs["bin_edges"][0]
    volatile_width = volatile.attrs["bin_edges"][-1] - volatile.attrs["bin_edges"][0]
    assert volatile_width > calm_width


def test_labels_change_when_the_fold_changes(bars):
    """Different edges must actually reassign observations, not just be recorded."""
    masks = fold_masks(len(bars), 2)
    first = horizon_quantile(bars, horizon_bars=HORIZON, n_bins=N_BINS, fit_mask=masks[0])
    second = horizon_quantile(bars, horizon_bars=HORIZON, n_bins=N_BINS, fit_mask=masks[1])
    assert not first["label"].equals(second["label"])


def test_a_global_fit_is_detectably_different_from_a_per_fold_one(bars):
    """The control: fitting on everything must not coincide with fitting on a fold.

    If it did, this whole suite would be measuring nothing.
    """
    everything = np.ones(len(bars), dtype=bool)
    global_edges = horizon_quantile(
        bars, horizon_bars=HORIZON, n_bins=N_BINS, fit_mask=everything
    ).attrs["bin_edges"]

    for mask in fold_masks(len(bars), N_FOLDS):
        fold_edges = horizon_quantile(
            bars, horizon_bars=HORIZON, n_bins=N_BINS, fit_mask=mask
        ).attrs["bin_edges"]
        assert fold_edges != global_edges


def test_quantile_labelling_cannot_run_without_a_fit_mask(bars):
    """There is no API path that fits edges globally by accident."""
    with pytest.raises(LeakageError, match="per fold"):
        horizon_quantile(bars, horizon_bars=HORIZON, n_bins=N_BINS, fit_mask=None)


def test_a_mismatched_mask_is_rejected(bars):
    with pytest.raises(LeakageError, match="entries"):
        horizon_quantile(
            bars,
            horizon_bars=HORIZON,
            n_bins=N_BINS,
            fit_mask=np.ones(len(bars) - 10, dtype=bool),
        )
