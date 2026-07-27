"""Purged k-fold cross-validation (§9).

Contiguous test blocks over the sample; training is everything else, with
overlapping labels purged and a post-test embargo applied.

Blocks are contiguous rather than shuffled, which is the whole point. Ordinary
`KFold` on a time series scatters test rows through the training period, so
almost every training sample sits next to a test sample and shares its feature
windows. The resulting score is not out-of-sample in any useful sense. §9 bans
plain `KFold` and `train_test_split` outright, and a CI check enforces it.

Note that purged k-fold still trains on data that *follows* the test block. That
is legitimate for measuring predictive relationships, and dishonest for
simulating a strategy someone could have run. Walk-forward is the scheme for the
latter, which is why it is the default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alphacore.cv.splits import Split, build_split
from alphacore.errors import CVError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["PurgedKFold"]

_MIN_SPLITS = 2


class PurgedKFold:
    """K contiguous test blocks with purging and an embargo."""

    def __init__(self, *, n_splits: int = 5, embargo_pct: float = 0.0) -> None:
        """Configure the fold count.

        Args:
            n_splits: Number of contiguous test blocks.
            embargo_pct: Fraction of the sample embargoed after each block.
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"PurgedKFold(n_splits={self.n_splits}, embargo_pct={self.embargo_pct})"

    def split(self, labels: pd.DataFrame) -> list[Split]:
        """Produce the folds.

        Args:
            labels: The label frame, carrying `t_end`.

        Returns:
            Folds in chronological order of their test blocks.

        Raises:
            CVError: If `n_splits` is below 2 or exceeds the sample size, or
                the label frame is empty.
        """
        if labels.empty:
            msg = "Cannot build folds from an empty label frame."
            raise CVError(msg)
        if self.n_splits < _MIN_SPLITS:
            msg = f"n_splits must be at least {_MIN_SPLITS}, got {self.n_splits}."
            raise CVError(msg)
        n = len(labels)
        if self.n_splits > n:
            msg = f"n_splits={self.n_splits} exceeds the {n} available labels."
            raise CVError(msg)

        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        splits: list[Split] = []
        for fold in range(self.n_splits):
            test_idx = np.arange(bounds[fold], bounds[fold + 1], dtype=np.int64)
            train_idx = np.concatenate(
                (
                    np.arange(0, bounds[fold], dtype=np.int64),
                    np.arange(bounds[fold + 1], n, dtype=np.int64),
                )
            )
            splits.append(
                build_split(fold, labels, train_idx, test_idx, embargo_pct=self.embargo_pct)
            )
        return splits
