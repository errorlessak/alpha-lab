"""Combinatorial purged cross-validation (§9).

Split the sample into `N` contiguous groups and hold out every combination of
`k` of them, giving `C(N, k)` folds instead of `N`.

The point is not more folds. It is that each bar is tested `C(N-1, k-1)` times,
in different company each time, which yields **many backtest paths** rather than
one. A single walk-forward gives one Sharpe ratio and no way to know whether it
was luck. CPCV gives a distribution, and that distribution is the input to the
probability of backtest overfitting in §11.3.

The cost is real: `C(6, 2)` is 15 folds, `C(10, 3)` is 120, and each one trains
a model. That is the trade CPCV asks you to make.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from alphacore.cv.splits import Split, build_split
from alphacore.errors import CVError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["CPCV"]


class CPCV:
    """All `C(n_groups, n_test_groups)` purged combinations."""

    def __init__(
        self, *, n_groups: int = 6, n_test_groups: int = 2, embargo_pct: float = 0.0
    ) -> None:
        """Configure the group geometry.

        Args:
            n_groups: Number of contiguous groups the sample is cut into.
            n_test_groups: How many groups are held out per fold.
            embargo_pct: Fraction of the sample embargoed after each test span.
        """
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    def __repr__(self) -> str:
        """Return a debug representation naming the combination count."""
        return f"CPCV(n_groups={self.n_groups}, n_test_groups={self.n_test_groups})"

    def split(self, labels: pd.DataFrame) -> list[Split]:
        """Produce every combination as a fold.

        Args:
            labels: The label frame, carrying `t_end`.

        Returns:
            `C(n_groups, n_test_groups)` folds, in lexicographic order of their
            held-out group combination — deterministic, so two runs produce the
            same fold numbering (§4, I8).

        Raises:
            CVError: If the geometry leaves no training data, or the sample is
                too small to cut into groups.
        """
        if labels.empty:
            msg = "Cannot build folds from an empty label frame."
            raise CVError(msg)
        if self.n_test_groups >= self.n_groups:
            msg = (
                f"n_test_groups={self.n_test_groups} must be below n_groups={self.n_groups}; "
                "otherwise no training data remains."
            )
            raise CVError(msg)
        if self.n_test_groups < 1:
            msg = f"n_test_groups must be at least 1, got {self.n_test_groups}."
            raise CVError(msg)
        n = len(labels)
        if self.n_groups > n:
            msg = f"n_groups={self.n_groups} exceeds the {n} available labels."
            raise CVError(msg)

        bounds = np.linspace(0, n, self.n_groups + 1).astype(int)
        groups = [np.arange(bounds[g], bounds[g + 1], dtype=np.int64) for g in range(self.n_groups)]

        splits: list[Split] = []
        for fold, held_out in enumerate(combinations(range(self.n_groups), self.n_test_groups)):
            test_idx = np.sort(np.concatenate([groups[g] for g in held_out]))
            train_idx = np.sort(
                np.concatenate([groups[g] for g in range(self.n_groups) if g not in held_out])
            )
            splits.append(
                build_split(fold, labels, train_idx, test_idx, embargo_pct=self.embargo_pct)
            )
        return splits
