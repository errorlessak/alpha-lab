"""Walk-forward validation (§9).

The primary scheme, and the one the UI's Fold Strip visualises. Test blocks
march forward in time; training always ends before the test block begins.

Two variants:

* **rolling** — a fixed `train_bars` window slides forward. Assumes the recent
  past is more informative than the distant past, and keeps the training
  distribution roughly stationary.
* **anchored** — training expands from the start of the sample. Uses more data
  and produces more stable estimates, at the cost of averaging over regimes
  that may no longer apply.

Neither is right in general. The scheme is configuration precisely because the
choice is a modelling assumption, and the fold artifact records which was used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alphacore.cv.splits import Split, build_split
from alphacore.errors import CVError

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["WalkForward"]


class WalkForward:
    """Anchored or rolling walk-forward splits, in chronological order."""

    def __init__(
        self,
        *,
        train_bars: int,
        test_bars: int,
        step_bars: int,
        anchored: bool = False,
        embargo_pct: float = 0.0,
        min_train_bars: int = 0,
    ) -> None:
        """Configure the split geometry.

        Args:
            train_bars: Training window length. For the anchored variant this
                is the *minimum* before the first fold.
            test_bars: Test block length.
            step_bars: Bars between consecutive test blocks.
            anchored: True to expand the training window from the sample start.
            embargo_pct: Fraction of the sample embargoed after each test block.
            min_train_bars: Reject any fold with fewer training samples.

        Raises:
            CVError: If any length is not positive.
        """
        for name, value in (
            ("train_bars", train_bars),
            ("test_bars", test_bars),
            ("step_bars", step_bars),
        ):
            if value < 1:
                msg = f"{name} must be positive, got {value}."
                raise CVError(msg)
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.anchored = anchored
        self.embargo_pct = embargo_pct
        self.min_train_bars = min_train_bars

    def __repr__(self) -> str:
        """Return a debug representation naming the geometry."""
        kind = "anchored" if self.anchored else "rolling"
        return (
            f"WalkForward({kind}, train={self.train_bars}, test={self.test_bars}, "
            f"step={self.step_bars})"
        )

    def split(self, labels: pd.DataFrame) -> list[Split]:
        """Produce the folds.

        Args:
            labels: The label frame, carrying `t_end`.

        Returns:
            Folds in chronological order.

        Raises:
            CVError: If the series is too short for even one fold, or a fold
                would train on fewer than `min_train_bars` samples.
        """
        n = len(labels)
        if n < self.train_bars + self.test_bars:
            msg = (
                f"Series of {n} labels is too short for a walk-forward with "
                f"train_bars={self.train_bars} and test_bars={self.test_bars}. "
                "Shorten the windows or ingest more history."
            )
            raise CVError(msg)

        splits: list[Split] = []
        fold = 0
        test_start = self.train_bars
        while test_start + self.test_bars <= n:
            test_idx = np.arange(test_start, test_start + self.test_bars, dtype=np.int64)
            train_start = 0 if self.anchored else max(0, test_start - self.train_bars)
            train_idx = np.arange(train_start, test_start, dtype=np.int64)

            split = build_split(fold, labels, train_idx, test_idx, embargo_pct=self.embargo_pct)
            if split.train_idx.size < self.min_train_bars:
                msg = (
                    f"Fold {fold} keeps only {split.train_idx.size} training samples after "
                    f"purging, below min_train_bars={self.min_train_bars}. Purging removed "
                    f"{split.purged} and the embargo {split.embargoed}; widen train_bars or "
                    "shorten the label horizon."
                )
                raise CVError(msg)

            splits.append(split)
            fold += 1
            test_start += self.step_bars

        if not splits:  # pragma: no cover - guarded by the length check above
            msg = "Walk-forward produced no folds."
            raise CVError(msg)
        return splits
