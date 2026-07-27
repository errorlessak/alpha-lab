"""Purged cross-validation (§9).

Three schemes, one contract: every fold is purged and embargoed before it is
handed back. Random `train_test_split` and plain `KFold` are banned outright,
and `tests/unit/test_no_banned_splitters.py` fails the build if either appears
outside the test tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alphacore.cv.cpcv import CPCV
from alphacore.cv.purged import PurgedKFold
from alphacore.cv.splits import (
    FOLDS_FILENAME,
    Split,
    apply_purge_and_embargo,
    build_split,
    splits_from_json,
    splits_to_json,
)
from alphacore.cv.walkforward import WalkForward
from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    import pandas as pd

    from alphacore.config.schema import CVConfig

__all__ = [
    "CPCV",
    "FOLDS_FILENAME",
    "PurgedKFold",
    "Split",
    "WalkForward",
    "apply_purge_and_embargo",
    "build_split",
    "build_splitter",
    "make_splits",
    "splits_from_json",
    "splits_to_json",
]


def build_splitter(config: CVConfig) -> WalkForward | PurgedKFold | CPCV:
    """Construct the scheme named by a CV config.

    Args:
        config: The `cv` section.

    Returns:
        The configured scheme.
    """
    if config.scheme == "walkforward":
        walk = config.walkforward
        return WalkForward(
            train_bars=walk.train_bars,
            test_bars=walk.test_bars,
            step_bars=walk.step_bars,
            anchored=walk.anchored,
            embargo_pct=config.embargo_pct,
            min_train_bars=config.min_train_bars,
        )
    if config.scheme == "purged_kfold":
        return PurgedKFold(n_splits=config.n_splits, embargo_pct=config.embargo_pct)
    return CPCV(
        n_groups=config.cpcv.n_groups,
        n_test_groups=config.cpcv.n_test_groups,
        embargo_pct=config.embargo_pct,
    )
    # No `else`: `CVConfig.scheme` is a Literal over exactly these three, so
    # mypy proves the branch exhaustive. Adding a fourth becomes a type error.


def make_splits(labels: pd.DataFrame, config: CVConfig) -> list[Split]:
    """Build the folds for a label frame under a CV config.

    Args:
        labels: The label frame, carrying `t_end`.
        config: The `cv` section.

    Returns:
        The folds, in the scheme's own order.

    Raises:
        CVError: If the geometry cannot produce a valid fold.
    """
    splits = build_splitter(config).split(labels)
    logger.bind(phase="cv").info(
        "{}: {} folds, {} samples purged, {} embargoed",
        config.scheme,
        len(splits),
        sum(s.purged for s in splits),
        sum(s.embargoed for s in splits),
    )
    return splits
