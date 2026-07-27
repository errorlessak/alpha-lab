"""Feature selection (§7.3).

Two pruning strategies, and one rule that governs both: **the decision is
fitted on an explicit subset of rows, never on everything**.

That rule is invariant I3 applied to feature selection, and it is easy to
violate without noticing. Computing a correlation matrix over the full sample
and dropping columns on that basis uses test-period data to decide what the
model may see — a leak that inflates out-of-sample results without ever
touching a label. `prune_correlated` therefore requires a `fit_mask` and raises
if pruning is requested without one.

`clustered_correlation` is the better of the two: raw pairwise pruning on a
family of thirty correlated moving averages keeps whichever it happens to visit
first, while clustering keeps one representative per group of genuinely similar
features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from alphacore.errors import LeakageError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alphacore.config.schema import PruneConfig

__all__ = ["clustered_correlation_prune", "correlation_prune", "prune_correlated"]

_MIN_COLUMNS_TO_COMPARE = 2
"""Below this there is no pair to correlate, so pruning is a no-op."""


def _fit_frame(frame: pd.DataFrame, fit_mask: NDArray[np.bool_] | None) -> pd.DataFrame:
    """Select the rows a selection decision may be fitted on.

    Args:
        frame: The full feature frame.
        fit_mask: Boolean mask over rows, or None.

    Returns:
        The masked frame.

    Raises:
        LeakageError: If no mask was supplied. Selecting on all rows fits a
            decision on the test period (§4, I3).
    """
    if fit_mask is None:
        msg = (
            "Correlation pruning needs an explicit fit_mask naming the training rows. "
            "Fitting the correlation matrix on every row uses test-period data to decide "
            "which features the model may see, which inflates out-of-sample results "
            "without ever touching a label (§4, I3)."
        )
        raise LeakageError(msg)
    if len(fit_mask) != len(frame):
        msg = f"fit_mask has {len(fit_mask)} entries for a frame of {len(frame)} rows."
        raise LeakageError(msg)
    return frame.loc[fit_mask]


def correlation_prune(
    frame: pd.DataFrame, *, threshold: float
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Drop columns correlated above a threshold with an already-kept column.

    Columns are visited in name order, so the survivor of a correlated pair does
    not depend on dict iteration order (§4, I8).

    Args:
        frame: Rows the decision is fitted on.
        threshold: Absolute Pearson correlation above which a column is dropped.

    Returns:
        The kept frame, and a mapping of dropped column to the reason it went.
    """
    if frame.shape[1] < _MIN_COLUMNS_TO_COMPARE:
        return frame.copy(), {}

    corr = frame.corr(numeric_only=True).abs()
    kept: list[str] = []
    dropped: dict[str, str] = {}

    for column in sorted(frame.columns):
        rival = next(
            (k for k in kept if float(corr.loc[column, k]) > threshold),
            None,
        )
        if rival is None:
            kept.append(column)
        else:
            dropped[column] = (
                f"|corr| = {float(corr.loc[column, rival]):.4f} with {rival!r}, "
                f"above the {threshold} threshold"
            )
    return frame.loc[:, [c for c in frame.columns if c in set(kept)]].copy(), dropped


def clustered_correlation_prune(
    frame: pd.DataFrame, *, threshold: float
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Cluster correlated columns and keep one representative each (§7.3).

    Distance is `sqrt(0.5 * (1 - corr))`, the standard correlation metric, and
    clusters are cut at the distance corresponding to `threshold`. The
    representative kept is the column with the highest average absolute
    correlation to its own cluster — the most typical member, rather than
    whichever came first alphabetically.

    Args:
        frame: Rows the decision is fitted on.
        threshold: Correlation above which columns are considered redundant.

    Returns:
        The kept frame, and a mapping of dropped column to the reason it went.
    """
    if frame.shape[1] < _MIN_COLUMNS_TO_COMPARE:
        return frame.copy(), {}

    columns = sorted(frame.columns)
    corr = frame.loc[:, columns].corr(numeric_only=True).fillna(0.0)
    distance = np.sqrt(np.clip(0.5 * (1.0 - corr.to_numpy()), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0

    cut = float(np.sqrt(0.5 * (1.0 - threshold)))
    labels = fcluster(
        linkage(squareform(distance, checks=False), method="average"), cut, "distance"
    )

    kept: list[str] = []
    dropped: dict[str, str] = {}
    for cluster in sorted(set(labels)):
        members = [columns[i] for i in range(len(columns)) if labels[i] == cluster]
        if len(members) == 1:
            kept.append(members[0])
            continue
        block = corr.loc[members, members].abs()
        representative = str(block.mean(axis=1).idxmax())
        kept.append(representative)
        for member in members:
            if member != representative:
                dropped[member] = (
                    f"clustered with {representative!r} "
                    f"(|corr| = {float(block.loc[member, representative]):.4f}) "
                    f"at threshold {threshold}"
                )
    keep = set(kept)
    return frame.loc[:, [c for c in frame.columns if c in keep]].copy(), dropped


def prune_correlated(
    frame: pd.DataFrame,
    config: PruneConfig,
    *,
    fit_mask: NDArray[np.bool_] | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Apply the configured pruning method.

    Args:
        frame: The full feature frame. Pruning is *fitted* on `fit_mask` rows
            and *applied* to all of them, which is the same discipline every
            other fitted transform follows (§4, I3).
        config: The `features.prune` section.
        fit_mask: Boolean mask naming the training rows. Required unless the
            method is `none`.

    Returns:
        The pruned frame, and a mapping of dropped column to the reason.

    Raises:
        LeakageError: If pruning is requested without a fit mask, or the mask
            length does not match the frame.
    """
    if config.method == "none":
        return frame.copy(), {}

    fitted_on = _fit_frame(frame, fit_mask)
    if config.method == "correlation":
        _, dropped = correlation_prune(fitted_on, threshold=config.threshold)
    else:
        _, dropped = clustered_correlation_prune(fitted_on, threshold=config.threshold)

    keep = [c for c in frame.columns if c not in dropped]
    return frame.loc[:, keep].copy(), dropped
