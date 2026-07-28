"""Out-of-fold prediction assembly (§10).

Fit on each fold's training rows, predict its test rows, stack the results. The
stacked frame is what every downstream metric reads, and it is the only place
predictions are allowed to come from — an in-sample prediction has no business
on a leaderboard.

Two properties this guarantees:

* **Every prediction is out-of-fold.** A row is scored by a model that never
  saw it, and never saw any label whose span overlapped it, because the folds
  arrived already purged and embargoed (§9).
* **No row is predicted twice by the same fold.** Walk-forward and purged
  k-fold produce disjoint test blocks; CPCV deliberately does not, so its
  repeated coverage is kept per fold rather than averaged away — those repeats
  are the multiple backtest paths PBO needs (§11.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from alphacore.errors import ModelError
from alphacore.features.selection import prune_correlated
from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from alphacore.config.schema import PruneConfig
    from alphacore.cv import Split
    from alphacore.models.protocol import AlphaModel

__all__ = ["OOFPredictions", "fit_predict_oof"]

_BINARY_CLASSES = 2


@dataclass(frozen=True)
class OOFPredictions:
    """Stacked out-of-fold predictions for one model.

    Attributes:
        model: Model name.
        frame: One row per scored observation, carrying `proba`, `prediction`,
            `label`, `realised_return`, `weight`, and `fold`.
        n_folds: How many folds contributed.
        importances: Per-fold feature importance, when the model reports it.
        selected_features: The columns that survived pruning in each fold, in
            fold order. Empty when no pruning was applied. Recorded per fold
            because the selection is *fitted*, so it legitimately differs
            between folds — and a single global count would hide that.
    """

    model: str
    frame: pd.DataFrame
    n_folds: int
    importances: list[pd.Series] = field(default_factory=list)
    selected_features: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def mean_importance(self) -> pd.Series | None:
        """Average feature importance across folds, descending.

        Returns:
            A Series indexed by feature name, or None if the model reports no
            importance.
        """
        if not self.importances:
            return None
        return pd.concat(self.importances, axis=1).mean(axis=1).sort_values(ascending=False)


def fit_predict_oof(
    model: AlphaModel,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: Sequence[Split],
    *,
    prune: PruneConfig | None = None,
) -> OOFPredictions:
    """Fit a model on every fold and collect its test-fold predictions.

    Args:
        model: The model to fit. Refitted from scratch on each fold, because a
            model carried between folds would accumulate information from test
            periods it has already seen.
        features: Feature frame, aligned to `labels`.
        labels: Label frame, carrying `label`, `realised_return` and `weight`.
        splits: Purged folds from `alphacore.cv`.
        prune: Correlation pruning, applied **inside each fold** with that
            fold's training rows as the fit mask. Passing `None` skips it.
            Pruning on the full sample is banned (§4, I3) and
            `prune_correlated` raises if asked to, so this is the only place
            the `features.prune` section can legitimately take effect.

    Returns:
        The stacked out-of-fold predictions.

    Raises:
        ModelError: If features and labels are misaligned, there are no folds,
            or a fold's training set is empty.
        LeakageError: If pruning is requested and the fit mask cannot be built.
    """
    if not features.index.equals(labels.index):
        msg = (
            f"Features and labels must share an index; got {len(features)} feature rows "
            f"and {len(labels)} label rows. Align them before assembling OOF predictions."
        )
        raise ModelError(msg)
    if not splits:
        msg = "Cannot assemble out-of-fold predictions without any folds."
        raise ModelError(msg)

    # A sequence model cuts its windows on positions in the *full* index, which
    # is the only way it can tell a purged block from the next bar (§10.3). It
    # is told once, here, because this is the only place that knows both the
    # complete index and which rows each fold kept.
    set_bar_grid = getattr(model, "set_bar_grid", None)
    if set_bar_grid is not None:
        set_bar_grid(features.index)

    y = labels["label"]
    weights = labels["weight"].to_numpy(dtype=float)
    pieces: list[pd.DataFrame] = []
    importances: list[pd.Series] = []
    selected: list[tuple[str, ...]] = []

    for split in splits:
        if split.train_idx.size == 0:
            msg = (
                f"Fold {split.fold} has no training rows left after purging. Widen the "
                "training window or shorten the label horizon."
            )
            raise ModelError(msg)

        fold_features = _prune_fold(features, split, prune)
        if prune is not None and prune.method != "none":
            selected.append(tuple(str(c) for c in fold_features.columns))
        train_X = fold_features.iloc[split.train_idx]
        test_X = fold_features.iloc[split.test_idx]
        fitted = _fit_fold(model, train_X, y.iloc[split.train_idx], weights[split.train_idx])

        proba = np.asarray(fitted.predict_proba(test_X), dtype=float)[:, 1]
        pieces.append(
            pd.DataFrame(
                {
                    "proba": proba,
                    "prediction": (proba >= 0.5).astype(int),  # noqa: PLR2004
                    "label": y.iloc[split.test_idx].to_numpy(),
                    "realised_return": labels["realised_return"]
                    .iloc[split.test_idx]
                    .to_numpy(dtype=float),
                    "weight": weights[split.test_idx],
                    "fold": split.fold,
                },
                index=features.index[split.test_idx],
            )
        )
        importance = fitted.feature_importance()
        if importance is not None:
            importances.append(importance)

    frame = pd.concat(pieces)
    frame.index.name = "ts"
    frame = frame.sort_index(kind="stable")

    logger.bind(phase="train").info(
        "{}: {} out-of-fold predictions across {} folds",
        getattr(model, "name", type(model).__name__),
        len(frame),
        len(splits),
    )
    return OOFPredictions(
        model=str(getattr(model, "name", type(model).__name__)),
        frame=frame,
        n_folds=len(splits),
        importances=importances,
        selected_features=selected,
    )


def _prune_fold(features: pd.DataFrame, split: Split, prune: PruneConfig | None) -> pd.DataFrame:
    """Apply correlation pruning fitted on one fold's training rows.

    Args:
        features: The full feature frame.
        split: The fold whose training rows the pruning may see.
        prune: The `features.prune` section, or None to skip.

    Returns:
        The frame with correlated columns removed, or the frame unchanged when
        pruning is disabled.

    Raises:
        LeakageError: If `prune_correlated` rejects the mask.
    """
    if prune is None or prune.method == "none":
        return features
    mask = np.zeros(len(features), dtype=bool)
    mask[split.train_idx] = True
    pruned, dropped = prune_correlated(features, prune, fit_mask=mask)
    if dropped:
        logger.bind(phase="train").debug(
            "fold {}: pruned {} of {} columns", split.fold, len(dropped), features.shape[1]
        )
    return pruned


def _fit_fold(model: AlphaModel, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray) -> Any:  # noqa: ANN401
    """Fit one fold, turning any estimator failure into a typed error.

    Args:
        model: The model to fit.
        X: Training features.
        y: Training labels.
        sample_weight: Per-sample weights.

    Returns:
        The fitted model.

    Raises:
        ModelError: If the fold has one class, or the estimator fails.
    """
    classes = np.unique(y)
    if len(classes) < _BINARY_CLASSES:
        msg = (
            f"A training fold contains only class {classes.tolist()}. "
            "The labelling is degenerate for this window; widen it or check the dead zone."
        )
        raise ModelError(msg)
    if len(classes) > _BINARY_CLASSES:
        msg = (
            f"A training fold has {len(classes)} classes {classes.tolist()}, but every model "
            "in alphacore.models is binary. Triple-barrier labels are {-1, 0, +1}; reduce "
            "them explicitly with alphacore.labels.to_binary(labels, method='direction') to "
            "predict which way it moved, or method='touch' to predict whether the profit "
            "barrier was hit. The choice changes what accuracy means, so it is never made "
            "for you."
        )
        raise ModelError(msg)
    return model.fit(X, y, sample_weight=sample_weight)
