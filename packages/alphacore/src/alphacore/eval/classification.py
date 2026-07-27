"""Prediction-quality metrics (§11.2).

Accuracy, AUC, Brier and log-loss are here because a reviewer expects them, not
because they are the point. §19 is blunt: accuracy above about 0.55 on next-bar
direction almost certainly means leakage, and real short-horizon edges live at
0.51 to 0.54 — a range where accuracy is nearly indistinguishable from noise.

The **information coefficient** is the useful one. It is the Spearman rank
correlation between a prediction and the forward return, so it measures whether
the model ranks opportunities correctly rather than whether it happens to clear
a threshold. Its t-statistic says whether the ranking is distinguishable from
chance at all.

Nothing here returns `NaN`. A metric that cannot be computed raises, because a
`NaN` quietly reaching a leaderboard is a critical bug rather than a missing
value (§0.7).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
from scipy import stats
from sklearn import metrics as skm

from alphacore.errors import ModelError

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "DEFAULT_THRESHOLD",
    "accuracy",
    "brier_score",
    "classification_metrics",
    "information_coefficient",
    "log_loss_score",
    "roc_auc",
]

DEFAULT_THRESHOLD: Final[float] = 0.5
_EPSILON: Final[float] = 1e-15
_MIN_ROWS_FOR_IC: Final[int] = 3
_BINARY_CLASSES: Final[int] = 2


def _check(y_true: NDArray[np.int_], proba: NDArray[np.float64]) -> None:
    """Validate a label/probability pair.

    Args:
        y_true: Binary labels.
        proba: Predicted probability of the positive class.

    Raises:
        ModelError: If the arrays disagree in length, are empty, or contain
            non-finite values.
    """
    if len(y_true) != len(proba):
        msg = f"Got {len(y_true)} labels for {len(proba)} predictions."
        raise ModelError(msg)
    if len(y_true) == 0:
        msg = "Cannot compute a metric on an empty prediction set."
        raise ModelError(msg)
    if not np.isfinite(proba).all():
        msg = "Predicted probabilities contain non-finite values."
        raise ModelError(msg)


def accuracy(
    y_true: NDArray[np.int_], proba: NDArray[np.float64], *, threshold: float = DEFAULT_THRESHOLD
) -> float:
    """Fraction of predictions on the correct side of a threshold.

    Args:
        y_true: Binary labels.
        proba: Predicted probability of the positive class.
        threshold: Decision boundary.

    Returns:
        Accuracy in `[0, 1]`.

    Raises:
        ModelError: If the inputs are invalid.
    """
    _check(y_true, proba)
    return float(np.mean((proba >= threshold).astype(int) == y_true))


def roc_auc(y_true: NDArray[np.int_], proba: NDArray[np.float64]) -> float:
    """Area under the ROC curve.

    Args:
        y_true: Binary labels.
        proba: Predicted probability of the positive class.

    Returns:
        AUC in `[0, 1]`; 0.5 is chance.

    Raises:
        ModelError: If the inputs are invalid, or every label is the same
            class — AUC is undefined there, and returning `NaN` would put an
            unreadable value on a leaderboard.
    """
    _check(y_true, proba)
    if len(np.unique(y_true)) < _BINARY_CLASSES:
        msg = (
            "ROC AUC is undefined when the labels contain only one class. "
            "This usually means a fold is degenerate, not that the model is perfect."
        )
        raise ModelError(msg)
    return float(skm.roc_auc_score(y_true, proba))


def brier_score(y_true: NDArray[np.int_], proba: NDArray[np.float64]) -> float:
    """Mean squared error of the predicted probability.

    `brier = mean((p - y)^2)`. Lower is better; 0.25 is what a constant 0.5
    prediction scores on a balanced problem.

    Args:
        y_true: Binary labels.
        proba: Predicted probability of the positive class.

    Returns:
        The Brier score.

    Raises:
        ModelError: If the inputs are invalid.
    """
    _check(y_true, proba)
    return float(np.mean((proba - y_true) ** 2))


def log_loss_score(y_true: NDArray[np.int_], proba: NDArray[np.float64]) -> float:
    """Binary cross-entropy.

    `-mean(y*ln(p) + (1-y)*ln(1-p))`, with probabilities clipped away from 0
    and 1 so a single confident mistake cannot return infinity.

    Args:
        y_true: Binary labels.
        proba: Predicted probability of the positive class.

    Returns:
        The log loss.

    Raises:
        ModelError: If the inputs are invalid.
    """
    _check(y_true, proba)
    clipped = np.clip(proba, _EPSILON, 1.0 - _EPSILON)
    return float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1.0 - clipped)))


def information_coefficient(
    prediction: NDArray[np.float64], forward_return: NDArray[np.float64]
) -> tuple[float, float]:
    """Spearman rank correlation of prediction against forward return (§11.2).

    More informative than accuracy because it measures ordering: a model that
    ranks the big moves correctly is useful even when its directional calls are
    barely better than a coin.

    The t-statistic is `IC * sqrt((n - 2) / (1 - IC^2))`, which is the standard
    test for a correlation being distinguishable from zero.

    Args:
        prediction: Model output; any monotone transform of it gives the same
            answer, so raw scores and calibrated probabilities agree.
        forward_return: The realised forward return.

    Returns:
        `(ic, t_statistic)`.

    Raises:
        ModelError: If the arrays disagree in length, are too short, or one of
            them is constant — a rank correlation is undefined there.
    """
    if len(prediction) != len(forward_return):
        msg = f"Got {len(prediction)} predictions for {len(forward_return)} returns."
        raise ModelError(msg)
    n = len(prediction)
    if n < _MIN_ROWS_FOR_IC:
        msg = f"Need at least {_MIN_ROWS_FOR_IC} observations for an IC, got {n}."
        raise ModelError(msg)
    if np.ptp(prediction) == 0 or np.ptp(forward_return) == 0:
        msg = (
            "Information coefficient is undefined when the prediction or the forward "
            "return is constant. A constant prediction is a baseline, not a ranking."
        )
        raise ModelError(msg)

    ic = float(stats.spearmanr(prediction, forward_return).statistic)
    if not np.isfinite(ic):  # pragma: no cover - guarded by the ptp check above
        msg = "Information coefficient came out non-finite."
        raise ModelError(msg)

    denominator = 1.0 - ic**2
    tstat = float("inf") if denominator <= 0 else float(ic * np.sqrt((n - 2) / denominator))
    return ic, tstat


def classification_metrics(
    y_true: NDArray[np.int_],
    proba: NDArray[np.float64],
    *,
    forward_return: NDArray[np.float64] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Compute the full prediction-quality set for one model.

    Args:
        y_true: Binary labels.
        proba: Predicted probability of the positive class.
        forward_return: Realised forward returns, for the IC. Omitted, the IC
            falls back to correlating against the label itself, which is
            weaker but still ordering-based.
        threshold: Decision boundary for the count-based metrics.

    Returns:
        A mapping of metric name to value. Every value is finite.

    Raises:
        ModelError: If the inputs are invalid or a metric is undefined.
    """
    _check(y_true, proba)
    predicted = (proba >= threshold).astype(int)
    target = forward_return if forward_return is not None else y_true.astype(float)
    ic, ic_tstat = information_coefficient(proba, np.asarray(target, dtype=float))

    return {
        "accuracy": accuracy(y_true, proba, threshold=threshold),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, predicted)),
        "precision": float(skm.precision_score(y_true, predicted, zero_division=0)),
        "recall": float(skm.recall_score(y_true, predicted, zero_division=0)),
        "f1": float(skm.f1_score(y_true, predicted, zero_division=0)),
        "auc": roc_auc(y_true, proba),
        "pr_auc": float(skm.average_precision_score(y_true, proba)),
        "brier": brier_score(y_true, proba),
        "log_loss": log_loss_score(y_true, proba),
        "ic": ic,
        "ic_tstat": ic_tstat,
    }
