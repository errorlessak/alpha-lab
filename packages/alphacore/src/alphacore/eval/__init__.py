"""Evaluation metrics (§11).

Phase 5 ships the prediction-quality set. The financial metrics, the deflated
Sharpe machinery, and PBO arrive with Phase 8.
"""

from __future__ import annotations

from alphacore.eval.classification import (
    DEFAULT_THRESHOLD,
    accuracy,
    brier_score,
    classification_metrics,
    information_coefficient,
    log_loss_score,
    roc_auc,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "accuracy",
    "brier_score",
    "classification_metrics",
    "information_coefficient",
    "log_loss_score",
    "roc_auc",
]
