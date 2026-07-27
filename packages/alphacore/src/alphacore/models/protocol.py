"""The model contract (§10.1).

One interface for a logistic regression, three gradient-boosted libraries, five
baselines, and later a handful of sequence models. Everything downstream — OOF
assembly, the leaderboard, tuning, SHAP — talks to this and nothing else.

`predict_proba` returning a two-column array even for the baselines is
deliberate. A buy-and-hold "model" that returned a bare signal would need a
special case in the OOF loop, in the calibration step, and in every metric; a
special case is where the leak gets in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

__all__ = ["AlphaModel", "Task"]

Task = Literal["binary", "multiclass", "regression"]


@runtime_checkable
class AlphaModel(Protocol):
    """Fits on a training fold and scores a test fold."""

    name: str
    task: Task

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: NDArray[np.float64] | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> Self:
        """Fit on one training fold.

        Args:
            X: Feature frame for the training rows only.
            y: Labels aligned to `X`.
            sample_weight: Per-sample weights from §8.2. Overlapping labels
                break the i.i.d. assumption, so passing these is not optional
                for any estimator that accepts them.
            eval_set: Optional inner validation split for early stopping.

        Returns:
            Self, fitted.

        Raises:
            ModelError: If the inputs are misaligned or the fit fails.
        """
        ...

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            X: Feature frame.

        Returns:
            An `(n_samples, n_classes)` array whose rows sum to 1.

        Raises:
            ModelError: If the model has not been fitted.
        """
        ...

    def predict(self, X: pd.DataFrame) -> NDArray[np.int_]:
        """Predict class labels.

        Args:
            X: Feature frame.

        Returns:
            An `(n_samples,)` array of labels.

        Raises:
            ModelError: If the model has not been fitted.
        """
        ...

    def feature_importance(self) -> pd.Series | None:
        """Report per-feature importance.

        Returns:
            A Series indexed by feature name, or None for models that have no
            meaningful notion of importance. None rather than zeros: a baseline
            that reports zero importance for every feature looks like a model
            that found nothing, which is a different claim.
        """
        ...

    def save(self, path: Path) -> None:
        """Persist the fitted model.

        Args:
            path: Destination file.
        """
        ...

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a persisted model.

        Args:
            path: Source file.

        Returns:
            The restored model.
        """
        ...
