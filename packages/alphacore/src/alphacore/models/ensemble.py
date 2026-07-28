"""Averaging ensemble over any two model families (§10.3).

Phase 13 asks for an ensemble of a sequence model with a GBM, which is worth
having for a reason that is not "ensembles score better": the two families fail
differently. A GBM reads each bar's feature vector independently and cannot see
a shape; a recurrent net reads the shape and is far easier to overfit on two
thousand rows. When the blend beats neither leg on out-of-fold economics, that
is a result about the signal, not about the blend.

The average is taken over **probabilities**, not over logits or ranks. The
probabilities are already calibrated per member (§10.2), so their mean is still
a probability the threshold rules in §config can act on; a mean of logits is
not, and would silently change what `signal.thresholds` means.

Nothing here is torch-specific. `EnsembleModel` takes any models satisfying
`AlphaModel`, which is also how it gets tested without the `dl` extra
installed.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from alphacore.errors import ModelError
from alphacore.models.protocol import Task

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from alphacore.models.protocol import AlphaModel

__all__ = ["EnsembleModel"]


class EnsembleModel:
    """Average the calibrated probabilities of several fitted models."""

    task: Task = "binary"

    def __init__(
        self,
        name: str,
        members: Sequence[AlphaModel],
        *,
        weights: Sequence[float] | None = None,
    ) -> None:
        """Assemble the ensemble.

        Args:
            name: Ensemble name, as it appears on the leaderboard.
            members: The models to blend. Fitted in the order given, each on the
                same rows.
            weights: Blend weights, defaulting to equal. Normalised to sum to 1
                so the blend stays a probability.

        Raises:
            ModelError: If there are no members, or the weights do not match
                them, are negative, or sum to zero.
        """
        if not members:
            msg = f"Ensemble {name!r} needs at least one member model."
            raise ModelError(msg)
        if weights is None:
            blend = np.full(len(members), 1.0 / len(members))
        else:
            if len(weights) != len(members):
                msg = f"Ensemble {name!r} got {len(weights)} weights for {len(members)} members."
                raise ModelError(msg)
            values = np.asarray(weights, dtype=float)
            if np.any(values < 0) or values.sum() <= 0:
                msg = f"Ensemble {name!r} needs non-negative weights that sum to more than zero."
                raise ModelError(msg)
            blend = values / values.sum()
        self.name = name
        self.members = list(members)
        self.weights = blend
        self._fitted = False

    def __repr__(self) -> str:
        """Return a debug representation naming the members."""
        return f"EnsembleModel({self.name!r}, members={[m.name for m in self.members]})"

    def set_bar_grid(self, index: pd.Index) -> None:
        """Pass the full bar grid on to any member that needs it.

        Sequence members cut their windows on grid positions, so they must be
        told the index the folds were built from (§10.3). Members that do not
        window are left alone.

        Args:
            index: The complete feature index.
        """
        for member in self.members:
            setter = getattr(member, "set_bar_grid", None)
            if setter is not None:
                setter(index)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: NDArray[np.float64] | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> EnsembleModel:
        """Fit every member on the same training rows.

        Args:
            X: Training features.
            y: Training labels.
            sample_weight: Per-sample weights (§8.2).
            eval_set: Optional inner validation split, forwarded unchanged.

        Returns:
            Self, fitted.

        Raises:
            ModelError: If a member refuses the data.
        """
        for member in self.members:
            member.fit(X, y, sample_weight=sample_weight, eval_set=eval_set)
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        """Check the ensemble has been fitted.

        Raises:
            ModelError: If it has not.
        """
        if not self._fitted:
            msg = f"Ensemble {self.name!r} must be fitted before it can predict."
            raise ModelError(msg)

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Blend the members' probabilities.

        Args:
            X: Feature frame.

        Returns:
            An `(n, 2)` array whose rows sum to 1.

        Raises:
            ModelError: If the ensemble has not been fitted.
        """
        self._require_fitted()
        stacked = np.column_stack(
            [np.asarray(member.predict_proba(X), dtype=float)[:, 1] for member in self.members]
        )
        positive = np.clip(stacked @ self.weights, 0.0, 1.0)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: pd.DataFrame) -> NDArray[np.int_]:
        """Predict class labels at a 0.5 threshold.

        Args:
            X: Feature frame.

        Returns:
            An `(n,)` integer array.

        Raises:
            ModelError: If the ensemble has not been fitted.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)  # noqa: PLR2004

    def feature_importance(self) -> pd.Series | None:
        """Average the importance of the members that report one.

        A sequence member reports nothing, so a GBM-plus-LSTM blend reports the
        GBM's ranking alone. That is the honest answer: averaging in a vector of
        zeros for the net would halve every score and imply the net agreed
        nothing mattered.

        Returns:
            The averaged importance, or None if no member reports any.
        """
        self._require_fitted()
        reported = [member.feature_importance() for member in self.members]
        usable = [series for series in reported if series is not None]
        if not usable:
            return None
        combined = pd.concat(usable, axis=1).mean(axis=1)
        combined.name = self.name
        return combined.sort_values(ascending=False)

    def save(self, path: Path) -> None:
        """Persist the fitted ensemble with pickle.

        Args:
            path: Destination file.

        Raises:
            ModelError: If the ensemble has not been fitted.
        """
        self._require_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> EnsembleModel:
        """Load a persisted ensemble.

        Args:
            path: Source file, written by `save`.

        Returns:
            The restored ensemble.

        Raises:
            ModelError: If the file does not contain one.
        """
        obj: Any = pickle.loads(path.read_bytes())
        if not isinstance(obj, EnsembleModel):
            msg = f"{path} does not contain an alphacore ensemble."
            raise ModelError(msg)
        return obj
