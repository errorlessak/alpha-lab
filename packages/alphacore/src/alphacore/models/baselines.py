"""Mandatory baselines (§10.1).

These are not filler. They are the reference points that turn a Sharpe ratio
into a claim someone can evaluate:

* **buy-and-hold** — the thing any strategy has to beat to be worth its costs.
* **always-long** — identical in prediction, different in the backtest: it
  re-enters every bar and therefore pays the cost model. The gap between the
  two is the turnover bill.
* **random-sign** — seeded, so the "random" comparison is reproducible.
* **momentum** — a rule with no fitting at all. A learned model that cannot
  beat `sign(recent return)` has learned nothing worth the machinery.
* **shuffled-label** — the honest one. The same estimator, trained on labels
  whose relationship to the features has been destroyed. Whatever Sharpe this
  produces is the Sharpe the pipeline manufactures from nothing, and §10.1 is
  explicit: if the real model's distribution overlaps it, say so prominently.

The shuffle happens inside `fit`, on training labels only, using a stream
derived from the run seed. Shuffling before the split would leak test labels
into training through the permutation itself.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from alphacore.errors import ModelError
from alphacore.models.base import as_label_series
from alphacore.runtime.determinism import deterministic_rng

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from alphacore.models.protocol import AlphaModel

__all__ = [
    "MANDATORY_BASELINES",
    "AlwaysLong",
    "BuyAndHold",
    "MomentumRule",
    "RandomSign",
    "ShuffledLabelModel",
]

MANDATORY_BASELINES: Final[tuple[str, ...]] = (
    "baseline_buyhold",
    "baseline_always_long",
    "baseline_random_sign",
    "baseline_momentum",
    "baseline_shuffled",
)
"""Every leaderboard carries all five, whether or not the config asked for
them. A leaderboard without a shuffled-label row is a leaderboard with no scale
on its axis."""

_CONFIDENT: Final[float] = 0.99
"""Probability a deterministic baseline reports. Not 1.0: a probability of
exactly one propagates as an infinite log-loss and makes the whole column
unreadable."""


class _Baseline:
    """Shared plumbing for the fitting-free baselines."""

    task = "binary"
    name = "baseline"

    def __init__(self, *, seed: int) -> None:
        """Record the seed.

        Args:
            seed: Run seed, used by the stochastic baselines.
        """
        self.seed = seed
        self._fitted = False

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"{type(self).__name__}(seed={self.seed})"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,  # noqa: ARG002
        *,
        sample_weight: NDArray[np.float64] | None = None,  # noqa: ARG002
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,  # noqa: ARG002
    ) -> _Baseline:
        """Record that the baseline has been "fitted".

        Args:
            X: Training features. Used only to remember the column order.
            y: Ignored.
            sample_weight: Ignored.
            eval_set: Ignored.

        Returns:
            Self.
        """
        self._columns = [str(c) for c in X.columns]
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        """Raise if the baseline has not been fitted.

        Raises:
            ModelError: If `fit` has not been called.
        """
        if not self._fitted:
            msg = f"Model {self.name!r} must be fitted before it can predict."
            raise ModelError(msg)

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            X: Feature frame.

        Returns:
            An `(n, 2)` array.

        Raises:
            NotImplementedError: Always; each baseline supplies its own rule.
        """
        raise NotImplementedError

    def _as_proba(self, positive: NDArray[np.float64]) -> NDArray[np.float64]:
        """Assemble a two-column probability array.

        Args:
            positive: Probability of the positive class.

        Returns:
            An `(n, 2)` array whose rows sum to 1.
        """
        clipped = np.clip(positive, 0.0, 1.0)
        return np.column_stack((1.0 - clipped, clipped))

    def predict(self, X: pd.DataFrame) -> NDArray[np.int_]:
        """Predict class labels at a 0.5 threshold.

        Args:
            X: Feature frame.

        Returns:
            An `(n,)` integer array.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)  # noqa: PLR2004

    def feature_importance(self) -> pd.Series | None:
        """Baselines have no feature importance.

        Returns:
            None — rather than zeros, which would read as "looked and found
            nothing" instead of "never looked".
        """
        return None

    def save(self, path: Path) -> None:
        """Persist the baseline.

        Args:
            path: Destination file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> _Baseline:
        """Load a persisted baseline.

        Args:
            path: Source file.

        Returns:
            The restored baseline.

        Raises:
            ModelError: If the file holds something else.
        """
        obj = pickle.loads(path.read_bytes())
        if not isinstance(obj, _Baseline):
            msg = f"{path} does not contain an alphacore baseline."
            raise ModelError(msg)
        return obj


class BuyAndHold(_Baseline):
    """Always predicts the positive class; never re-enters."""

    name = "baseline_buyhold"

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Predict the positive class with fixed confidence.

        Args:
            X: Feature frame.

        Returns:
            An `(n, 2)` array.
        """
        self._require_fitted()
        return self._as_proba(np.full(len(X), _CONFIDENT))


class AlwaysLong(BuyAndHold):
    """Identical predictions to buy-and-hold; distinguished in the backtest.

    Buy-and-hold enters once. Always-long re-establishes a full position every
    bar, so it pays the cost model on every rebalance. The distance between the
    two rows on a leaderboard is the turnover bill, made visible.
    """

    name = "baseline_always_long"


class RandomSign(_Baseline):
    """Seeded coin flips (§10.1)."""

    name = "baseline_random_sign"

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Draw reproducible pseudo-random probabilities.

        Args:
            X: Feature frame.

        Returns:
            An `(n, 2)` array.
        """
        self._require_fitted()
        rng = deterministic_rng(self.seed, "baseline_random_sign", len(X))
        return self._as_proba(rng.uniform(size=len(X)))


class MomentumRule(_Baseline):
    """Goes long when a recent-return feature is positive.

    Deliberately fitting-free. It is the bar a learned model has to clear
    before its machinery has earned its complexity.
    """

    name = "baseline_momentum"

    def __init__(self, *, seed: int, column: str | None = None) -> None:
        """Configure which column carries the momentum signal.

        Args:
            seed: Unused; present for interface symmetry.
            column: Feature to read. None picks the first column whose name
                looks like a return or a momentum measure.
        """
        super().__init__(seed=seed)
        self.column = column

    def _signal_column(self, X: pd.DataFrame) -> str:
        """Choose the column to read.

        Args:
            X: Feature frame.

        Returns:
            A column name.

        Raises:
            ModelError: If a configured column is absent.
        """
        if self.column is not None:
            if self.column not in X.columns:
                msg = f"Momentum baseline needs column {self.column!r}, which is not present."
                raise ModelError(msg)
            return self.column
        for hint in ("log_return", "simple_return", "momentum", "roc"):
            matches = sorted(str(c) for c in X.columns if str(c).startswith(hint))
            if matches:
                return matches[0]
        return str(X.columns[0])

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Map the sign of the signal column onto a probability.

        Args:
            X: Feature frame.

        Returns:
            An `(n, 2)` array.
        """
        self._require_fitted()
        values = np.asarray(X[self._signal_column(X)], dtype=float)
        return self._as_proba(np.where(values > 0, _CONFIDENT, 1.0 - _CONFIDENT))


class ShuffledLabelModel:
    """Wraps a real model and destroys the label relationship before fitting.

    The single most informative row on the leaderboard. Whatever this scores is
    what the pipeline produces from noise — the floor against which every other
    row should be read.
    """

    task = "binary"
    name = "baseline_shuffled"

    def __init__(self, inner: AlphaModel, *, seed: int) -> None:
        """Wrap an estimator.

        Args:
            inner: The model to train on shuffled labels.
            seed: Run seed; the permutation is drawn from its own stream.
        """
        self.inner = inner
        self.seed = seed

    def __repr__(self) -> str:
        """Return a debug representation naming the wrapped model."""
        return f"ShuffledLabelModel({self.inner!r}, seed={self.seed})"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: NDArray[np.float64] | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> ShuffledLabelModel:
        """Shuffle the training labels, then fit the wrapped model.

        The shuffle happens here, inside the fold, on training labels only.
        Permuting before the split would let the permutation itself carry
        information across the boundary.

        Args:
            X: Training features.
            y: Training labels, about to be permuted.
            sample_weight: Per-sample weights, permuted alongside the labels so
                each weight stays with its observation.
            eval_set: Passed through.

        Returns:
            Self, fitted.
        """
        labels = as_label_series(y, X.index)
        rng = deterministic_rng(self.seed, "shuffled_labels", len(labels))
        order = rng.permutation(len(labels))
        shuffled = pd.Series(labels.to_numpy()[order], index=labels.index, name=labels.name)
        weights = None if sample_weight is None else sample_weight[order]
        self.inner.fit(X, shuffled, sample_weight=weights, eval_set=eval_set)
        return self

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Delegate to the wrapped model.

        Args:
            X: Feature frame.

        Returns:
            An `(n, 2)` array.
        """
        return self.inner.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> NDArray[np.int_]:
        """Delegate to the wrapped model.

        Args:
            X: Feature frame.

        Returns:
            An `(n,)` integer array.
        """
        return self.inner.predict(X)

    def feature_importance(self) -> pd.Series | None:
        """Report the wrapped model's importance, which describes noise.

        Returns:
            None. Importances fitted to shuffled labels rank noise, and putting
            them on a chart invites someone to read meaning into them.
        """
        return None

    def save(self, path: Path) -> None:
        """Persist the wrapper and its inner model.

        Args:
            path: Destination file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> ShuffledLabelModel:
        """Load a persisted wrapper.

        Args:
            path: Source file.

        Returns:
            The restored model.

        Raises:
            ModelError: If the file holds something else.
        """
        obj = pickle.loads(path.read_bytes())
        if not isinstance(obj, ShuffledLabelModel):
            msg = f"{path} does not contain a shuffled-label model."
            raise ModelError(msg)
        return obj
