"""Shared model machinery: the sklearn pipeline and calibration (§10.1, §10.2).

Two invariants are enforced here rather than in each adapter, because doing it
per-adapter is doing it four times and forgetting once.

**I3 — fit only on train.** Every estimator sits behind an sklearn `Pipeline`
whose first steps are a winsoriser and a scaler. Those are fitted inside
`fit`, which only ever sees training rows, so there is no path where a scaler
learns the test period's mean. There are no pre-scaled global matrices anywhere
in this codebase.

**§10.2 — calibrate on an inner split.** Raw gradient-boosted probabilities are
not probabilities; they are scores that happen to lie in `[0, 1]`. The
calibrator is fitted on a held-out *tail* of the training fold, never on the
test fold. The tail rather than a random subset, because a random inner split
of a time series has the same overlap problem as a random outer one.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alphacore.errors import ModelError
from alphacore.models.protocol import Task

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

__all__ = [
    "CALIBRATION_METHODS",
    "FittedModel",
    "Winsoriser",
    "apply_calibrator",
    "as_label_series",
    "fit_calibrator",
    "make_preprocessor",
]

CALIBRATION_METHODS: Final[tuple[str, ...]] = ("none", "sigmoid", "isotonic")

_INNER_SPLIT_FRACTION: Final[float] = 0.25
_MIN_INNER_ROWS: Final[int] = 30
_N_CLASSES: Final[int] = 2


def as_label_series(y: Any, index: pd.Index) -> pd.Series:  # noqa: ANN401
    """Coerce labels to a Series aligned to the feature index.

    Callers legitimately pass either a Series or a bare array. Positional
    slicing inside `fit` needs the former, so the coercion happens once here
    rather than being assumed in five places.

    Args:
        y: Labels, as a Series or an array.
        index: The feature frame's index to align to.

    Returns:
        A Series carrying `index`.

    Raises:
        ModelError: If the label count does not match the index.
    """
    if isinstance(y, pd.Series):
        return y
    values = np.asarray(y)
    if values.shape[0] != len(index):
        msg = f"Got {values.shape[0]} labels for {len(index)} feature rows."
        raise ModelError(msg)
    return pd.Series(values, index=index, name="label")


class Winsoriser(BaseEstimator, TransformerMixin):
    """Clip every column to quantiles learned on the training rows (§7.3).

    Fitted, not global. A winsoriser whose bounds came from the whole sample
    would encode the test period's tail behaviour into training data.
    """

    def __init__(self, quantile: float = 0.001) -> None:
        """Configure the clipping quantile.

        Args:
            quantile: Tail fraction clipped at each end. Zero disables it.
        """
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y: Any = None) -> Winsoriser:  # noqa: ANN401, ARG002
        """Learn the clipping bounds from the training rows.

        Args:
            X: Training features.
            y: Ignored; present for the sklearn interface.

        Returns:
            Self, fitted.
        """
        values = np.asarray(X, dtype=float)
        if self.quantile <= 0:
            self.lower_ = np.full(values.shape[1], -np.inf)
            self.upper_ = np.full(values.shape[1], np.inf)
        else:
            self.lower_ = np.nanquantile(values, self.quantile, axis=0)
            self.upper_ = np.nanquantile(values, 1.0 - self.quantile, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Clip to the learned bounds.

        Args:
            X: Features to transform.

        Returns:
            The clipped array.

        Raises:
            ModelError: If called before `fit`.
        """
        if not hasattr(self, "lower_"):
            msg = "Winsoriser.transform called before fit."
            raise ModelError(msg)
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def make_preprocessor(
    *, winsorise_quantile: float = 0.001, scale: bool = True
) -> list[tuple[str, Any]]:
    """Build the preprocessing steps every model shares.

    Args:
        winsorise_quantile: Tail fraction clipped at each end.
        scale: Whether to standardise. Trees do not need it; linear models do.

    Returns:
        sklearn pipeline steps, in order.
    """
    steps: list[tuple[str, Any]] = [("winsorise", Winsoriser(winsorise_quantile))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


def fit_calibrator(method: str, raw: NDArray[np.float64], y: pd.Series) -> Any:  # noqa: ANN401
    """Fit a probability calibrator on held-out scores (§10.2).

    Shared with the sequence models, which hold out the same kind of inner tail
    for early stopping and calibrate on it rather than growing a second one.

    Args:
        method: `sigmoid` or `isotonic`. `none` returns nothing to apply.
        raw: Uncalibrated positive-class scores from the inner split.
        y: The inner split's labels.

    Returns:
        The fitted calibrator, or None when the method is `none` or the inner
        split turned out to be single-class — a calibrator fitted on one class
        maps everything to that class.
    """
    if method == "none" or len(np.unique(y)) < _N_CLASSES:
        return None
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw, y)
        return calibrator
    calibrator = LogisticRegression(max_iter=1_000)
    calibrator.fit(np.asarray(raw, dtype=float).reshape(-1, 1), y)
    return calibrator


def apply_calibrator(calibrator: Any, raw: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: ANN401
    """Map raw scores through a fitted calibrator.

    Args:
        calibrator: The object returned by `fit_calibrator`.
        raw: Uncalibrated positive-class scores.

    Returns:
        Calibrated probabilities, or `raw` unchanged if there is no calibrator.
    """
    if calibrator is None:
        return np.asarray(raw, dtype=float)
    if isinstance(calibrator, IsotonicRegression):
        return np.asarray(calibrator.predict(raw), dtype=float)
    return np.asarray(
        calibrator.predict_proba(np.asarray(raw, dtype=float).reshape(-1, 1))[:, 1], dtype=float
    )


class FittedModel:
    """Base class implementing `AlphaModel` over an sklearn-style estimator."""

    task: Task = "binary"

    def __init__(
        self,
        name: str,
        estimator: Any,  # noqa: ANN401 - any sklearn-compatible classifier
        *,
        preprocessor: list[tuple[str, Any]] | None = None,
        calibrate: str = "none",
        supports_sample_weight: bool = True,
    ) -> None:
        """Wrap an estimator in the shared pipeline.

        Args:
            name: Model name, as it appears on the leaderboard.
            estimator: An sklearn-compatible classifier.
            preprocessor: Steps prepended to the estimator.
            calibrate: One of `none`, `sigmoid`, `isotonic`.
            supports_sample_weight: Whether the estimator accepts weights.

        Raises:
            ModelError: If the calibration method is unknown.
        """
        if calibrate not in CALIBRATION_METHODS:
            msg = (
                f"Unknown calibration method {calibrate!r} for model {name!r}. "
                f"Known: {list(CALIBRATION_METHODS)}."
            )
            raise ModelError(msg)
        self.name = name
        self.calibrate = calibrate
        self.supports_sample_weight = supports_sample_weight
        self._steps = (preprocessor or []) + [("estimator", estimator)]
        self._pipeline: Pipeline | None = None
        self._calibrator: Any = None
        self._columns: list[str] = []

    def __repr__(self) -> str:
        """Return a debug representation naming the model and calibration."""
        return f"{type(self).__name__}({self.name!r}, calibrate={self.calibrate!r})"

    def _require_fitted(self) -> Pipeline:
        """Return the fitted pipeline.

        Returns:
            The pipeline.

        Raises:
            ModelError: If the model has not been fitted.
        """
        if self._pipeline is None:
            msg = f"Model {self.name!r} must be fitted before it can predict."
            raise ModelError(msg)
        return self._pipeline

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        sample_weight: NDArray[np.float64] | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,  # noqa: ARG002
    ) -> FittedModel:
        """Fit the pipeline, and the calibrator on an inner tail split.

        Args:
            X: Training features.
            y: Training labels.
            sample_weight: Per-sample weights (§8.2).
            eval_set: Unused by the sklearn path; tree adapters override.

        Returns:
            Self, fitted.

        Raises:
            ModelError: If `X` and `y` disagree in length, or the fold has only
                one class.
        """
        if len(X) != len(y):
            msg = f"Model {self.name!r} got {len(X)} rows of features for {len(y)} labels."
            raise ModelError(msg)
        y = as_label_series(y, X.index)
        self._columns = [str(c) for c in X.columns]

        if self.calibrate == "none":
            self._pipeline = self._fit_pipeline(X, y, sample_weight)
            self._calibrator = None
            return self

        inner = max(_MIN_INNER_ROWS, int(len(X) * _INNER_SPLIT_FRACTION))
        if len(X) <= inner + _MIN_INNER_ROWS:
            # Too little data to hold anything back; fitting the calibrator on
            # the same rows as the model would make it worse than useless.
            self._pipeline = self._fit_pipeline(X, y, sample_weight)
            self._calibrator = None
            return self

        cut = len(X) - inner
        weights = None if sample_weight is None else sample_weight[:cut]
        self._pipeline = self._fit_pipeline(X.iloc[:cut], y.iloc[:cut], weights)
        self._calibrator = self._fit_calibrator(X.iloc[cut:], y.iloc[cut:])
        return self

    def _fit_pipeline(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: NDArray[np.float64] | None
    ) -> Pipeline:
        """Fit the preprocessing-plus-estimator pipeline.

        Args:
            X: Training features.
            y: Training labels.
            sample_weight: Per-sample weights.

        Returns:
            The fitted pipeline.

        Raises:
            ModelError: If the estimator refuses the data.
        """
        pipeline = Pipeline(self._steps)
        kwargs = (
            {"estimator__sample_weight": sample_weight}
            if sample_weight is not None and self.supports_sample_weight
            else {}
        )
        try:
            pipeline.fit(X, y, **kwargs)
        except Exception as exc:
            msg = f"Model {self.name!r} failed to fit: {exc}"
            raise ModelError(msg) from exc
        return pipeline

    def _fit_calibrator(self, X: pd.DataFrame, y: pd.Series) -> Any:  # noqa: ANN401
        """Fit the probability calibrator on the held-out inner tail.

        Args:
            X: Inner validation features.
            y: Inner validation labels.

        Returns:
            The fitted calibrator, or None if the inner split is single-class.
        """
        assert self._pipeline is not None
        raw = np.asarray(self._pipeline.predict_proba(X), dtype=float)[:, 1]
        return fit_calibrator(self.calibrate, raw, y)

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Predict calibrated class probabilities.

        Args:
            X: Feature frame.

        Returns:
            An `(n, 2)` array whose rows sum to 1.

        Raises:
            ModelError: If the model has not been fitted.
        """
        pipeline = self._require_fitted()
        raw = np.asarray(pipeline.predict_proba(X), dtype=float)[:, 1]
        positive = np.clip(apply_calibrator(self._calibrator, raw), 0.0, 1.0)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: pd.DataFrame) -> NDArray[np.int_]:
        """Predict class labels at a 0.5 threshold.

        Args:
            X: Feature frame.

        Returns:
            An `(n,)` integer array.

        Raises:
            ModelError: If the model has not been fitted.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)  # noqa: PLR2004

    def feature_importance(self) -> pd.Series | None:
        """Report per-feature importance if the estimator exposes one.

        Returns:
            A Series indexed by feature name, or None.
        """
        pipeline = self._require_fitted()
        estimator = pipeline.named_steps["estimator"]
        values = getattr(estimator, "feature_importances_", None)
        if values is None:
            coef = getattr(estimator, "coef_", None)
            if coef is None:
                return None
            values = np.abs(np.asarray(coef, dtype=float)).ravel()
        values = np.asarray(values, dtype=float).ravel()
        if values.size != len(self._columns):  # pragma: no cover - defensive
            return None
        return pd.Series(values, index=self._columns, name=self.name).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        """Persist the fitted model with pickle.

        Args:
            path: Destination file.

        Raises:
            ModelError: If the model has not been fitted.
        """
        self._require_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> FittedModel:
        """Load a persisted model.

        Args:
            path: Source file, written by `save`.

        Returns:
            The restored model.

        Raises:
            ModelError: If the file does not contain a model of this kind.
        """
        obj = pickle.loads(path.read_bytes())
        if not isinstance(obj, FittedModel):
            msg = f"{path} does not contain an alphacore model."
            raise ModelError(msg)
        return obj
