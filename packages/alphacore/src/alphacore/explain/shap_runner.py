"""SHAP explanations for tree models (§9).

SHAP values answer "why this prediction?" with an additive decomposition:

```
sum(shap_values) + base_value = raw_model_output
```

That identity is not decoration — it is the property that makes the numbers
trustworthy, and `verify_additivity` checks it rather than assuming it. §15's
Phase 9 exit criterion is exactly this.

**Computed by the boosting libraries, not the `shap` package.** LightGBM,
XGBoost and CatBoost each implement exact TreeSHAP in C++ and expose it
directly — `pred_contrib`, `pred_contribs`, and `ShapValues` respectively. Using
those gives the same algorithm, agrees to ~1e-15 rather than the `shap`
package's ~1e-4, and avoids a dependency chain (`shap` → `numba` → `llvmlite`)
that does not currently resolve on Python 3.12 with NumPy 2.5. See PROGRESS.md.

**Contributions are in raw-score space** — log-odds for a binary classifier,
not probability. The additivity identity only holds there, because the sigmoid
is not additive. `raw_output` is returned alongside so the check compares like
with like.

**Sampled, not exhaustive.** Exact SHAP over a full out-of-fold set is minutes
of compute for a chart nobody reads at that resolution. The sample is seeded
and its size recorded, so the plot can say what it was computed on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from alphacore.errors import ModelError
from alphacore.runtime.determinism import deterministic_rng
from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    from alphacore.models.protocol import AlphaModel

__all__ = [
    "ADDITIVITY_TOLERANCE",
    "DEFAULT_SAMPLE",
    "SHAP_SUMMARY_FILENAME",
    "ShapResult",
    "explain_prediction",
    "shap_summary",
    "verify_additivity",
]

DEFAULT_SAMPLE: Final[int] = 2_000
SHAP_SUMMARY_FILENAME: Final[str] = "shap_summary.parquet"

ADDITIVITY_TOLERANCE: Final[float] = 1e-6
"""How far `sum(shap) + base` may sit from the raw model output.

Native TreeSHAP is exact and lands around 1e-15; the slack here is for float32
accumulation inside the boosting libraries, not for a wrong decomposition. It
is deliberately far tighter than the 1e-4 the `shap` package needs."""


@dataclass(frozen=True)
class ShapResult:
    """SHAP values for one sampled explanation set.

    Attributes:
        values: `(n_samples, n_features)` contributions, in raw-score space.
        base_value: The model's expected raw output, `E[f(x)]`.
        features: The transformed feature values explained.
        raw_output: The model's raw output for the same rows.
        model: Name of the explained model.
        n_available: Rows the sample was drawn from, so a plot can say what
            fraction it represents.
    """

    values: pd.DataFrame
    base_value: float
    features: pd.DataFrame
    raw_output: np.ndarray
    model: str
    n_available: int

    @property
    def mean_abs(self) -> pd.Series:
        """Mean absolute contribution per feature, descending."""
        return self.values.abs().mean().sort_values(ascending=False)

    def summary_frame(self) -> pd.DataFrame:
        """Build the summary written to `shap_summary.parquet`.

        Returns:
            One row per feature: mean absolute contribution, mean signed
            contribution, and the correlation between the feature's value and
            its own contribution — which is what says whether high values push
            the prediction up or down, and is the colour axis of a beeswarm.
        """
        rows = []
        for feature in self.values.columns:
            contribution = self.values[feature].to_numpy(dtype=float)
            values = self.features[feature].to_numpy(dtype=float)
            correlation = (
                float(np.corrcoef(values, contribution)[0, 1])
                if np.ptp(values) > 0 and np.ptp(contribution) > 0
                else 0.0
            )
            rows.append(
                {
                    "feature": str(feature),
                    "mean_abs_shap": float(np.abs(contribution).mean()),
                    "mean_shap": float(contribution.mean()),
                    "direction": correlation,
                }
            )
        frame = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
        return frame.set_index("feature")

    def top_features(self, n: int = 10) -> tuple[str, ...]:
        """The `n` most influential features, most influential first.

        Args:
            n: How many to return.

        Returns:
            Feature names.
        """
        return tuple(str(name) for name in self.mean_abs.head(n).index)


def _estimator(model: AlphaModel) -> Any:  # noqa: ANN401
    """Extract the fitted estimator from a wrapped model.

    Args:
        model: An alphacore model.

    Returns:
        The estimator.

    Raises:
        ModelError: If the model is unfitted or has no estimator.
    """
    pipeline = getattr(model, "_pipeline", None)
    if pipeline is None:
        msg = (
            f"Model {getattr(model, 'name', model)!r} is not fitted, or is a baseline with "
            "no estimator to explain."
        )
        raise ModelError(msg)
    return pipeline.named_steps.get("estimator")


def _transform(model: AlphaModel, features: pd.DataFrame) -> pd.DataFrame:
    """Apply the model's preprocessing, so SHAP sees what the estimator saw.

    Args:
        model: The fitted model.
        features: Raw features.

    Returns:
        The transformed frame, with column names preserved.
    """
    pipeline = getattr(model, "_pipeline", None)
    if pipeline is None:  # pragma: no cover - guarded by _estimator
        return features.copy()
    transformed: Any = features
    for _, step in pipeline.steps[:-1]:
        transformed = step.transform(transformed)
    return pd.DataFrame(
        np.asarray(transformed, dtype=float), index=features.index, columns=features.columns
    )


def _contributions(estimator: Any, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401
    """Compute exact TreeSHAP contributions using the library's own routine.

    Args:
        estimator: A fitted LightGBM, XGBoost or CatBoost classifier.
        data: The transformed feature frame.

    Returns:
        `(contributions, base_values)` where contributions is
        `(n_samples, n_features)` and base is `(n_samples,)`.

    Raises:
        ModelError: If the estimator is not a supported tree model. A linear
            model's coefficients are already its explanation, and forcing
            TreeSHAP onto a baseline would produce numbers that look like
            explanations and are not.
    """
    kind = type(estimator).__name__

    if hasattr(estimator, "booster_"):  # LightGBM
        raw = np.asarray(estimator.predict(data, pred_contrib=True), dtype=float)
    elif hasattr(estimator, "get_booster"):  # XGBoost
        import xgboost as xgb  # noqa: PLC0415 - only on the explain path

        booster = estimator.get_booster()
        raw = np.asarray(booster.predict(xgb.DMatrix(data), pred_contribs=True), dtype=float)
    elif hasattr(estimator, "get_feature_importance"):  # CatBoost
        import catboost  # noqa: PLC0415 - only on the explain path

        raw = np.asarray(
            estimator.get_feature_importance(catboost.Pool(data), type="ShapValues"),
            dtype=float,
        )
    else:
        msg = (
            f"{kind} is not a tree model TreeSHAP can explain. §9 explains tree models; "
            "a linear model's coefficients are already its explanation, and a baseline "
            "has nothing to decompose."
        )
        raise ModelError(msg)

    if raw.ndim == 3:  # noqa: PLR2004 - some versions return one slab per class
        raw = raw[:, -1, :]
    return raw[:, :-1], raw[:, -1]


def _raw_scores(estimator: Any, data: pd.DataFrame) -> np.ndarray:  # noqa: ANN401
    """Model output on the same scale the contributions are expressed in.

    Args:
        estimator: The fitted estimator.
        data: The transformed feature frame.

    Returns:
        Raw scores, normally log-odds.

    Raises:
        ModelError: If raw scores cannot be obtained.
    """
    if hasattr(estimator, "booster_"):
        return np.asarray(estimator.predict(data, raw_score=True), dtype=float)
    if hasattr(estimator, "get_booster"):
        import xgboost as xgb  # noqa: PLC0415

        return np.asarray(
            estimator.get_booster().predict(xgb.DMatrix(data), output_margin=True), dtype=float
        )
    if hasattr(estimator, "get_feature_importance"):
        return np.asarray(estimator.predict(data, prediction_type="RawFormulaVal"), dtype=float)
    msg = f"Cannot obtain raw scores from {type(estimator).__name__}."  # pragma: no cover
    raise ModelError(msg)


def shap_summary(
    model: AlphaModel,
    features: pd.DataFrame,
    *,
    seed: int,
    sample: int = DEFAULT_SAMPLE,
) -> ShapResult:
    """Compute SHAP values for a sampled subset of an out-of-fold set.

    Args:
        model: A fitted tree model.
        features: The feature frame to explain.
        seed: Run seed. The sample is drawn from its own stream, so adding an
            explanation cannot perturb any other random draw (§4, I8).
        sample: Maximum rows to explain.

    Returns:
        The SHAP result.

    Raises:
        ModelError: If the model is not an explainable tree model, or the
            feature frame is empty.
    """
    if features.empty:
        msg = "Cannot compute SHAP values on an empty feature frame."
        raise ModelError(msg)

    estimator = _estimator(model)
    if estimator is None:  # pragma: no cover - guarded by _estimator
        msg = "Model has no estimator to explain."
        raise ModelError(msg)

    if len(features) > sample:
        rng = deterministic_rng(seed, "shap_sample", sample)
        picks = np.sort(rng.choice(len(features), size=sample, replace=False))
        subset = features.iloc[picks]
    else:
        subset = features

    transformed = _transform(model, subset)
    contributions, base_values = _contributions(estimator, transformed)
    raw_output = _raw_scores(estimator, transformed)

    logger.bind(phase="explain").info(
        "SHAP: explained {} of {} rows for {}",
        len(subset),
        len(features),
        getattr(model, "name", "model"),
    )
    return ShapResult(
        values=pd.DataFrame(contributions, index=subset.index, columns=subset.columns),
        base_value=float(np.mean(base_values)),
        features=transformed,
        raw_output=raw_output,
        model=str(getattr(model, "name", "model")),
        n_available=len(features),
    )


def verify_additivity(result: ShapResult, *, tolerance: float = ADDITIVITY_TOLERANCE) -> float:
    """Check `sum(shap) + base == f(x)` (§15, Phase 9 exit).

    Args:
        result: The SHAP result.
        tolerance: Maximum absolute discrepancy.

    Returns:
        The largest absolute discrepancy observed.

    Raises:
        ModelError: If the identity fails. A SHAP plot whose contributions do
            not reconstruct the prediction is not an explanation of anything,
            and shipping one would be worse than shipping no explanation.
    """
    reconstructed = result.values.to_numpy(dtype=float).sum(axis=1) + result.base_value
    discrepancy = float(np.abs(reconstructed - result.raw_output).max())
    if discrepancy > tolerance:
        msg = (
            f"SHAP additivity failed for {result.model!r}: contributions reconstruct the raw "
            f"model output only to within {discrepancy:.2e}, above the {tolerance:.0e} "
            "tolerance. The decomposition does not explain the prediction it claims to."
        )
        raise ModelError(msg)
    return discrepancy


def explain_prediction(
    result: ShapResult, timestamp: pd.Timestamp, *, n: int = 10
) -> dict[str, Any]:
    """Build the waterfall data for a single prediction (§13.4).

    Args:
        result: The SHAP result.
        timestamp: Which row to explain.
        n: How many contributions to return.

    Returns:
        A mapping with the base value, the top contributions, and the
        reconstructed output — the last so the UI can *display* the additivity
        identity rather than ask the reader to trust it.

    Raises:
        ModelError: If the timestamp is not in the explained sample.
    """
    if timestamp not in result.values.index:
        msg = (
            f"{timestamp} is not in the explained sample. SHAP was computed on a seeded "
            f"subset of {len(result.values)} of {result.n_available} rows; ask for one of those."
        )
        raise ModelError(msg)

    row = result.values.loc[timestamp]
    ordered = row.reindex(row.abs().sort_values(ascending=False).index).head(n)
    return {
        "ts": timestamp.isoformat(),
        "base_value": result.base_value,
        "output": float(row.sum() + result.base_value),
        "contributions": [
            {
                "feature": str(feature),
                "shap": float(value),
                "feature_value": float(result.features.loc[timestamp, feature]),
            }
            for feature, value in ordered.items()
        ],
    }
