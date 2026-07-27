"""The model registry (§10.1).

Maps a config name to a constructed, unfitted model. Keeping the mapping in one
place means a config typo is caught with a list of what is available rather
than an `AttributeError` three modules deep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

from alphacore.errors import ModelError
from alphacore.models.baselines import (
    AlwaysLong,
    BuyAndHold,
    MomentumRule,
    RandomSign,
    ShuffledLabelModel,
)
from alphacore.models.linear import make_elasticnet, make_logistic
from alphacore.models.trees import make_catboost, make_lightgbm, make_xgboost

if TYPE_CHECKING:
    from collections.abc import Callable

    from alphacore.models.protocol import AlphaModel

__all__ = ["SHUFFLED_INNER_MODEL", "build_model", "registered_models"]

SHUFFLED_INNER_MODEL: Final[str] = "lightgbm"
"""Estimator the shuffled-label baseline wraps. A GBM rather than a linear
model on purpose: the point of the row is to show how much apparent edge a
*flexible* learner manufactures from noise, and a logistic regression
manufactures far less."""

_BUILDERS: Final[dict[str, Callable[..., Any]]] = {
    "logistic": make_logistic,
    "elasticnet": make_elasticnet,
    "lightgbm": make_lightgbm,
    "xgboost": make_xgboost,
    "catboost": make_catboost,
    "baseline_buyhold": lambda *, seed, **_: BuyAndHold(seed=seed),
    "baseline_always_long": lambda *, seed, **_: AlwaysLong(seed=seed),
    "baseline_random_sign": lambda *, seed, **_: RandomSign(seed=seed),
    "baseline_momentum": lambda *, seed, **_: MomentumRule(seed=seed),
}


def registered_models() -> tuple[str, ...]:
    """List the model names a config may use.

    Returns:
        Names, sorted.
    """
    return tuple(sorted([*_BUILDERS, "baseline_shuffled"]))


def build_model(name: str, *, seed: int, calibrate: str = "none", **params: Any) -> AlphaModel:  # noqa: ANN401
    """Construct an unfitted model by name.

    Args:
        name: Registry name from `models.candidates`.
        seed: Run seed.
        calibrate: Calibration method. Ignored by the baselines, which have no
            probabilities worth calibrating.
        **params: Extra keyword arguments forwarded to the builder.

    Returns:
        The unfitted model.

    Raises:
        ModelError: If the name is unknown, an optional dependency is missing,
            or the calibration method is invalid.
    """
    if name == "baseline_shuffled":
        inner = build_model(SHUFFLED_INNER_MODEL, seed=seed, calibrate=calibrate, **params)
        return cast("AlphaModel", ShuffledLabelModel(inner, seed=seed))

    builder = _BUILDERS.get(name)
    if builder is None:
        msg = f"Unknown model {name!r}. Registered: {list(registered_models())}."
        raise ModelError(msg)
    # The builders return concrete classes; conformance to the protocol is
    # asserted at runtime in the tests rather than inferred here.
    return cast("AlphaModel", builder(seed=seed, calibrate=calibrate, **params))
