"""Linear baselines (§10.1).

Always on the leaderboard, and the reason is worth stating plainly: **a linear
model that beats your gradient-boosted machine is important information**. It
usually means the GBM is fitting noise, that the feature set carries no
interaction worth learning, or that the sample is too small for the capacity
being thrown at it. Discovering that from a leaderboard is cheap; discovering
it after deploying the GBM is not.

Both go through the shared pipeline, so they are winsorised and standardised on
training rows only (§4, I3). Scaling matters more here than for trees — an
unscaled elastic net penalises whichever feature happens to have the largest
units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from sklearn.linear_model import LogisticRegression

from alphacore.models.base import FittedModel, make_preprocessor, resolve_params

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["make_elasticnet", "make_logistic"]

DEFAULT_MAX_ITER = 2_000

_LOGISTIC_DEFAULTS: Final[dict[str, Any]] = {"C": 1.0}
_ELASTICNET_DEFAULTS: Final[dict[str, Any]] = {"C": 1.0, "l1_ratio": 0.5}


def make_logistic(
    *, seed: int, calibrate: str = "none", params: Mapping[str, Any] | None = None
) -> FittedModel:
    """Build an L2-regularised logistic regression.

    Args:
        seed: Random seed, for the solver's tie-breaking.
        calibrate: Calibration method.
        params: Overrides merged onto the defaults. `C` is the inverse
            regularisation strength.

    Returns:
        The model, unfitted.

    Raises:
        ModelError: If a parameter is not one this model has.
    """
    settings = resolve_params("logistic", _LOGISTIC_DEFAULTS, params)
    return FittedModel(
        "logistic",
        LogisticRegression(C=settings["C"], max_iter=DEFAULT_MAX_ITER, random_state=seed),
        preprocessor=make_preprocessor(scale=True),
        calibrate=calibrate,
    )


def make_elasticnet(
    *, seed: int, calibrate: str = "none", params: Mapping[str, Any] | None = None
) -> FittedModel:
    """Build an elastic-net logistic regression.

    The L1 component drives coefficients to exactly zero, which on a feature
    set built from thirty correlated moving averages is a form of selection the
    clustered pruning in §7.3 does not replace.

    Args:
        seed: Random seed.
        calibrate: Calibration method.
        params: Overrides merged onto the defaults. `C` is the inverse
            regularisation strength; `l1_ratio` mixes L1 (1.0) with L2 (0.0).

    Returns:
        The model, unfitted.

    Raises:
        ModelError: If a parameter is not one this model has.
    """
    settings = resolve_params("elasticnet", _ELASTICNET_DEFAULTS, params)
    # scikit-learn 1.8 deprecated the `penalty` argument: the mix is expressed
    # by `l1_ratio` alone now, with 0 meaning pure L2 and 1 pure L1.
    return FittedModel(
        "elasticnet",
        LogisticRegression(
            solver="saga",
            l1_ratio=settings["l1_ratio"],
            C=settings["C"],
            max_iter=DEFAULT_MAX_ITER,
            random_state=seed,
        ),
        preprocessor=make_preprocessor(scale=True),
        calibrate=calibrate,
    )
