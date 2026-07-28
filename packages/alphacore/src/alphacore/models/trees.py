"""Gradient-boosted tree adapters (§10.1).

LightGBM, XGBoost and CatBoost behind the same interface. All three are
imported lazily and live in the optional `gbm` extra, for the same reason torch
lives in `[dl]` (ADR 0003): `import alphacore` should not pull three compiled
boosting libraries into the path of a process that only wants to read a
tearsheet.

Defaults are deliberately conservative — shallow trees, strong subsampling,
few leaves. A GBM at library defaults will fit noise in a 2,000-row financial
sample with enthusiasm, and the resulting in-sample accuracy is exactly the
number §19 warns about. Tuning in Phase 10 searches from here, against an
out-of-fold economic objective.

Determinism (§4, I8) needs more than a seed: every library gets a fixed thread
count, because their histogram reductions are order-dependent across threads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from alphacore.errors import ModelError
from alphacore.models.base import FittedModel, make_preprocessor, resolve_params

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["make_catboost", "make_lightgbm", "make_xgboost"]

N_JOBS: Final[int] = 1
"""Single-threaded on purpose. Multi-threaded histogram building sums
floating-point gradients in a nondeterministic order, so the same seed gives
different trees run to run — which breaks byte-identical artifacts."""

_DEFAULTS: Final[dict[str, Any]] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample": 0.8,
}
"""The shared vocabulary. Names are the registry's, not any one library's:
CatBoost calls these `iterations`, `depth` and `rsm`, and a search space that
had to know which library it landed on would have to be rewritten to move a
parameter between them (§10.4)."""

_LIGHTGBM_DEFAULTS: Final[dict[str, Any]] = {
    **_DEFAULTS,
    "num_leaves": None,
    "min_child_samples": 20,
}
"""`num_leaves` defaults to `2 ** max_depth` — expressed as None here, because
a literal default would silently stop tracking `max_depth`."""

_CATBOOST_DEFAULTS: Final[dict[str, Any]] = {
    key: value for key, value in _DEFAULTS.items() if key != "subsample"
}
"""CatBoost's own subsampling is bound up with its bootstrap type, and the
adapter does not set it. Rejecting `subsample` is better than accepting it and
doing nothing: a tuning study that reports a best `subsample` the model never
applied is worse than one that could not run."""


def _require(module: str, extra: str = "gbm") -> Any:  # noqa: ANN401
    """Import an optional boosting library, or explain how to install it.

    Args:
        module: Module name.
        extra: The optional extra that provides it.

    Returns:
        The imported module.

    Raises:
        ModelError: If it is not installed.
    """
    import importlib  # noqa: PLC0415

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        msg = (
            f"{module} is not installed. It ships in the optional '{extra}' extra: "
            f"run `uv sync --all-packages --extra {extra}`."
        )
        raise ModelError(msg) from exc


def make_lightgbm(
    *, seed: int, calibrate: str = "none", params: Mapping[str, Any] | None = None
) -> FittedModel:
    """Build a LightGBM classifier.

    Args:
        seed: Random seed.
        calibrate: Calibration method.
        params: Overrides merged onto the conservative defaults.

    Returns:
        The model, unfitted.

    Raises:
        ModelError: If LightGBM is not installed, or a parameter is not one this
            model has.
    """
    lgb = _require("lightgbm")
    settings = resolve_params("lightgbm", _LIGHTGBM_DEFAULTS, params)
    leaves = settings["num_leaves"] or 2 ** settings["max_depth"]
    return FittedModel(
        "lightgbm",
        lgb.LGBMClassifier(
            n_estimators=settings["n_estimators"],
            learning_rate=settings["learning_rate"],
            max_depth=settings["max_depth"],
            num_leaves=leaves,
            min_child_samples=settings["min_child_samples"],
            subsample=settings["subsample"],
            subsample_freq=1,
            colsample_bytree=settings["colsample"],
            random_state=seed,
            n_jobs=N_JOBS,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        ),
        preprocessor=make_preprocessor(scale=False),
        calibrate=calibrate,
    )


def make_xgboost(
    *, seed: int, calibrate: str = "none", params: Mapping[str, Any] | None = None
) -> FittedModel:
    """Build an XGBoost classifier.

    Args:
        seed: Random seed.
        calibrate: Calibration method.
        params: Overrides merged onto the conservative defaults.

    Returns:
        The model, unfitted.

    Raises:
        ModelError: If XGBoost is not installed, or a parameter is not one this
            model has.
    """
    xgb = _require("xgboost")
    settings = resolve_params("xgboost", _DEFAULTS, params)
    return FittedModel(
        "xgboost",
        xgb.XGBClassifier(
            n_estimators=settings["n_estimators"],
            learning_rate=settings["learning_rate"],
            max_depth=settings["max_depth"],
            subsample=settings["subsample"],
            colsample_bytree=settings["colsample"],
            random_state=seed,
            n_jobs=N_JOBS,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0,
        ),
        preprocessor=make_preprocessor(scale=False),
        calibrate=calibrate,
    )


def make_catboost(
    *, seed: int, calibrate: str = "none", params: Mapping[str, Any] | None = None
) -> FittedModel:
    """Build a CatBoost classifier.

    Args:
        seed: Random seed.
        calibrate: Calibration method.
        params: Overrides merged onto the conservative defaults.

    Returns:
        The model, unfitted.

    Raises:
        ModelError: If CatBoost is not installed, or a parameter is not one this
            model has.
    """
    catboost = _require("catboost")
    settings = resolve_params("catboost", _CATBOOST_DEFAULTS, params)
    return FittedModel(
        "catboost",
        catboost.CatBoostClassifier(
            iterations=settings["n_estimators"],
            learning_rate=settings["learning_rate"],
            depth=settings["max_depth"],
            rsm=settings["colsample"],
            random_seed=seed,
            thread_count=N_JOBS,
            allow_writing_files=False,
            verbose=False,
        ),
        preprocessor=make_preprocessor(scale=False),
        calibrate=calibrate,
    )
