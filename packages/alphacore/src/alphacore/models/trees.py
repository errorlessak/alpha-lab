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
from alphacore.models.base import FittedModel, make_preprocessor

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
        ModelError: If LightGBM is not installed.
    """
    lgb = _require("lightgbm")
    settings = {**_DEFAULTS, **(params or {})}
    return FittedModel(
        "lightgbm",
        lgb.LGBMClassifier(
            n_estimators=settings["n_estimators"],
            learning_rate=settings["learning_rate"],
            max_depth=settings["max_depth"],
            num_leaves=2 ** settings["max_depth"],
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
        ModelError: If XGBoost is not installed.
    """
    xgb = _require("xgboost")
    settings = {**_DEFAULTS, **(params or {})}
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
        ModelError: If CatBoost is not installed.
    """
    catboost = _require("catboost")
    settings = {**_DEFAULTS, **(params or {})}
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
