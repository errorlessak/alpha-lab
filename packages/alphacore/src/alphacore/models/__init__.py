"""Models, baselines, and the out-of-fold leaderboard (§10)."""

from __future__ import annotations

from alphacore.models.base import CALIBRATION_METHODS, FittedModel, Winsoriser, make_preprocessor
from alphacore.models.baselines import (
    MANDATORY_BASELINES,
    AlwaysLong,
    BuyAndHold,
    MomentumRule,
    RandomSign,
    ShuffledLabelModel,
)
from alphacore.models.leaderboard import (
    ECONOMIC_COLUMNS,
    LeaderboardRow,
    build_leaderboard,
    leaderboard_row,
    validate_leaderboard,
)
from alphacore.models.linear import make_elasticnet, make_logistic
from alphacore.models.oof import OOFPredictions, fit_predict_oof
from alphacore.models.protocol import AlphaModel, Task
from alphacore.models.registry import build_model, registered_models
from alphacore.models.trees import make_catboost, make_lightgbm, make_xgboost

__all__ = [
    "CALIBRATION_METHODS",
    "ECONOMIC_COLUMNS",
    "MANDATORY_BASELINES",
    "AlphaModel",
    "AlwaysLong",
    "BuyAndHold",
    "FittedModel",
    "LeaderboardRow",
    "MomentumRule",
    "OOFPredictions",
    "RandomSign",
    "ShuffledLabelModel",
    "Task",
    "Winsoriser",
    "build_leaderboard",
    "build_model",
    "fit_predict_oof",
    "leaderboard_row",
    "make_catboost",
    "make_elasticnet",
    "make_lightgbm",
    "make_logistic",
    "make_preprocessor",
    "make_xgboost",
    "registered_models",
    "validate_leaderboard",
]
