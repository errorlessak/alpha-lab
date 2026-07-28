"""Models, baselines, and the out-of-fold leaderboard (§10)."""

from __future__ import annotations

from alphacore.models.base import (
    CALIBRATION_METHODS,
    FittedModel,
    Winsoriser,
    apply_calibrator,
    as_label_series,
    fit_calibrator,
    make_preprocessor,
)
from alphacore.models.baselines import (
    MANDATORY_BASELINES,
    AlwaysLong,
    BuyAndHold,
    MomentumRule,
    RandomSign,
    ShuffledLabelModel,
)
from alphacore.models.ensemble import EnsembleModel
from alphacore.models.leaderboard import (
    ECONOMIC_COLUMNS,
    LeaderboardRow,
    attach_economics,
    build_leaderboard,
    leaderboard_row,
    score_models,
    validate_leaderboard,
)
from alphacore.models.linear import make_elasticnet, make_logistic
from alphacore.models.oof import OOFPredictions, fit_predict_oof
from alphacore.models.protocol import AlphaModel, Task
from alphacore.models.registry import (
    ENSEMBLE_LEGS,
    build_model,
    make_gbm_sequence_ensemble,
    registered_models,
)
from alphacore.models.sequence import (
    ARCHITECTURES,
    SEQUENCE_DEFAULTS,
    SequenceModel,
    make_bilstm,
    make_gru,
    make_lstm,
    make_stacked_lstm,
    make_tcn,
)
from alphacore.models.trees import make_catboost, make_lightgbm, make_xgboost
from alphacore.models.windows import build_windows, contiguous_runs, grid_positions

__all__ = [
    "ARCHITECTURES",
    "CALIBRATION_METHODS",
    "ECONOMIC_COLUMNS",
    "ENSEMBLE_LEGS",
    "MANDATORY_BASELINES",
    "SEQUENCE_DEFAULTS",
    "AlphaModel",
    "AlwaysLong",
    "BuyAndHold",
    "EnsembleModel",
    "FittedModel",
    "LeaderboardRow",
    "MomentumRule",
    "OOFPredictions",
    "RandomSign",
    "SequenceModel",
    "ShuffledLabelModel",
    "Task",
    "Winsoriser",
    "apply_calibrator",
    "as_label_series",
    "attach_economics",
    "build_leaderboard",
    "build_model",
    "build_windows",
    "contiguous_runs",
    "fit_calibrator",
    "fit_predict_oof",
    "grid_positions",
    "leaderboard_row",
    "make_bilstm",
    "make_catboost",
    "make_elasticnet",
    "make_gbm_sequence_ensemble",
    "make_gru",
    "make_lightgbm",
    "make_logistic",
    "make_lstm",
    "make_preprocessor",
    "make_stacked_lstm",
    "make_tcn",
    "make_xgboost",
    "registered_models",
    "score_models",
    "validate_leaderboard",
]
