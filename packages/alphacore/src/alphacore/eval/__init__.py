"""Evaluation metrics (§11).

Prediction quality, risk-adjusted returns, and the honesty metrics — PSR, DSR,
minimum track record length, bootstrap intervals, and the probability of
backtest overfitting. The last group is the one that makes the rest worth
reading.
"""

from __future__ import annotations

from alphacore.eval.bootstrap import (
    block_bootstrap_ci,
    politis_white_block_length,
    stationary_bootstrap_sample,
)
from alphacore.eval.classification import (
    DEFAULT_THRESHOLD,
    accuracy,
    brier_score,
    classification_metrics,
    information_coefficient,
    log_loss_score,
    roc_auc,
)
from alphacore.eval.deflated import (
    EULER_MASCHERONI,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)
from alphacore.eval.financial import (
    annualised_return,
    annualised_volatility,
    break_even_cost_bps,
    calmar_ratio,
    cvar,
    drawdown_series,
    information_ratio,
    max_drawdown,
    max_drawdown_duration,
    net_sharpe,
    return_metrics,
    rolling_sharpe,
    sharpe_ratio,
    sortino_ratio,
    trade_metrics,
    ulcer_index,
    var,
)
from alphacore.eval.pbo import DEFAULT_CHUNKS, cscv_pbo

__all__ = [
    "DEFAULT_CHUNKS",
    "DEFAULT_THRESHOLD",
    "EULER_MASCHERONI",
    "accuracy",
    "annualised_return",
    "annualised_volatility",
    "block_bootstrap_ci",
    "break_even_cost_bps",
    "brier_score",
    "calmar_ratio",
    "classification_metrics",
    "cscv_pbo",
    "cvar",
    "deflated_sharpe_ratio",
    "drawdown_series",
    "expected_max_sharpe",
    "information_coefficient",
    "information_ratio",
    "log_loss_score",
    "max_drawdown",
    "max_drawdown_duration",
    "min_track_record_length",
    "net_sharpe",
    "politis_white_block_length",
    "probabilistic_sharpe_ratio",
    "return_metrics",
    "roc_auc",
    "rolling_sharpe",
    "sharpe_ratio",
    "sortino_ratio",
    "stationary_bootstrap_sample",
    "trade_metrics",
    "ulcer_index",
    "var",
]
