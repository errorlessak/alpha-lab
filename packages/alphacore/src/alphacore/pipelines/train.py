"""The cross-validation and training stage (§9, §10, Phase 10).

Cuts purged folds, optionally runs the economic hyperparameter search, scores
every candidate and every mandatory baseline out of fold, then backtests each
one so the leaderboard's economic columns are filled from a costed run rather
than left empty.

The ordering matters and is not negotiable. Folds are cut **before** anything is
fitted, tuning is scored on **those** folds, and the backtest reads the **same**
out-of-fold probabilities the leaderboard row was scored from. A search that
picked its own splits, or a backtest that re-predicted, would each let the
result drift away from the thing being reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from alphacore.cv import make_splits
from alphacore.errors import AlphaError, PipelineError
from alphacore.models.leaderboard import attach_economics, score_models
from alphacore.pipelines.backtest import (
    BacktestOutcome,
    backtest_predictions,
    deflate_outcomes,
    economic_columns,
)
from alphacore.pipelines.tune import tune_candidates
from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from alphacore.config.schema import AlphaConfig
    from alphacore.cv import Split
    from alphacore.models.oof import OOFPredictions
    from alphacore.pipelines.build_features import FeatureSet
    from alphacore.pipelines.tune import TuningResult

__all__ = ["TrainingResult", "train_models"]


@dataclass(frozen=True)
class TrainingResult:
    """Everything the training and evaluation stages produced.

    Attributes:
        rows: Leaderboard rows, ranked, with economic columns filled wherever a
            backtest succeeded.
        predictions: Out-of-fold predictions by model name.
        outcomes: Backtest outcomes by model name.
        splits: The folds every model was scored on.
        tuning: Study results by model name, empty when tuning is disabled.
        n_trials_total: Configurations evaluated across the whole run. This is
            the `N` behind the deflated Sharpe (§11.2).
    """

    rows: list[dict[str, Any]]
    predictions: dict[str, OOFPredictions]
    outcomes: dict[str, BacktestOutcome]
    splits: list[Split]
    tuning: dict[str, TuningResult] = field(default_factory=dict)
    n_trials_total: int = 1

    @property
    def selected(self) -> str | None:
        """The top-ranked model, or None if the leaderboard is empty."""
        return str(self.rows[0]["model"]) if self.rows else None

    @property
    def selected_outcome(self) -> BacktestOutcome | None:
        """The backtest behind the top-ranked model, if it has one."""
        name = self.selected
        return self.outcomes.get(name) if name else None


def train_models(
    config: AlphaConfig,
    feature_set: FeatureSet,
    *,
    splits: Sequence[Split] | None = None,
) -> TrainingResult:
    """Cut folds, tune, score, and backtest every model.

    Args:
        config: The resolved configuration.
        feature_set: Aligned features, labels and bars.
        splits: Override the folds. Used by tests that need a smaller geometry
            than the config's; production passes None.

    Returns:
        The completed training result.

    Raises:
        CVError: If the fold geometry cannot produce a valid split.
        ModelError: If no model could be scored.
        PipelineError: If tuning is enabled without optuna installed.
    """
    folds = list(splits) if splits is not None else make_splits(feature_set.labels, config.cv)

    tuning = tune_candidates(feature_set, folds, config)
    tuned_params = {name: result.best_params for name, result in tuning.items()}
    n_trials_by_model = {name: result.n_trials for name, result in tuning.items()}

    rows, predictions = score_models(
        feature_set.features,
        feature_set.labels,
        folds,
        config.models,
        seed=config.meta.seed,
        prune=config.features.prune,
        n_trials_by_model=n_trials_by_model,
        params_by_model=tuned_params,
    )

    outcomes = _backtest_all(config, feature_set, predictions, tuning=tuning)
    rows = attach_economics(
        rows, {name: economic_columns(outcome) for name, outcome in outcomes.items()}
    )

    # The row's trial count must be the `N` its deflated Sharpe was computed
    # against, or the two columns describe different searches. Deflation counts
    # every configuration this run evaluated, not just the ones tried for this
    # model, so the count is copied back from the outcome rather than left at
    # whatever the tuning study reported per model.
    n_trials_total = max([outcome.n_trials for outcome in outcomes.values()] + [len(rows)])
    for row in rows:
        if row["model"] in outcomes:
            row["n_trials"] = outcomes[str(row["model"])].n_trials
    logger.bind(phase="train").info(
        "trained {} models over {} folds; {} configurations evaluated in total",
        len(rows),
        len(folds),
        n_trials_total,
    )
    return TrainingResult(
        rows=rows,
        predictions=predictions,
        outcomes=outcomes,
        splits=folds,
        tuning=tuning,
        n_trials_total=n_trials_total,
    )


def _backtest_all(
    config: AlphaConfig,
    feature_set: FeatureSet,
    predictions: dict[str, OOFPredictions],
    *,
    tuning: dict[str, TuningResult],
) -> dict[str, BacktestOutcome]:
    """Backtest every model that produced out-of-fold predictions, then deflate.

    The engine runs once per model. Deflation happens afterwards, when the
    spread of Sharpe ratios across every configuration this run evaluated —
    leaderboard models and discarded tuning trials alike — is finally known.

    Args:
        config: The resolved configuration.
        feature_set: Aligned features, labels and bars.
        predictions: Out-of-fold predictions by model name.
        tuning: Study results, whose trial values widen the observed spread.

    Returns:
        Outcomes by model name. A model whose backtest failed is absent, and
        the failure is logged rather than swallowed silently.

    Raises:
        PipelineError: If not one model could be backtested, which would leave
            the leaderboard with no economic columns at all.
    """
    outcomes: dict[str, BacktestOutcome] = {}
    for name, oof in sorted(predictions.items()):
        try:
            outcomes[name] = backtest_predictions(
                oof,
                feature_set.bars,
                config,
                periods_per_year=feature_set.periods_per_year,
                seed=config.meta.seed,
            )
        except AlphaError as exc:
            logger.bind(phase="backtest").warning("{} could not be backtested: {}", name, exc)

    if not outcomes:
        msg = (
            "No model could be backtested, so the leaderboard would have no economic "
            "columns at all. The signal or sizing configuration is rejecting every "
            "model's predictions."
        )
        raise PipelineError(msg)

    return deflate_outcomes(
        outcomes,
        n_trials_by_model={name: result.n_trials for name, result in tuning.items()},
        extra_sharpes=[value for result in tuning.values() for value in result.values],
    )
