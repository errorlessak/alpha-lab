"""Hyperparameter search with an economic objective (§10.4, ADR 0010).

The objective is **out-of-fold and economic**. Every trial fits the model across
the same purged folds the leaderboard uses, generates a signal from the stacked
out-of-fold probabilities, runs the costed backtest, and returns net Sharpe.
Optimising in-fold log-loss is, per §10.4, the single most common way tutorials
produce fantasy results: it rewards a model for being confident about bars it
was trained on, and says nothing about whether the edge survives execution.

Two things this module refuses to let a caller do:

* **Hide the trial count.** `TuningResult.n_trials` is what the study actually
  ran, and it flows straight into the deflated-Sharpe haircut. Setting it low
  by hand is self-deception, so it is never an input.
* **Search a space nobody declared.** A model with no registered search space
  raises rather than being tuned over an invented one.

Optuna is imported lazily and lives behind the `tune` extra, so `import
alphacore` stays light for the runs that never search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from alphacore.errors import ModelError, PipelineError
from alphacore.models.oof import fit_predict_oof
from alphacore.models.registry import build_model
from alphacore.pipelines.backtest import backtest_predictions
from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from alphacore.config.schema import AlphaConfig
    from alphacore.cv import Split
    from alphacore.pipelines.build_features import FeatureSet

__all__ = ["SEARCH_SPACES", "TuningResult", "tune_candidates", "tune_model"]

SEARCH_SPACES: Final[dict[str, dict[str, tuple[Any, ...]]]] = {
    "logistic": {
        "C": ("float_log", 1e-3, 1e2),
    },
    "elasticnet": {
        "C": ("float_log", 1e-3, 1e2),
        "l1_ratio": ("float", 0.0, 1.0),
    },
    "lightgbm": {
        "num_leaves": ("int", 7, 63),
        "learning_rate": ("float_log", 0.01, 0.3),
        "n_estimators": ("int", 50, 400),
        "min_child_samples": ("int", 10, 200),
    },
    "xgboost": {
        "max_depth": ("int", 2, 8),
        "learning_rate": ("float_log", 0.01, 0.3),
        "n_estimators": ("int", 50, 400),
        "subsample": ("float", 0.5, 1.0),
    },
    "catboost": {
        # The registry's names, not CatBoost's `depth` and `iterations`. The
        # adapter translates; a search space that spoke each library's dialect
        # would have to be rewritten to move a parameter between models.
        "max_depth": ("int", 2, 8),
        "learning_rate": ("float_log", 0.01, 0.3),
        "n_estimators": ("int", 50, 400),
    },
}
"""Per-model search spaces. Deliberately small: a wide space over a short
sample is a machine for manufacturing the best of a thousand coin flips, and
every extra trial makes the deflated Sharpe haircut larger."""

_FAILED_TRIAL_VALUE: Final[float] = -1e9
"""Objective value for a trial whose model would not fit. Large and negative so
the sampler avoids that region, rather than a `NaN` the sampler cannot order."""


@dataclass(frozen=True)
class TuningResult:
    """What a study found for one model.

    Attributes:
        model: Model name.
        best_params: The parameters that maximised out-of-fold net Sharpe.
        best_value: That Sharpe.
        n_trials: Trials actually completed. This is the `N` in the deflated
            Sharpe (§11.2), collected automatically.
        values: Every trial's objective value, in trial order. Kept because the
            *spread* of these is the `Var[SR_n]` the deflation needs.
    """

    model: str
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    values: list[float] = field(default_factory=list)


def _suggest(trial: Any, space: dict[str, tuple[Any, ...]]) -> dict[str, Any]:  # noqa: ANN401
    """Draw one parameter set from a declared search space.

    Args:
        trial: The optuna trial.
        space: Parameter name to `(kind, low, high)`.

    Returns:
        Concrete parameters.

    Raises:
        PipelineError: If a space declares an unknown parameter kind.
    """
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind, low, high = spec
        if kind == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(low), float(high))
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, float(low), float(high), log=True)
        else:
            msg = f"Unknown search-space kind {kind!r} for parameter {name!r}."
            raise PipelineError(msg)
    return params


def tune_model(
    name: str,
    feature_set: FeatureSet,
    splits: Sequence[Split],
    config: AlphaConfig,
) -> TuningResult:
    """Search one model's hyperparameters against out-of-fold net Sharpe.

    Args:
        name: Model name, which must have a registered search space.
        feature_set: Aligned features, labels and bars.
        splits: The purged folds. The same folds the leaderboard uses, so the
            search cannot be scored on an easier split than the result is.
        config: The resolved configuration.

    Returns:
        The study's result.

    Raises:
        PipelineError: If optuna is not installed, or the model has no
            registered search space.
    """
    optuna = _import_optuna()
    space = SEARCH_SPACES.get(name)
    if space is None:
        msg = (
            f"No search space is registered for {name!r}, so it cannot be tuned. "
            f"Known: {sorted(SEARCH_SPACES)}. Add one to SEARCH_SPACES or drop the model "
            "from models.candidates while tuning is enabled."
        )
        raise PipelineError(msg)

    tuning = config.models.tuning
    periods_per_year = feature_set.periods_per_year

    def objective(trial: Any) -> float:  # noqa: ANN401 - optuna's Trial is untyped here
        params = _suggest(trial, space)
        try:
            model = build_model(
                name, seed=config.meta.seed, calibrate=config.models.calibrate, **params
            )
            oof = fit_predict_oof(
                model,
                feature_set.features,
                feature_set.labels,
                splits,
                prune=config.features.prune,
            )
            outcome = backtest_predictions(
                oof,
                feature_set.bars,
                config,
                periods_per_year=periods_per_year,
                seed=config.meta.seed,
            )
        except (ModelError, PipelineError) as exc:
            # A parameter combination the estimator rejects is a fact about the
            # space, not a reason to abandon the study.
            logger.bind(phase="tune").debug("{} trial rejected: {}", name, exc)
            return _FAILED_TRIAL_VALUE
        return outcome.sharpe

    sampler = optuna.samplers.TPESampler(seed=config.meta.seed)
    pruner = (
        optuna.pruners.MedianPruner()
        if tuning.pruner == "median"
        else optuna.pruners.HyperbandPruner()
        if tuning.pruner == "hyperband"
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=tuning.n_trials, timeout=tuning.timeout_s)

    values = [
        float(t.value)
        for t in study.trials
        if t.value is not None and t.value > _FAILED_TRIAL_VALUE
    ]
    logger.bind(phase="tune").info(
        "{}: {} trials, best out-of-fold net Sharpe {:.3f}",
        name,
        len(study.trials),
        study.best_value,
    )
    return TuningResult(
        model=name,
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        values=values,
    )


def tune_candidates(
    feature_set: FeatureSet,
    splits: Sequence[Split],
    config: AlphaConfig,
) -> dict[str, TuningResult]:
    """Tune every candidate that has a registered search space.

    Candidates without one are skipped rather than refused: the mandatory
    baselines have no hyperparameters, and a leaderboard without them is a
    leaderboard with no scale on its axis.

    Args:
        feature_set: Aligned features, labels and bars.
        splits: The purged folds.
        config: The resolved configuration.

    Returns:
        Results by model name, empty when tuning is disabled.

    Raises:
        PipelineError: If tuning is enabled and optuna is not installed.
    """
    if not config.models.tuning.enabled:
        return {}
    results: dict[str, TuningResult] = {}
    for name in config.models.candidates:
        if name not in SEARCH_SPACES:
            logger.bind(phase="tune").info("{} has no search space; not tuned", name)
            continue
        results[name] = tune_model(name, feature_set, splits, config)
    return results


def _import_optuna() -> Any:  # noqa: ANN401 - the module object, typed as Any by design
    """Import optuna, turning the absence of the extra into a typed error.

    Returns:
        The optuna module, with its own logging quietened so the run's log
        remains the record of what happened.

    Raises:
        PipelineError: If optuna is not installed.
    """
    try:
        import optuna  # noqa: PLC0415 - lazy so `import alphacore` stays light
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        msg = (
            "models.tuning.enabled is true but optuna is not installed. It lives behind "
            "the `tune` extra: `uv sync --all-packages --extra tune`. Disable tuning to "
            "run without it."
        )
        raise PipelineError(msg) from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna
