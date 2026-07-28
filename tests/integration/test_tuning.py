"""The economic hyperparameter search (§10.4, ADR 0010).

The property that matters is not "does optuna run". It is that the objective is
out-of-fold and economic — a trial is scored by the net Sharpe of a costed
backtest of its own out-of-fold probabilities — and that the trial count reaches
the deflated Sharpe by itself.
"""

from __future__ import annotations

import pytest

from alphacore.config import load_config
from alphacore.config.schema import ModelsConfig, TuningConfig, WalkForwardConfig
from alphacore.cv import make_splits
from alphacore.errors import PipelineError
from alphacore.models.oof import fit_predict_oof
from alphacore.models.registry import build_model
from alphacore.pipelines import build_feature_set, ingest_symbols, load_bars, train_models
from alphacore.pipelines.backtest import backtest_predictions
from alphacore.pipelines.tune import SEARCH_SPACES, tune_candidates, tune_model

optuna = pytest.importorskip("optuna", reason="tuning lives behind the `tune` extra (ADR 0010)")

TRIALS = 3
"""Enough for a study to have a spread, few enough for the suite to stay fast."""


@pytest.fixture(scope="module")
def tunable(repo_root, tmp_path_factory):
    """A small feature set, its folds, and a config with tuning switched on."""
    root = tmp_path_factory.mktemp("tuning")
    config = load_config(repo_root / "configs" / "experiments" / "smoke.yaml", use_env=False)
    config = config.model_copy(
        update={
            "data": config.data.model_copy(update={"csv_dir": str(repo_root / "samples")}),
            "models": ModelsConfig(
                candidates=("logistic",),
                calibrate="sigmoid",
                tuning=TuningConfig(enabled=True, n_trials=TRIALS, timeout_s=120),
            ),
            "cv": config.cv.model_copy(
                update={
                    "walkforward": WalkForwardConfig(
                        train_bars=500, test_bars=250, step_bars=250, anchored=False
                    )
                }
            ),
        }
    )
    ingest_symbols(config, data_dir=root / "lake")
    feature_set = build_feature_set(config, load_bars(config, data_dir=root / "lake"))
    return config, feature_set, make_splits(feature_set.labels, config.cv)


def test_every_registered_space_names_a_known_parameter_kind():
    kinds = {spec[0] for space in SEARCH_SPACES.values() for spec in space.values()}
    assert kinds <= {"int", "float", "float_log"}


def test_a_study_reports_the_trials_it_actually_ran(tunable):
    config, feature_set, splits = tunable
    result = tune_model("logistic", feature_set, splits, config)
    assert result.n_trials == TRIALS
    assert set(result.best_params) == set(SEARCH_SPACES["logistic"])


def test_the_objective_is_the_net_sharpe_of_a_costed_backtest(tunable):
    """The value a trial returns must be reproducible by backtesting it.

    If the objective were in-fold log-loss, or an uncosted Sharpe, this would
    not hold — and §10.4 names optimising in-fold log-loss as the single most
    common way tutorials produce fantasy results.
    """
    config, feature_set, splits = tunable
    result = tune_model("logistic", feature_set, splits, config)

    model = build_model(
        "logistic",
        seed=config.meta.seed,
        calibrate=config.models.calibrate,
        **result.best_params,
    )
    oof = fit_predict_oof(
        model, feature_set.features, feature_set.labels, splits, prune=config.features.prune
    )
    replayed = backtest_predictions(
        oof,
        feature_set.bars,
        config,
        periods_per_year=feature_set.periods_per_year,
        seed=config.meta.seed,
    )
    assert replayed.sharpe == pytest.approx(result.best_value, abs=1e-9)


def test_models_without_a_search_space_are_skipped_not_refused(tunable):
    config, feature_set, splits = tunable
    # The mandatory baselines have no hyperparameters, and a leaderboard
    # without them has no scale on its axis.
    widened = config.model_copy(
        update={
            "models": config.models.model_copy(
                update={"candidates": ("logistic", "baseline_buyhold")}
            )
        }
    )
    results = tune_candidates(feature_set, splits, widened)
    assert set(results) == {"logistic"}


def test_tuning_an_unknown_model_names_the_gap(tunable):
    config, feature_set, splits = tunable
    with pytest.raises(PipelineError, match="No search space is registered"):
        tune_model("baseline_buyhold", feature_set, splits, config)


def test_disabling_tuning_runs_no_study(tunable):
    config, feature_set, splits = tunable
    off = config.model_copy(
        update={
            "models": config.models.model_copy(
                update={"tuning": config.models.tuning.model_copy(update={"enabled": False})}
            )
        }
    )
    assert tune_candidates(feature_set, splits, off) == {}


@pytest.mark.slow
def test_the_trial_count_reaches_the_deflated_sharpe_automatically(tunable):
    """§10.4: `n_trials` must flow into the DSR haircut by itself.

    Searching harder must make the haircut larger. A trial count that had to be
    entered by hand would be a number nobody ever raises.
    """
    config, feature_set, splits = tunable
    tuned = train_models(config, feature_set, splits=splits)

    untuned_config = config.model_copy(
        update={
            "models": config.models.model_copy(
                update={"tuning": config.models.tuning.model_copy(update={"enabled": False})}
            )
        }
    )
    untuned = train_models(untuned_config, feature_set, splits=splits)

    assert tuned.n_trials_total > untuned.n_trials_total
    for row in tuned.rows:
        assert row["n_trials"] == tuned.n_trials_total
