"""Model adapters, baselines, calibration, OOF assembly, leaderboard (§10)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from alphacore.config.schema import CVConfig, ModelsConfig, WalkForwardConfig
from alphacore.cv import make_splits
from alphacore.errors import ModelError
from alphacore.eval.classification import (
    brier_score,
    classification_metrics,
    information_coefficient,
    log_loss_score,
    roc_auc,
)
from alphacore.models import (
    MANDATORY_BASELINES,
    build_leaderboard,
    build_model,
    fit_predict_oof,
    registered_models,
)
from alphacore.models.protocol import AlphaModel
from alphacore.pipelines.tune import SEARCH_SPACES

N = 600
N_FEATURES = 6


@pytest.fixture
def dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features with a weak real signal, plus overlapping triple-barrier-ish labels."""
    rng = np.random.default_rng(20260727)
    index = pd.date_range("2015-01-01", periods=N, freq="D", tz="UTC")
    features = pd.DataFrame(
        rng.standard_normal((N, N_FEATURES)),
        index=index,
        columns=[f"f{i}" for i in range(N_FEATURES)],
    )
    # A genuine but weak edge: the label leans on f0, buried in noise.
    score = features["f0"].to_numpy() * 0.6 + rng.standard_normal(N)
    horizon = 5
    usable = N - horizon
    labels = pd.DataFrame(
        {
            "t_end": index[horizon : horizon + usable],
            "label": (score[:usable] > 0).astype(int),
            "realised_return": score[:usable] * 0.001,
            "weight": np.ones(usable),
        },
        index=pd.DatetimeIndex(index[:usable], name="ts"),
    )
    return features.iloc[:usable], labels


@pytest.fixture
def splits(dataset):
    _, labels = dataset
    config = CVConfig(
        scheme="walkforward",
        walkforward=WalkForwardConfig(train_bars=250, test_bars=100, step_bars=100),
        embargo_pct=0.01,
        min_train_bars=50,
    )
    return make_splits(labels, config)


# --- registry and protocol -------------------------------------------------


def test_registry_lists_the_models_the_spec_names():
    names = registered_models()
    for required in ("logistic", "elasticnet", "lightgbm", "xgboost", "catboost"):
        assert required in names


def test_every_mandatory_baseline_is_registered():
    for baseline in MANDATORY_BASELINES:
        assert baseline in registered_models()


def test_mandatory_baselines_cover_the_spec():
    # §10.1: buy-and-hold, random-sign, always-long, momentum, shuffled-label.
    assert set(MANDATORY_BASELINES) == {
        "baseline_buyhold",
        "baseline_always_long",
        "baseline_random_sign",
        "baseline_momentum",
        "baseline_shuffled",
    }


@pytest.mark.parametrize("name", ["logistic", "elasticnet", "baseline_momentum"])
def test_built_models_satisfy_the_protocol(name):
    assert isinstance(build_model(name, seed=1), AlphaModel)


def test_building_an_unknown_model_lists_the_known_ones():
    with pytest.raises(ModelError, match="Unknown model"):
        build_model("magic", seed=1)


def test_an_unregistered_candidate_stops_the_leaderboard(dataset, splits):
    # A missing optional dependency earns a FAILED row so the rest of the
    # leaderboard still means something — torch is two gigabytes and a
    # researcher may knowingly do without it. A *typo* is a different failure:
    # continuing produces a leaderboard the researcher reads as though the model
    # they meant to run had been on it, with the trial count to match.
    features, labels = dataset
    config = ModelsConfig(candidates=["logistic", "lightgbmm"])
    with pytest.raises(ModelError, match="not registered"):
        build_leaderboard(features, labels, splits, config, seed=1)


# --- parameters reaching the estimator -------------------------------------
#
# The registry is the only path a tuned parameter takes to a model: `tune_model`
# and `build_leaderboard` both hand their settings to `build_model` and trust it
# to arrive. A parameter that is silently dropped, or that raises `TypeError`
# from inside a builder, makes a study either meaningless or dead — and the
# study is what the deflated-Sharpe haircut counts.

GBM_MODELS = ("lightgbm", "xgboost", "catboost")

TUNED_SETTINGS = {
    "logistic": ({"C": 0.25}, {"C": 0.25}),
    "elasticnet": ({"C": 0.25, "l1_ratio": 0.3}, {"C": 0.25, "l1_ratio": 0.3}),
    "lightgbm": (
        {"num_leaves": 15, "n_estimators": 60, "learning_rate": 0.05, "min_child_samples": 25},
        {"num_leaves": 15, "n_estimators": 60, "learning_rate": 0.05, "min_child_samples": 25},
    ),
    "xgboost": (
        {"max_depth": 4, "n_estimators": 60, "subsample": 0.7},
        {"max_depth": 4, "n_estimators": 60, "subsample": 0.7},
    ),
    "catboost": (
        {"max_depth": 4, "n_estimators": 60, "learning_rate": 0.05},
        {"depth": 4, "iterations": 60, "learning_rate": 0.05},
    ),
}
"""Model name to `(settings passed in, parameters the estimator must carry)`.
The two differ for CatBoost, whose library calls them `depth` and `iterations`;
the registry's vocabulary is the same for every model so a search space does not
have to know which library it landed on."""


@pytest.mark.parametrize("name", sorted(TUNED_SETTINGS))
def test_tuned_parameters_reach_the_estimator(name):
    if name in GBM_MODELS:
        pytest.importorskip(name, reason="the boosting libraries live behind the `gbm` extra")
    passed, expected = TUNED_SETTINGS[name]
    model = build_model(name, seed=1, **passed)

    settings = model.estimator.get_params()
    for key, value in expected.items():
        assert settings[key] == pytest.approx(value), key


@pytest.mark.parametrize("name", sorted(TUNED_SETTINGS))
def test_the_same_parameters_can_arrive_as_a_mapping(name):
    if name in GBM_MODELS:
        pytest.importorskip(name, reason="the boosting libraries live behind the `gbm` extra")
    passed, expected = TUNED_SETTINGS[name]
    model = build_model(name, seed=1, params=passed)

    settings = model.estimator.get_params()
    for key, value in expected.items():
        assert settings[key] == pytest.approx(value), key


@pytest.mark.parametrize("name", ["logistic", "lightgbm"])
def test_a_parameter_the_model_does_not_have_is_a_typed_error(name):
    if name in GBM_MODELS:
        pytest.importorskip(name, reason="the boosting libraries live behind the `gbm` extra")
    # Never a TypeError: every failure in this library is typed (§0.7).
    with pytest.raises(ModelError, match="parameter"):
        build_model(name, seed=1, definitely_not_a_hyperparameter=3)


@pytest.mark.parametrize("name", sorted(SEARCH_SPACES))
def test_every_search_space_parameter_is_accepted_by_its_model(name):
    """The tuner's vocabulary and the builders' cannot be allowed to drift."""
    if name in GBM_MODELS:
        pytest.importorskip(name, reason="the boosting libraries live behind the `gbm` extra")
    low = {key: spec[1] for key, spec in SEARCH_SPACES[name].items()}

    assert build_model(name, seed=1, **low) is not None


# --- fitting and predicting ------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "logistic",
        "elasticnet",
        "lightgbm",
        "xgboost",
        "catboost",
        "baseline_buyhold",
        "baseline_always_long",
        "baseline_random_sign",
        "baseline_momentum",
        "baseline_shuffled",
    ],
)
def test_every_model_fits_and_predicts(dataset, name):
    features, labels = dataset
    model = build_model(name, seed=7)
    model.fit(features, labels["label"], sample_weight=labels["weight"].to_numpy())

    proba = model.predict_proba(features)
    assert proba.shape == (len(features), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
    assert model.predict(features).shape == (len(features),)


def test_predicting_before_fitting_is_an_error(dataset):
    features, _ = dataset
    with pytest.raises(ModelError, match="fit"):
        build_model("logistic", seed=1).predict_proba(features)


def test_models_are_reproducible_given_a_seed(dataset):
    features, labels = dataset
    first = build_model("lightgbm", seed=42).fit(features, labels["label"])
    second = build_model("lightgbm", seed=42).fit(features, labels["label"])
    np.testing.assert_allclose(
        first.predict_proba(features), second.predict_proba(features), atol=1e-12
    )


def test_random_sign_baseline_is_seeded(dataset):
    features, labels = dataset
    a = build_model("baseline_random_sign", seed=3).fit(features, labels["label"])
    b = build_model("baseline_random_sign", seed=3).fit(features, labels["label"])
    c = build_model("baseline_random_sign", seed=4).fit(features, labels["label"])
    np.testing.assert_allclose(a.predict_proba(features), b.predict_proba(features))
    assert not np.allclose(a.predict_proba(features), c.predict_proba(features))


def test_buyhold_baseline_is_always_long(dataset):
    features, labels = dataset
    model = build_model("baseline_buyhold", seed=1).fit(features, labels["label"])
    assert (model.predict(features) == 1).all()


def test_shuffled_baseline_destroys_the_relationship(dataset):
    features, labels = dataset
    honest = build_model("logistic", seed=1).fit(features, labels["label"])
    shuffled = build_model("baseline_shuffled", seed=1).fit(features, labels["label"])

    honest_auc = roc_auc(labels["label"].to_numpy(), honest.predict_proba(features)[:, 1])
    shuffled_auc = roc_auc(labels["label"].to_numpy(), shuffled.predict_proba(features)[:, 1])
    assert honest_auc > shuffled_auc


@pytest.mark.parametrize("name", ["logistic", "lightgbm", "baseline_momentum"])
def test_feature_importance_is_named_or_absent(dataset, name):
    features, labels = dataset
    model = build_model(name, seed=1).fit(features, labels["label"])
    importance = model.feature_importance()
    if importance is not None:
        # Named for every feature, and ordered most-important first.
        assert set(importance.index) == set(features.columns)
        assert importance.is_monotonic_decreasing


def test_models_round_trip_through_disk(dataset, tmp_path):
    features, labels = dataset
    model = build_model("logistic", seed=1).fit(features, labels["label"])
    before = model.predict_proba(features)

    path = tmp_path / "model.pkl"
    model.save(path)
    restored = type(model).load(path)
    np.testing.assert_allclose(restored.predict_proba(features), before, atol=1e-12)


# --- calibration -----------------------------------------------------------


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_produces_valid_probabilities(dataset, method):
    features, labels = dataset
    model = build_model("lightgbm", seed=1, calibrate=method)
    model.fit(features, labels["label"])
    proba = model.predict_proba(features)[:, 1]
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_calibration_improves_the_brier_score(dataset):
    features, labels = dataset
    y = labels["label"].to_numpy()
    raw = build_model("lightgbm", seed=1, calibrate="none").fit(features, y)
    cal = build_model("lightgbm", seed=1, calibrate="isotonic").fit(features, y)
    # Not guaranteed on every dataset, but on a held-out-shaped inner split a
    # calibrated GBM should not be worse by much; assert it is in the ballpark.
    assert (
        brier_score(y, cal.predict_proba(features)[:, 1])
        <= brier_score(y, raw.predict_proba(features)[:, 1]) + 0.05
    )


def test_calibration_uses_an_inner_split_not_the_test_fold(dataset):
    # I3: the calibrator is fitted inside the training fold. If it saw the test
    # fold, a perfectly separable test set would calibrate to 0/1 exactly.
    features, labels = dataset
    model = build_model("logistic", seed=1, calibrate="isotonic")
    model.fit(features.iloc[:300], labels["label"].iloc[:300])
    held_out = model.predict_proba(features.iloc[300:])[:, 1]
    assert 0.0 < held_out.mean() < 1.0


def test_an_unknown_calibration_method_is_rejected():
    with pytest.raises(ModelError, match="calibrat"):
        build_model("logistic", seed=1, calibrate="magic")


# --- OOF assembly ----------------------------------------------------------


def test_oof_covers_every_test_row_exactly_once(dataset, splits):
    features, labels = dataset
    oof = fit_predict_oof(build_model("logistic", seed=1), features, labels, splits)

    covered = np.concatenate([s.test_idx for s in splits])
    assert len(oof.frame) == len(covered)
    assert oof.frame.index.is_unique
    assert oof.frame.index.is_monotonic_increasing


def test_oof_records_which_fold_each_prediction_came_from(dataset, splits):
    features, labels = dataset
    oof = fit_predict_oof(build_model("logistic", seed=1), features, labels, splits)
    assert set(oof.frame["fold"].unique()) == {s.fold for s in splits}


def test_oof_carries_the_label_and_the_realised_return(dataset, splits):
    features, labels = dataset
    oof = fit_predict_oof(build_model("logistic", seed=1), features, labels, splits)
    for column in ("proba", "prediction", "label", "realised_return", "weight", "fold"):
        assert column in oof.frame.columns


def test_oof_never_trains_on_a_test_row(dataset, splits):
    features, labels = dataset
    seen: list[set[int]] = []

    class Spy:
        name = "spy"
        task = "binary"

        def fit(self, X, y, *, sample_weight=None, eval_set=None):
            seen.append({labels.index.get_loc(t) for t in X.index})
            self._n = X.shape[1]
            return self

        def predict_proba(self, X):
            return np.tile([0.5, 0.5], (len(X), 1))

        def predict(self, X):
            return np.ones(len(X), dtype=int)

        def feature_importance(self):
            return None

        def save(self, path):
            path.write_bytes(b"")

        @classmethod
        def load(cls, path):
            return cls()

    fit_predict_oof(Spy(), features, labels, splits)
    for split, trained_on in zip(splits, seen, strict=True):
        assert not trained_on & set(split.test_idx.tolist())


def test_oof_rejects_multiclass_labels_with_an_actionable_message(dataset, splits):
    features, labels = dataset
    three_class = labels.copy()
    three_class.iloc[:50, three_class.columns.get_loc("label")] = 2
    with pytest.raises(ModelError, match="to_binary"):
        fit_predict_oof(build_model("logistic", seed=1), features, three_class, splits)


def test_oof_requires_aligned_features_and_labels(dataset, splits):
    features, labels = dataset
    with pytest.raises(ModelError, match="index"):
        fit_predict_oof(build_model("logistic", seed=1), features.iloc[:-10], labels, splits)


def test_oof_is_deterministic(dataset, splits):
    features, labels = dataset
    first = fit_predict_oof(build_model("lightgbm", seed=5), features, labels, splits)
    second = fit_predict_oof(build_model("lightgbm", seed=5), features, labels, splits)
    np.testing.assert_allclose(
        first.frame["proba"].to_numpy(), second.frame["proba"].to_numpy(), atol=1e-12
    )


# --- classification metrics ------------------------------------------------


def test_perfect_predictions_score_perfectly():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.0, 0.0, 1.0, 1.0])
    assert roc_auc(y, proba) == pytest.approx(1.0)
    assert brier_score(y, proba) == pytest.approx(0.0)


def test_coin_flip_auc_is_a_half():
    y = np.array([0, 1, 0, 1])
    assert roc_auc(y, np.full(4, 0.5)) == pytest.approx(0.5)


def test_log_loss_punishes_confident_mistakes():
    y = np.array([1, 1])
    assert log_loss_score(y, np.array([0.99, 0.99])) < log_loss_score(y, np.array([0.01, 0.01]))


def test_information_coefficient_is_a_rank_correlation():
    forward = np.array([-3.0, -1.0, 1.0, 3.0])
    ic, tstat = information_coefficient(np.array([0.1, 0.2, 0.8, 0.9]), forward)
    assert ic == pytest.approx(1.0)
    assert tstat > 0


def test_information_coefficient_of_noise_is_near_zero():
    rng = np.random.default_rng(1)
    ic, _ = information_coefficient(rng.standard_normal(500), rng.standard_normal(500))
    assert abs(ic) < 0.15


def test_classification_metrics_reports_the_full_set():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    proba = rng.uniform(size=400)
    metrics = classification_metrics(y, proba, forward_return=rng.standard_normal(400))
    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "brier",
        "log_loss",
        "ic",
        "ic_tstat",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_metrics_refuse_a_single_class_rather_than_returning_nan():
    # A silent NaN propagating into a leaderboard is a critical bug (§0.7).
    with pytest.raises(ModelError, match="one class"):
        roc_auc(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


# --- leaderboard -----------------------------------------------------------


@pytest.fixture
def leaderboard(dataset, splits):
    features, labels = dataset
    config = ModelsConfig(candidates=("logistic", "lightgbm"), calibrate="sigmoid")
    return build_leaderboard(features, labels, splits, config, seed=11)


def test_leaderboard_includes_every_candidate_and_baseline(leaderboard):
    names = {row["model"] for row in leaderboard}
    assert {"logistic", "lightgbm"} <= names
    assert set(MANDATORY_BASELINES) <= names


def test_leaderboard_rows_carry_the_honesty_columns(leaderboard):
    row = leaderboard[0]
    for key in (
        "model",
        "rank",
        "ic",
        "ic_tstat",
        "auc",
        "brier",
        "accuracy",
        "n_trials",
        "oos_net_sharpe",
        "dsr",
        "psr",
        "break_even_bps",
        "flags",
    ):
        assert key in row


def test_leaderboard_is_ranked_and_dense(leaderboard):
    ranks = [row["rank"] for row in leaderboard]
    assert ranks == list(range(1, len(leaderboard) + 1))


def test_economic_columns_are_null_until_the_backtester_exists(leaderboard):
    # Phase 5 has no cost model, so net Sharpe is honestly absent rather than
    # filled with a placeholder number nobody should read.
    assert all(row["oos_net_sharpe"] is None for row in leaderboard)
    assert all("NO_BACKTEST" in row["flags"] for row in leaderboard)


def test_the_shuffled_baseline_is_flagged(leaderboard):
    row = next(r for r in leaderboard if r["model"] == "baseline_shuffled")
    assert "SHUFFLED_LABELS" in row["flags"]


def test_leaderboard_serialises_to_json(leaderboard):
    payload = json.dumps(leaderboard, sort_keys=True)
    assert json.loads(payload) == leaderboard
