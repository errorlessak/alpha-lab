"""Sequence models and the GBM ensemble (§10.3, Phase 13).

Split deliberately in two. The ensemble and the registry wiring are torch-free
and always run; everything that trains a network is gated on the optional `dl`
extra, the same way the tuning tests are gated on `tune` (ADR 0010).
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from alphacore.errors import ModelError
from alphacore.models import build_model, registered_models
from alphacore.models.ensemble import EnsembleModel
from alphacore.models.protocol import AlphaModel

N = 260
N_FEATURES = 3
SEQ_LEN = 5
FAST = {
    "seq_len": SEQ_LEN,
    "hidden_size": 8,
    "max_epochs": 4,
    "patience": 2,
    "batch_size": 64,
}

SEQUENCE_MODELS = ("lstm", "lstm_stacked", "bilstm", "gru", "tcn")


@pytest.fixture
def dataset():
    """A small frame with a weak real edge, on an unbroken daily grid."""
    rng = np.random.default_rng(20260728)
    index = pd.date_range("2020-01-01", periods=N, freq="D", tz="UTC")
    features = pd.DataFrame(
        rng.standard_normal((N, N_FEATURES)),
        index=index,
        columns=[f"f{i}" for i in range(N_FEATURES)],
    )
    score = features["f0"].to_numpy() * 0.8 + rng.standard_normal(N) * 0.5
    labels = pd.Series((score > 0).astype(int), index=index, name="label")
    return features, labels


def _fitted(name, dataset, **params):
    """Build a sequence model, tell it the grid, and fit it."""
    features, labels = dataset
    model = build_model(name, seed=5, params={**FAST, **params})
    model.set_bar_grid(features.index)
    return model.fit(features, labels)


# --- registry, with and without torch --------------------------------------


def test_every_sequence_model_the_spec_names_is_registered():
    names = registered_models()
    for required in (*SEQUENCE_MODELS, "ensemble_gbm_lstm"):
        assert required in names


def test_building_a_sequence_model_without_torch_names_the_extra(monkeypatch):
    real = importlib.import_module

    def refuse(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            msg = "No module named 'torch'"
            raise ImportError(msg)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", refuse)
    with pytest.raises(ModelError, match="--extra dl"):
        build_model("lstm", seed=1)


# --- the ensemble, which needs no torch ------------------------------------


def test_the_ensemble_averages_its_members(dataset):
    features, labels = dataset
    members = [build_model("logistic", seed=1), build_model("elasticnet", seed=1)]
    ensemble = EnsembleModel("ensemble_linear", members)
    ensemble.fit(features, labels)

    expected = np.mean([m.predict_proba(features)[:, 1] for m in members], axis=0)
    np.testing.assert_allclose(ensemble.predict_proba(features)[:, 1], expected, atol=1e-12)


def test_the_ensemble_satisfies_the_model_protocol():
    ensemble = EnsembleModel("ensemble_linear", [build_model("logistic", seed=1)])
    assert isinstance(ensemble, AlphaModel)


def test_an_ensemble_with_no_members_is_rejected():
    with pytest.raises(ModelError, match="member"):
        EnsembleModel("ensemble_empty", [])


def test_the_ensemble_averages_the_feature_importance_it_can_get(dataset):
    features, labels = dataset
    ensemble = EnsembleModel(
        "ensemble_mixed", [build_model("logistic", seed=1), build_model("baseline_buyhold", seed=1)]
    )
    ensemble.fit(features, labels)

    importance = ensemble.feature_importance()
    assert importance is not None
    assert set(importance.index) == set(features.columns)


def test_the_ensemble_must_be_fitted_before_it_predicts(dataset):
    features, _ = dataset
    ensemble = EnsembleModel("ensemble_linear", [build_model("logistic", seed=1)])
    with pytest.raises(ModelError, match="fit"):
        ensemble.predict_proba(features)


# --- everything below trains a network -------------------------------------

torch = pytest.importorskip("torch", reason="sequence models live behind the `dl` extra")


@pytest.mark.parametrize("name", SEQUENCE_MODELS)
def test_sequence_models_satisfy_the_model_protocol(name):
    assert isinstance(build_model(name, seed=1, params=FAST), AlphaModel)


@pytest.mark.parametrize("name", SEQUENCE_MODELS)
def test_a_sequence_model_fits_and_predicts_every_row(dataset, name):
    features, _ = dataset
    model = _fitted(name, dataset)

    proba = model.predict_proba(features)
    assert proba.shape == (len(features), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
    assert not np.isnan(proba).any()
    assert model.predict(features).shape == (len(features),)


def test_fitting_without_the_bar_grid_is_an_error(dataset):
    features, labels = dataset
    model = build_model("lstm", seed=1, params=FAST)
    with pytest.raises(ModelError, match="set_bar_grid"):
        model.fit(features, labels)


def test_predicting_before_fitting_is_an_error(dataset):
    features, _ = dataset
    with pytest.raises(ModelError, match="fit"):
        build_model("lstm", seed=1, params=FAST).predict_proba(features)


def test_training_windows_stop_at_a_purge_gap(dataset):
    features, labels = dataset
    kept = np.r_[0:100, 160:N]
    model = build_model("lstm", seed=1, params=FAST)
    model.set_bar_grid(features.index)
    model.fit(features.iloc[kept], labels.iloc[kept])

    # Two runs, each losing `seq_len - 1` rows to the incomplete history at its
    # head. Stitching the runs together would have produced SEQ_LEN - 1 more.
    assert model.n_train_windows == len(kept) - 2 * (SEQ_LEN - 1)


def test_a_sequence_model_is_reproducible_given_a_seed(dataset):
    features, _ = dataset
    first = _fitted("lstm", dataset)
    second = _fitted("lstm", dataset)

    np.testing.assert_allclose(
        first.predict_proba(features), second.predict_proba(features), atol=1e-12
    )


def test_a_different_seed_gives_a_different_fit(dataset):
    features, labels = dataset
    a = build_model("lstm", seed=1, params=FAST)
    b = build_model("lstm", seed=2, params=FAST)
    for model in (a, b):
        model.set_bar_grid(features.index)
        model.fit(features, labels)

    assert not np.allclose(a.predict_proba(features), b.predict_proba(features))


def test_early_stopping_keeps_the_best_epoch_not_the_last(dataset):
    model = _fitted("lstm", dataset, max_epochs=6, patience=2)

    assert 1 <= model.best_epoch <= len(model.history)
    losses = [entry["val_loss"] for entry in model.history]
    assert losses[model.best_epoch - 1] == pytest.approx(min(losses))


def test_early_stopping_gives_up_before_the_epoch_budget_when_nothing_improves(dataset):
    model = _fitted("lstm", dataset, max_epochs=60, patience=1, lr=0.0)

    # A zero learning rate cannot improve anything, so patience must end it.
    assert len(model.history) < 60


def test_the_learning_rate_schedule_is_recorded(dataset):
    model = _fitted("lstm", dataset, max_epochs=4)

    assert all(entry["lr"] > 0 for entry in model.history)


def test_a_sequence_model_reports_no_feature_importance(dataset):
    assert _fitted("lstm", dataset).feature_importance() is None


def test_a_sequence_model_round_trips_through_disk(dataset, tmp_path):
    features, _ = dataset
    model = _fitted("lstm", dataset)
    before = model.predict_proba(features)

    path = tmp_path / "model.pkl"
    model.save(path)
    restored = type(model).load(path)

    np.testing.assert_allclose(restored.predict_proba(features), before, atol=1e-12)


def test_an_unknown_architecture_is_rejected():
    with pytest.raises(ModelError, match="architecture"):
        build_model("lstm", seed=1, params={**FAST, "architecture": "transformer"})


def test_a_bidirectional_window_still_cannot_see_the_future(dataset):
    """Bidirectional over a *past* window is legal; over the future it is a leak."""
    features, _ = dataset
    model = _fitted("bilstm", dataset)
    before = model.predict_proba(features)[:, 1]

    poisoned = features.copy()
    poisoned.iloc[150:] += 99.0
    after = model.predict_proba(poisoned)[:, 1]

    # Rows before the poison, and far enough back that their window cannot
    # reach it, must be untouched.
    np.testing.assert_allclose(after[: 150 - SEQ_LEN], before[: 150 - SEQ_LEN], atol=1e-12)
    # The control: the rows whose own window *does* contain poisoned bars move.
    assert not np.allclose(after[150:], before[150:])


def test_the_gbm_sequence_ensemble_blends_both_kinds(dataset):
    features, labels = dataset
    pytest.importorskip("lightgbm", reason="the ensemble's GBM leg lives behind the `gbm` extra")
    ensemble = build_model("ensemble_gbm_lstm", seed=3, params=FAST)
    ensemble.set_bar_grid(features.index)
    ensemble.fit(features, labels)

    proba = ensemble.predict_proba(features)
    assert proba.shape == (len(features), 2)
    assert [member.name for member in ensemble.members] == ["lightgbm", "lstm"]
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
