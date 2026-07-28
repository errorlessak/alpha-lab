"""A sequence model reaches the leaderboard on its own (§15, Phase 13 exit).

The exit criterion is not "the network trains". It is that a sequence model goes
through the *same* out-of-fold path, against the *same* mandatory baselines, and
is ranked by the same economic score — including being told the bar grid by the
OOF loop rather than by a caller who remembered to. Nothing in this file calls
`set_bar_grid`; if `fit_predict_oof` stops doing it, these tests fail with the
message that names the omission.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacore.config.schema import CVConfig, ModelsConfig, WalkForwardConfig
from alphacore.cv import make_splits
from alphacore.models import build_leaderboard, build_model, fit_predict_oof

pytest.importorskip("torch", reason="sequence models live behind the `dl` extra")

N = 700
HORIZON = 5
FAST = {"seq_len": 8, "hidden_size": 8, "max_epochs": 3, "patience": 2}


@pytest.fixture(scope="module")
def dataset():
    """Features and overlapping labels with a weak, real edge on f0."""
    rng = np.random.default_rng(20260728)
    index = pd.date_range("2016-01-01", periods=N, freq="D", tz="UTC")
    features = pd.DataFrame(
        rng.standard_normal((N, 4)),
        index=index,
        columns=[f"f{i}" for i in range(4)],
    )
    score = features["f0"].to_numpy() * 0.7 + rng.standard_normal(N) * 0.6
    usable = N - HORIZON
    labels = pd.DataFrame(
        {
            "t_end": index[HORIZON : HORIZON + usable],
            "label": (score[:usable] > 0).astype(int),
            "realised_return": score[:usable] * 0.001,
            "weight": np.ones(usable),
        },
        index=pd.DatetimeIndex(index[:usable], name="ts"),
    )
    return features.iloc[:usable], labels


@pytest.fixture(scope="module")
def splits(dataset):
    _, labels = dataset
    config = CVConfig(
        scheme="walkforward",
        walkforward=WalkForwardConfig(train_bars=300, test_bars=150, step_bars=150),
        embargo_pct=0.01,
        min_train_bars=100,
    )
    return make_splits(labels, config)


@pytest.fixture(scope="module")
def oof(dataset, splits):
    features, labels = dataset
    model = build_model("lstm", seed=13, params=FAST)
    return fit_predict_oof(model, features, labels, splits)


def test_the_oof_loop_tells_the_model_its_bar_grid(oof, dataset, splits):
    _, labels = dataset
    expected = sum(int(split.test_idx.size) for split in splits)
    assert oof.model == "lstm"
    assert len(oof.frame) == expected
    assert oof.frame["proba"].notna().all()
    assert labels.index.union(oof.frame.index).equals(labels.index)


def test_every_out_of_fold_probability_is_a_probability(oof):
    proba = oof.frame["proba"].to_numpy()
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
    # A network that collapsed to one constant would still pass the range check.
    assert proba.std() > 0.0


def test_a_sequence_model_appears_on_the_leaderboard(dataset, splits):
    features, labels = dataset
    config = ModelsConfig(candidates=("logistic", "lstm"), calibrate="none")
    rows = build_leaderboard(features, labels, splits, config, seed=13)

    scored = {row["model"]: row for row in rows}
    assert "lstm" in scored
    lstm = scored["lstm"]
    assert lstm["n_predictions"] > 0
    assert lstm["n_folds"] == len(splits)
    assert lstm["rank"] >= 1
    # Ranked against the mandatory baselines, not in a category of its own.
    assert "baseline_shuffled" in scored
    assert {row["rank"] for row in rows} == set(range(1, max(r["rank"] for r in rows) + 1))


def test_the_ensemble_is_scored_beside_its_legs(dataset, splits):
    pytest.importorskip("lightgbm", reason="the ensemble's GBM leg needs the `gbm` extra")
    features, labels = dataset
    config = ModelsConfig(candidates=("lightgbm", "ensemble_gbm_lstm"), calibrate="none")
    rows = build_leaderboard(features, labels, splits, config, seed=13)

    scored = {row["model"]: row for row in rows}
    assert scored["ensemble_gbm_lstm"]["n_predictions"] > 0
    assert scored["ensemble_gbm_lstm"]["n_folds"] == len(splits)
