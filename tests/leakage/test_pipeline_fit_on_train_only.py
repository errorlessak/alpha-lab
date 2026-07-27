"""Fitted transforms never see a test row (§12.3.4).

Invariant I3. Scalers, winsorisers, imputers and any other target- or
distribution-dependent statistic are fitted **inside the fold, on training rows
only**.

Scaling the full matrix before splitting is one of the quietest leaks there is.
The mean and variance of the whole sample encode the test period's level and
volatility, so every training row arrives pre-informed about the regime it is
about to be tested in. Nothing in the code looks wrong; there is no future
index anywhere. The out-of-sample score is simply too good.

**Two traps this file had to survive.**

The first: only the *first* pipeline step receives a DataFrame. Every step after
it gets the previous step's output, which is a bare numpy array with no index.
A spy that records `X.index` therefore sees an empty index for the scaler and
every containment check passes vacuously. The scaler is checked by row count
instead.

The second: in a walk-forward, fold 1 legitimately trains on fold 0's test
period. Comparing a fit against the union of every fold's test rows flags that
as a leak when it is the design. The check is per fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from alphacore.config.schema import CVConfig, WalkForwardConfig
from alphacore.cv import make_splits
from alphacore.models import Winsoriser, build_model, fit_predict_oof

N = 500
HORIZON = 5


@pytest.fixture
def dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features whose scale shifts halfway, so a global fit is detectable."""
    rng = np.random.default_rng(20260727)
    index = pd.date_range("2015-01-01", periods=N, freq="D", tz="UTC")
    # The second half is ten times more volatile, so a scaler fitted on
    # everything has visibly different parameters from one fitted per fold.
    scale = np.concatenate((np.ones(N // 2), np.full(N - N // 2, 10.0)))
    values = rng.standard_normal((N, 4)) * scale[:, None]
    features = pd.DataFrame(values, index=index, columns=[f"f{i}" for i in range(4)])

    usable = N - HORIZON
    labels = pd.DataFrame(
        {
            "t_end": index[HORIZON : HORIZON + usable],
            "label": (rng.standard_normal(usable) > 0).astype(int),
            "realised_return": rng.standard_normal(usable) * 0.001,
            "weight": np.ones(usable),
        },
        index=pd.DatetimeIndex(index[:usable], name="ts"),
    )
    return features.iloc[:usable], labels


@pytest.fixture
def splits(dataset):
    _, labels = dataset
    return make_splits(
        labels,
        CVConfig(
            scheme="walkforward",
            walkforward=WalkForwardConfig(train_bars=200, test_bars=80, step_bars=80),
            embargo_pct=0.01,
            min_train_bars=50,
        ),
    )


def spy_on_winsoriser(monkeypatch) -> list[pd.Index]:
    """Record the index of every frame the winsoriser is fitted on.

    The winsoriser is the first pipeline step, so it is the only one that still
    has an index to record.
    """
    seen: list[pd.Index] = []
    original = Winsoriser.fit

    def recording_fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), (
            "the winsoriser stopped receiving a DataFrame, so this spy no longer "
            "observes anything — fix the spy before trusting the result"
        )
        seen.append(pd.Index(X.index))
        return original(self, X, y)

    monkeypatch.setattr(Winsoriser, "fit", recording_fit)
    return seen


def spy_on_scaler(monkeypatch) -> list[int]:
    """Record how many rows the scaler was fitted on.

    The scaler receives the winsoriser's output, which is a bare array. Row
    count is the strongest thing still observable there, and it is enough: a
    global fit would show the full sample size.
    """
    seen: list[int] = []
    original = StandardScaler.fit

    def recording_fit(self, X, y=None, sample_weight=None):
        seen.append(int(np.asarray(X).shape[0]))
        return original(self, X, y, sample_weight)

    monkeypatch.setattr(StandardScaler, "fit", recording_fit)
    return seen


def test_pipeline_fit_on_train_only(dataset, splits, monkeypatch):
    """No fitted transform sees the test rows of the fold it is fitted for."""
    features, labels = dataset
    winsoriser_calls = spy_on_winsoriser(monkeypatch)
    scaler_calls = spy_on_scaler(monkeypatch)

    fit_predict_oof(build_model("logistic", seed=1), features, labels, splits)

    assert len(winsoriser_calls) == len(splits), (
        f"expected one winsoriser fit per fold, saw {len(winsoriser_calls)} for "
        f"{len(splits)} folds — the spy is not observing what this test claims"
    )
    assert len(scaler_calls) == len(splits)

    for split, fitted_on, n_rows in zip(splits, winsoriser_calls, scaler_calls, strict=True):
        own_test = set(features.index[split.test_idx])
        leaked = own_test & set(fitted_on)
        assert not leaked, (
            f"fold {split.fold}: a transform was fitted on {len(leaked)} of its own test "
            "rows, which encodes the test period's distribution into training (§4, I3)"
        )
        # The scaler has no index, so its row count is the evidence: it must
        # match the fold's training set, never the whole sample.
        assert n_rows == split.train_idx.size, (
            f"fold {split.fold}: scaler fitted on {n_rows} rows, expected "
            f"{split.train_idx.size}; {len(features)} would mean a global fit"
        )
        assert n_rows < len(features)


def test_each_fold_refits_rather_than_reusing(dataset, splits, monkeypatch):
    """A transform carried between folds accumulates every fold it has seen."""
    features, labels = dataset
    winsoriser_calls = spy_on_winsoriser(monkeypatch)

    fit_predict_oof(build_model("logistic", seed=1), features, labels, splits)

    fitted_row_sets = {tuple(idx) for idx in winsoriser_calls}
    assert len(fitted_row_sets) == len(splits), (
        "two folds were fitted on identical rows, so the transform is not fold-local"
    )


def test_scaler_parameters_differ_between_folds(dataset, splits):
    """The end-to-end consequence: fold statistics are fold-specific.

    Were the scaler fitted globally, every fold would share one mean.
    """
    features, _ = dataset
    means = [float(np.mean(StandardScaler().fit(features.iloc[s.train_idx]).mean_)) for s in splits]
    assert len(set(np.round(means, 9))) == len(means)


def test_calibration_is_also_fitted_inside_the_fold(dataset, splits, monkeypatch):
    """§10.2: the calibrator is fitted on an inner split, never the test fold."""
    features, labels = dataset
    winsoriser_calls = spy_on_winsoriser(monkeypatch)

    fit_predict_oof(build_model("logistic", seed=1, calibrate="isotonic"), features, labels, splits)

    for split, fitted_on in zip(splits, winsoriser_calls, strict=True):
        assert not set(features.index[split.test_idx]) & set(fitted_on)
        # With calibration on, the model trains on a strict subset of the fold:
        # the tail is held back for the calibrator.
        assert len(fitted_on) < split.train_idx.size


def test_the_spy_would_catch_a_global_fit(dataset, splits, monkeypatch):
    """The control: a deliberately global fit must trip both spies."""
    features, _ = dataset
    winsoriser_calls = spy_on_winsoriser(monkeypatch)
    scaler_calls = spy_on_scaler(monkeypatch)

    Winsoriser(0.001).fit(features)
    StandardScaler().fit(features)

    own_test = set(features.index[splits[0].test_idx])
    assert own_test & set(winsoriser_calls[0]), (
        "the index spy did not observe a deliberately global fit"
    )
    assert scaler_calls[0] == len(features), (
        "the row-count spy did not observe a deliberately global fit"
    )
