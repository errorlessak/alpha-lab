"""Shuffled labels must produce no measurable edge (§12.3.5).

The integration-level canary. Everything upstream — causal features, purged
folds, fit-on-train-only pipelines — is checked in isolation elsewhere. This
runs the whole thing end to end on labels whose relationship to the features
has been destroyed, and asserts the result is indistinguishable from zero.

If it is not, something in the stack is manufacturing signal from nothing, and
every number the system produces is suspect regardless of which component the
bug is in.

§12.3 phrases the assertion in terms of the out-of-sample Sharpe confidence
interval. There is no backtester until Phase 7, so the same claim is made about
the **information coefficient**: its bootstrap confidence interval must contain
zero. The Sharpe form is added in Phase 8 when the cost model exists — the
`information_coefficient` here is the same quantity the leaderboard ranks on
today, so the canary is testing the metric that is actually in use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacore.config.schema import CVConfig, WalkForwardConfig
from alphacore.cv import make_splits
from alphacore.eval.classification import information_coefficient, roc_auc
from alphacore.models import build_model, fit_predict_oof
from alphacore.runtime.determinism import deterministic_rng

N = 900
HORIZON = 5
SEED = 20260727
BOOTSTRAP_SAMPLES = 400
BLOCK = 20
"""Block length for the bootstrap. Resampling single observations would treat
overlapping labels as independent and produce an interval far too narrow — the
same mistake the sample weights in §8.2 exist to correct."""


def make_dataset(*, signal: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features and labels with a controllable amount of real edge.

    Args:
        signal: Weight of the genuine relationship. Zero gives pure noise.
        seed: RNG seed.

    Returns:
        `(features, labels)`, aligned.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2015-01-01", periods=N, freq="D", tz="UTC")
    features = pd.DataFrame(
        rng.standard_normal((N, 5)),
        index=index,
        columns=[f"f{i}" for i in range(5)],
    )
    score = features["f0"].to_numpy() * signal + rng.standard_normal(N)
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


def folds(labels: pd.DataFrame):
    """Walk-forward folds over a label frame."""
    return make_splits(
        labels,
        CVConfig(
            scheme="walkforward",
            walkforward=WalkForwardConfig(train_bars=300, test_bars=150, step_bars=150),
            embargo_pct=0.01,
            min_train_bars=100,
        ),
    )


def block_bootstrap_ci(
    prediction: np.ndarray,
    forward: np.ndarray,
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
    block: int = BLOCK,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for the information coefficient.

    Resamples contiguous blocks rather than individual observations, so the
    serial dependence that overlapping labels create is preserved.

    Args:
        prediction: Model scores.
        forward: Realised forward returns.
        seed: RNG seed.
        samples: Bootstrap replicates.
        block: Block length in observations.

    Returns:
        The 2.5th and 97.5th percentile of the resampled IC.
    """
    rng = deterministic_rng(seed, "ic_bootstrap")
    n = len(prediction)
    n_blocks = max(1, n // block)
    stats = []
    for _ in range(samples):
        starts = rng.integers(0, max(1, n - block), size=n_blocks)
        picks = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
        sampled_pred = prediction[picks]
        sampled_fwd = forward[picks]
        if np.ptp(sampled_pred) == 0 or np.ptp(sampled_fwd) == 0:
            continue
        stats.append(information_coefficient(sampled_pred, sampled_fwd)[0])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


@pytest.fixture(scope="module")
def shuffled_oof():
    """Out-of-fold predictions from the shuffled-label model on real features."""
    features, labels = make_dataset(signal=0.8, seed=SEED)
    model = build_model("baseline_shuffled", seed=SEED)
    return fit_predict_oof(model, features, labels, folds(labels))


def test_shuffled_labels_have_no_edge(shuffled_oof):
    """The canary: the IC confidence interval must contain zero (§12.3.5)."""
    frame = shuffled_oof.frame
    low, high = block_bootstrap_ci(
        frame["proba"].to_numpy(dtype=float),
        frame["realised_return"].to_numpy(dtype=float),
        seed=SEED,
    )
    assert low <= 0.0 <= high, (
        f"shuffled labels produced an IC interval of [{low:.4f}, {high:.4f}], which "
        "excludes zero. Something upstream is manufacturing signal from noise."
    )


def test_shuffled_labels_score_near_chance_auc(shuffled_oof):
    frame = shuffled_oof.frame
    auc = roc_auc(frame["label"].to_numpy(), frame["proba"].to_numpy(dtype=float))
    assert 0.4 < auc < 0.6, f"shuffled-label AUC of {auc:.3f} is not chance"


def test_shuffled_labels_score_near_chance_accuracy(shuffled_oof):
    frame = shuffled_oof.frame
    accuracy = float((frame["prediction"] == frame["label"]).mean())
    # §19: real short-horizon edges live at 0.51-0.54. Shuffled labels must
    # not reach even that.
    assert 0.4 < accuracy < 0.6


def test_the_canary_can_detect_a_real_edge():
    """The control.

    An honest model on the same features, with the labels left intact, must
    produce an IC interval that excludes zero. Without this, a broken harness
    would report "no edge" for everything and the canary above would be
    meaningless.
    """
    features, labels = make_dataset(signal=1.5, seed=SEED)
    oof = fit_predict_oof(build_model("logistic", seed=SEED), features, labels, folds(labels))
    frame = oof.frame

    low, high = block_bootstrap_ci(
        frame["proba"].to_numpy(dtype=float),
        frame["realised_return"].to_numpy(dtype=float),
        seed=SEED,
    )
    assert low > 0.0, (
        f"a deliberately strong signal produced an IC interval of [{low:.4f}, {high:.4f}] "
        "that still contains zero — the harness cannot detect an edge at all"
    )


def test_the_honest_model_beats_the_shuffled_one():
    """The comparison §10.1 asks to be reported prominently."""
    features, labels = make_dataset(signal=1.5, seed=SEED)
    splits = folds(labels)

    honest = fit_predict_oof(build_model("logistic", seed=SEED), features, labels, splits).frame
    shuffled = fit_predict_oof(
        build_model("baseline_shuffled", seed=SEED), features, labels, splits
    ).frame

    honest_ic = information_coefficient(
        honest["proba"].to_numpy(dtype=float), honest["realised_return"].to_numpy(dtype=float)
    )[0]
    shuffled_ic = information_coefficient(
        shuffled["proba"].to_numpy(dtype=float),
        shuffled["realised_return"].to_numpy(dtype=float),
    )[0]
    assert honest_ic > shuffled_ic
