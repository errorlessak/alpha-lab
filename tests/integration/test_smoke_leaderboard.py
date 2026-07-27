"""Bars to leaderboard on the committed fixture (§15, Phase 5 exit).

The first test that runs the phases built so far end to end: ingest the
committed CSV, build features, label with a triple barrier, cut purged
walk-forward folds, and score every candidate plus every mandatory baseline
out of fold.

Offline by construction, and timed — §15 requires every model to train on the
smoke config in under sixty seconds.
"""

from __future__ import annotations

import json
import time

import pytest

from alphacore.config import load_config
from alphacore.config.schema import ModelsConfig, WalkForwardConfig
from alphacore.cv import make_splits
from alphacore.data import read_bars
from alphacore.features import build_features
from alphacore.labels import build_labels, to_binary
from alphacore.models import MANDATORY_BASELINES, build_leaderboard, validate_leaderboard
from alphacore.pipelines import ingest_symbols
from alphacore.pipelines.ingest import resolve_calendar

TIME_BUDGET_S = 60.0


@pytest.fixture(scope="module")
def smoke_config(repo_root):
    config = load_config(repo_root / "configs" / "experiments" / "smoke.yaml", use_env=False)
    data = config.data.model_copy(update={"csv_dir": str(repo_root / "samples")})
    return config.model_copy(update={"data": data})


@pytest.fixture(scope="module")
def prepared(smoke_config, tmp_path_factory):
    """Ingest, build features, and label — the inputs a leaderboard needs."""
    lake = tmp_path_factory.mktemp("lake")
    ingest_symbols(smoke_config, data_dir=lake)
    bars = read_bars(root=lake, provider="csv", symbol="SYNTH", interval=smoke_config.data.interval)

    features, manifest = build_features(
        bars,
        smoke_config.features,
        interval=smoke_config.data.interval,
        calendar=resolve_calendar(smoke_config.data),
    )
    labels = build_labels(bars.loc[features.index], smoke_config.labels)
    # Triple-barrier labels are {-1, 0, +1}; the models are binary. The
    # reduction is explicit so that what "accuracy" means stays visible.
    labels = to_binary(labels, method="direction")

    common = features.index.intersection(labels.index)
    return features.loc[common], labels.loc[common], manifest


def test_the_fixture_survives_features_and_labelling(prepared):
    features, labels, manifest = prepared
    assert len(features) > 1_000
    assert features.notna().all().all()
    assert manifest.n_columns > 5
    assert labels.index.equals(features.index)
    assert (labels["weight"] > 0).all()
    assert set(labels["label"].unique()) == {0, 1}


def test_the_annualisation_factor_is_recorded(prepared):
    _, _, manifest = prepared
    # §11.1: a metric annualised by an undocumented factor is unfalsifiable.
    assert manifest.periods_per_year > 250
    assert manifest.calendar == "XNYS"


@pytest.fixture(scope="module")
def scored(prepared, smoke_config):
    features, labels, _ = prepared
    splits = make_splits(
        labels,
        smoke_config.cv.model_copy(
            update={"walkforward": WalkForwardConfig(train_bars=500, test_bars=250, step_bars=250)}
        ),
    )
    config = ModelsConfig(candidates=("logistic",), calibrate="sigmoid")

    started = time.perf_counter()
    rows = build_leaderboard(features, labels, splits, config, seed=smoke_config.meta.seed)
    return rows, time.perf_counter() - started, splits


def test_every_model_scores_within_the_time_budget(scored):
    rows, elapsed, _ = scored
    assert elapsed < TIME_BUDGET_S, (
        f"scoring {len(rows)} models took {elapsed:.1f}s, over the {TIME_BUDGET_S}s budget "
        "§15 sets for the smoke config"
    )


def test_the_leaderboard_carries_every_candidate_and_baseline(scored):
    rows, _, _ = scored
    names = {row["model"] for row in rows}
    assert "logistic" in names
    assert set(MANDATORY_BASELINES) <= names


def test_the_leaderboard_validates_against_its_schema(scored):
    rows, _, _ = scored
    parsed = validate_leaderboard(rows)
    assert len(parsed) == len(rows)
    assert [p.rank for p in parsed] == list(range(1, len(rows) + 1))


def test_the_leaderboard_is_json_serialisable(scored):
    rows, _, _ = scored
    assert json.loads(json.dumps(rows, sort_keys=True)) == rows


def test_no_model_finds_an_edge_in_a_random_walk(scored):
    """The honest headline §19 asks for.

    `samples/SYNTH_1d.csv` is a seeded random walk with no predictable
    structure by construction. Any model reporting a strong information
    coefficient on it has found something that is not there — which would mean
    a leak somewhere upstream, not a discovery.
    """
    rows, _, _ = scored
    for row in rows:
        tstat = row.get("ic_tstat")
        if tstat is None:
            continue
        assert abs(tstat) < 4.0, (
            f"{row['model']} reports an IC t-stat of {tstat:.2f} on a synthetic random "
            "walk. There is no edge to find, so this indicates leakage."
        )


def test_every_row_is_flagged_as_uncosted(scored):
    rows, _, _ = scored
    # No backtester until Phase 7, so the economic columns are honestly absent
    # rather than filled with a number someone might read.
    assert all("NO_BACKTEST" in row["flags"] for row in rows)
    assert all(row["oos_net_sharpe"] is None for row in rows)
