"""The tearsheet artifact, its exports, the research note, and SHAP (§11.4, §9)."""

from __future__ import annotations

import datetime as dt
import json
import time

import numpy as np
import pandas as pd
import pytest

from alphacore.errors import ArtifactError, EvaluationError, ModelError
from alphacore.explain import (
    ADDITIVITY_TOLERANCE,
    explain_prediction,
    shap_summary,
    verify_additivity,
)
from alphacore.models import build_model
from alphacore.models.leaderboard import LeaderboardRow
from alphacore.report import (
    TEARSHEET_SCHEMA_VERSION,
    BacktestSection,
    CVSection,
    DatasetSection,
    ExplainSection,
    FeaturesSection,
    FoldSection,
    LabelsSection,
    ManifestSection,
    OverfittingSection,
    Tearsheet,
    export_frame,
    export_run_artifacts,
    export_table,
    lttb_downsample,
    render_research_markdown,
    validate_tearsheet,
)
from alphacore.runtime import ArtifactStore


def make_tearsheet(**overrides) -> Tearsheet:
    base = {
        "run_id": "2026-07-27T14-22-01Z_smoke_a3f91c",
        "created_at": dt.datetime(2026, 7, 27, 14, 22, 1, tzinfo=dt.UTC),
        "manifest": ManifestSection(
            config_hash="a3f91c7e2b4d5f60",
            data_hash="7e2b",
            code_version="abc1234",
            seed=20260727,
            python="3.12.13",
            packages={"lightgbm": "4.7.0"},
            duration_s=412.7,
            host="cyborg1326",
        ),
        "dataset": DatasetSection(
            symbols=("SYNTH",),
            interval="1d",
            calendar="XNYS",
            span=(dt.date(2015, 1, 2), dt.date(2024, 12, 31)),
            n_bars=2516,
            quality="PASS",
            periods_per_year=251.6,
        ),
        "features": FeaturesSection(
            n_built=214, n_after_prune=96, max_lookback=282, families={"trend": 40}
        ),
        "labels": LabelsSection(
            method="triple_barrier",
            horizon_bars=10,
            class_balance={"0": 0.49, "1": 0.51},
            mean_uniqueness=0.31,
        ),
        "cv": CVSection(
            scheme="walkforward",
            embargo_pct=0.01,
            folds=(
                FoldSection(
                    fold=0,
                    train_span=(
                        dt.datetime(2015, 1, 2, tzinfo=dt.UTC),
                        dt.datetime(2019, 1, 2, tzinfo=dt.UTC),
                    ),
                    test_span=(
                        dt.datetime(2019, 1, 3, tzinfo=dt.UTC),
                        dt.datetime(2020, 1, 2, tzinfo=dt.UTC),
                    ),
                    n_train=1000,
                    n_test=252,
                    purged=9,
                    embargoed=25,
                ),
            ),
        ),
        "leaderboard": (
            LeaderboardRow(
                model="lightgbm",
                rank=1,
                n_predictions=1000,
                n_folds=4,
                n_trials=60,
                oos_net_sharpe=0.71,
                sharpe_ci=(0.12, 1.28),
                dsr=0.34,
                psr=0.81,
                ic=0.031,
                ic_tstat=2.1,
                auc=0.534,
                brier=0.248,
                flags=["LOW_DSR"],
            ),
            LeaderboardRow(
                model="baseline_shuffled",
                rank=2,
                n_predictions=1000,
                n_folds=4,
                n_trials=1,
                ic=0.002,
                ic_tstat=0.1,
                auc=0.501,
                flags=["SHUFFLED_LABELS"],
            ),
        ),
        "selected_model": "lightgbm",
        "backtest": BacktestSection(
            metrics={"sharpe": 0.71, "max_drawdown": -0.19},
            trade_metrics={"n_trades": 120.0, "win_rate": 0.52},
            cost_sensitivity=({"bps": 0.0, "sharpe": 1.02}, {"bps": 10.0, "sharpe": -0.3}),
            break_even_bps=6.4,
            round_trip_bps=3.0,
        ),
        "overfitting": OverfittingSection(
            pbo=0.41, n_configs_evaluated=312, n_splits=12870, shuffled_label_ic=0.002
        ),
        "explain": ExplainSection(
            shap_summary_ref="shap_summary.parquet",
            top_features=("rsi_window14",),
            n_explained=2000,
        ),
        "warnings": ("LOW_DSR",),
    }
    return Tearsheet.model_validate({**base, **overrides})


# --- tearsheet schema ------------------------------------------------------


def test_a_complete_tearsheet_validates():
    sheet = make_tearsheet()
    assert sheet.schema_version == TEARSHEET_SCHEMA_VERSION
    assert sheet.selected_model == "lightgbm"


def test_the_tearsheet_round_trips_through_json():
    sheet = make_tearsheet()
    payload = json.loads(json.dumps(sheet.model_dump(mode="json"), sort_keys=True))
    assert validate_tearsheet(payload) == sheet


def test_an_unknown_key_is_rejected():
    payload = make_tearsheet().model_dump(mode="json")
    payload["sharpe"] = 3.0
    with pytest.raises(EvaluationError, match="does not match schema"):
        validate_tearsheet(payload)


def test_the_tearsheet_is_frozen():
    sheet = make_tearsheet()
    with pytest.raises(ValueError, match="frozen"):
        sheet.selected_model = "xgboost"


def test_the_disclaimer_is_always_present():
    assert "Not investment advice" in make_tearsheet().disclaimer


def test_the_headline_carries_the_qualifiers_not_just_the_sharpe():
    headline = make_tearsheet().headline
    # §11.2: the break-even cost sits next to the headline Sharpe, always.
    assert headline["oos_net_sharpe"] == 0.71
    assert headline["dsr"] == 0.34
    assert headline["break_even_bps"] == 6.4
    assert headline["assumed_round_trip_bps"] == 3.0
    assert headline["pbo"] == 0.41
    assert headline["sharpe_ci"] == [0.12, 1.28]


def test_the_annualisation_factor_is_recorded():
    assert make_tearsheet().dataset.periods_per_year == 251.6


# --- the PBO verdict -------------------------------------------------------


def test_a_high_pbo_says_worse_than_chance_in_plain_language():
    sheet = make_tearsheet(overfitting=OverfittingSection(pbo=0.62))
    assert "worse than chance" in sheet.overfitting.verdict


def test_a_low_pbo_says_it_holds_up():
    sheet = make_tearsheet(overfitting=OverfittingSection(pbo=0.1))
    assert "holds up" in sheet.overfitting.verdict


def test_an_uncomputed_pbo_says_so_rather_than_implying_zero():
    sheet = make_tearsheet(overfitting=OverfittingSection())
    assert "not computed" in sheet.overfitting.verdict


# --- exporters -------------------------------------------------------------


@pytest.fixture
def run(tmp_path):
    return ArtifactStore(tmp_path / "runs").create_run("test-run")


def test_a_frame_round_trips_through_parquet(run):
    frame = pd.DataFrame(
        {"equity": [100.0, 101.0]},
        index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"], tz="UTC", name="ts"),
    )
    name = export_frame(run, "equity.parquet", frame)
    restored = pd.read_parquet(run.artifact_path(name))
    pd.testing.assert_frame_equal(restored, frame)


def test_exports_are_write_once(run):
    frame = pd.DataFrame({"a": [1.0]})
    export_frame(run, "equity.parquet", frame)
    with pytest.raises(ArtifactError, match="immutable"):
        export_frame(run, "equity.parquet", frame)


def test_a_csv_export_uses_lf_line_endings(run):
    export_table(run, "leaderboard.csv", pd.DataFrame({"model": ["a"], "rank": [1]}))
    assert b"\r\n" not in run.read_bytes("leaderboard.csv")


def test_export_run_artifacts_writes_the_standard_set(run):
    index = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=5, tz="UTC"), name="ts")
    refs = export_run_artifacts(
        run,
        equity=pd.DataFrame({"equity": np.linspace(100, 105, 5)}, index=index),
        drawdown=pd.DataFrame({"drawdown": np.zeros(5)}, index=index),
        trades=pd.DataFrame({"pnl": [1.0, -1.0]}),
        leaderboard=[{"model": "lightgbm", "rank": 1}],
    )
    for key in ("equity", "drawdown", "trades", "trades_csv", "leaderboard_csv"):
        assert key in refs
        assert run.exists(refs[key])


# --- LTTB ------------------------------------------------------------------


def test_downsampling_keeps_the_endpoints():
    index = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=1_000, tz="UTC"), name="ts")
    frame = pd.DataFrame(
        {"equity": np.random.default_rng(1).standard_normal(1_000).cumsum()}, index=index
    )
    reduced = lttb_downsample(frame, "equity", width=100)
    assert len(reduced) <= 100
    assert reduced.index[0] == frame.index[0]
    assert reduced.index[-1] == frame.index[-1]


def test_downsampling_preserves_a_one_bar_spike():
    # Naive decimation drops the spike, which is exactly the bar a reviewer
    # needs to see.
    values = np.zeros(1_000)
    values[500] = 100.0
    index = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=1_000, tz="UTC"), name="ts")
    frame = pd.DataFrame({"equity": values}, index=index)
    reduced = lttb_downsample(frame, "equity", width=50)
    assert reduced["equity"].max() == pytest.approx(100.0)


def test_a_short_series_is_returned_unchanged():
    frame = pd.DataFrame({"equity": [1.0, 2.0, 3.0]})
    assert len(lttb_downsample(frame, "equity", width=100)) == 3


def test_downsampling_an_unknown_column_is_an_error():
    with pytest.raises(ArtifactError, match="missing column"):
        lttb_downsample(pd.DataFrame({"a": [1.0]}), "b", width=10)


# --- research note ---------------------------------------------------------


def test_the_research_note_leads_with_what_is_not_established():
    markdown = render_research_markdown(make_tearsheet())
    intro = markdown.split("## Headline")[0]
    assert "does not establish an edge" in intro


def test_the_research_note_reports_the_deflated_sharpe_beside_the_raw_one():
    markdown = render_research_markdown(make_tearsheet())
    assert "deflated Sharpe (DSR)" in markdown
    assert "0.340" in markdown or "0.34" in markdown


def test_the_research_note_calls_out_the_shuffled_baseline():
    markdown = render_research_markdown(make_tearsheet())
    assert "shuffled-label baseline" in markdown


def test_the_research_note_explains_every_warning():
    markdown = render_research_markdown(make_tearsheet(warnings=("UNCOSTED", "LOW_DSR")))
    assert "charged no costs" in markdown
    assert "deflated Sharpe ratio is low" in markdown


def test_the_research_note_warns_when_costs_exceed_break_even():
    sheet = make_tearsheet(
        backtest=BacktestSection(break_even_bps=1.0, round_trip_bps=5.0),
    )
    assert "does not survive its own cost model" in render_research_markdown(sheet)


def test_the_research_note_records_provenance():
    markdown = render_research_markdown(make_tearsheet())
    assert "a3f91c7e2b4d5f60" in markdown
    assert "251.60" in markdown


def test_the_research_note_is_deterministic():
    sheet = make_tearsheet()
    assert render_research_markdown(sheet) == render_research_markdown(sheet)


def test_the_research_note_carries_the_disclaimer():
    assert "Not investment advice" in render_research_markdown(make_tearsheet())


# --- SHAP ------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(20260727)
    n = 800
    index = pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC")
    features = pd.DataFrame(
        rng.standard_normal((n, 5)), index=index, columns=[f"f{i}" for i in range(5)]
    )
    y = pd.Series(
        (features["f0"].to_numpy() * 1.2 + rng.standard_normal(n) > 0).astype(int), index=index
    )
    model = build_model("lightgbm", seed=1).fit(features, y)
    return model, features


def test_shap_values_sum_to_the_model_output(fitted):
    """§15 Phase 9 exit: values sum to f(x) - E[f(x)] within tolerance."""
    model, features = fitted
    result = shap_summary(model, features, seed=1)
    discrepancy = verify_additivity(result)
    assert discrepancy <= ADDITIVITY_TOLERANCE


def test_shap_runs_inside_the_time_budget(fitted):
    model, features = fitted
    started = time.perf_counter()
    shap_summary(model, features, seed=1)
    assert time.perf_counter() - started < 30.0


def test_shap_ranks_the_informative_feature_first(fitted):
    model, features = fitted
    result = shap_summary(model, features, seed=1)
    # f0 is the only feature the label depends on.
    assert result.top_features(1) == ("f0",)


def test_the_summary_frame_has_one_row_per_feature(fitted):
    model, features = fitted
    summary = shap_summary(model, features, seed=1).summary_frame()
    assert set(summary.index) == set(features.columns)
    assert (summary["mean_abs_shap"] >= 0).all()
    assert summary["mean_abs_shap"].is_monotonic_decreasing


def test_the_direction_column_signs_the_relationship(fitted):
    model, features = fitted
    summary = shap_summary(model, features, seed=1).summary_frame()
    # Higher f0 means a higher probability of the positive class.
    assert summary.loc["f0", "direction"] > 0


def test_sampling_is_seeded_and_reproducible(fitted):
    model, features = fitted
    first = shap_summary(model, features, seed=3, sample=200)
    second = shap_summary(model, features, seed=3, sample=200)
    pd.testing.assert_frame_equal(first.values, second.values)


def test_sampling_caps_the_explained_rows(fitted):
    model, features = fitted
    result = shap_summary(model, features, seed=1, sample=100)
    assert len(result.values) == 100
    assert result.n_available == len(features)


def test_a_single_prediction_reconstructs_its_own_output(fitted):
    model, features = fitted
    result = shap_summary(model, features, seed=1)
    ts = result.values.index[5]
    waterfall = explain_prediction(result, ts, n=5)

    assert len(waterfall["contributions"]) == 5
    # The UI displays the identity rather than asking for trust.
    full = float(result.values.loc[ts].sum() + result.base_value)
    assert waterfall["output"] == pytest.approx(full)


def test_explaining_an_unsampled_row_is_an_error(fitted):
    model, features = fitted
    result = shap_summary(model, features, seed=1, sample=50)
    missing = next(ts for ts in features.index if ts not in result.values.index)
    with pytest.raises(ModelError, match="not in the explained sample"):
        explain_prediction(result, missing)


def test_a_linear_model_is_refused_rather_than_forced(fitted):
    _, features = fitted
    rng = np.random.default_rng(2)
    y = pd.Series(rng.integers(0, 2, len(features)), index=features.index)
    linear = build_model("logistic", seed=1).fit(features, y)
    with pytest.raises(ModelError, match="not a tree model"):
        shap_summary(linear, features, seed=1)


def test_an_unfitted_model_is_refused(fitted):
    _, features = fitted
    with pytest.raises(ModelError, match="not fitted"):
        shap_summary(build_model("lightgbm", seed=1), features, seed=1)


def test_an_empty_frame_is_refused(fitted):
    model, features = fitted
    with pytest.raises(ModelError, match="empty"):
        shap_summary(model, features.iloc[:0], seed=1)


def test_additivity_failure_is_reported_not_swallowed(fitted):
    model, features = fitted
    result = shap_summary(model, features, seed=1)
    broken = type(result)(
        values=result.values * 2.0,
        base_value=result.base_value,
        features=result.features,
        raw_output=result.raw_output,
        model=result.model,
        n_available=result.n_available,
    )
    with pytest.raises(ModelError, match="additivity failed"):
        verify_additivity(broken)
