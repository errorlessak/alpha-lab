"""The feature registry, transforms, pipeline, and selection (§7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphacore.config.schema import FeaturesConfig, FeatureSpecConfig, Interval, PruneConfig
from alphacore.errors import FeatureError, LeakageError
from alphacore.features import registry as reg
from alphacore.features.pipeline import build_features, expand_params
from alphacore.features.selection import correlation_prune, prune_correlated
from alphacore.features.transforms import add_lags, drop_warmup
from alphacore.schemas.bars import normalise_bars, validate_bars


@pytest.fixture
def bars() -> pd.DataFrame:
    """400 daily bars, enough for a 100-bar lookback plus room to spare."""
    rng = np.random.default_rng(20260727)
    n = 400
    ts = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    open_ = np.concatenate(([close[0]], close[:-1]))
    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    frame = pd.DataFrame(
        {
            "ts": ts,
            "open": open_,
            "high": body_high * (1.0 + np.abs(rng.standard_normal(n)) * 0.004),
            "low": body_low * (1.0 - np.abs(rng.standard_normal(n)) * 0.004),
            "close": close,
            "volume": 1e6 * (1.0 + np.abs(rng.standard_normal(n)) * 0.3),
            "adj_factor": 1.0,
            "symbol": "SYNTH",
        }
    )
    return validate_bars(normalise_bars(frame))


def features_config(**overrides) -> FeaturesConfig:
    base = {
        "specs": (
            FeatureSpecConfig(name="sma", params={"window": [10, 20]}),
            FeatureSpecConfig(name="rsi", params={"window": 14}),
        ),
        "prune": PruneConfig(method="none"),
    }
    return FeaturesConfig.model_validate({**base, **overrides})


# --- registry --------------------------------------------------------------


def test_registry_is_not_empty():
    assert len(reg.registered_names()) > 15


def test_registered_names_are_sorted():
    names = reg.registered_names()
    assert list(names) == sorted(names)


@pytest.mark.parametrize(
    "name",
    [
        "sma",
        "ema",
        "wma",
        "hma",
        "rsi",
        "macd",
        "roc",
        "momentum",
        "ppo",
        "atr",
        "true_range",
        "realised_vol",
        "bollinger",
        "rolling_std",
        "parkinson",
        "garman_klass",
        "obv",
        "vwap_dist",
        "cmf",
        "volume_ratio",
        "dollar_volume",
        "log_return",
        "simple_return",
        "rolling_stat",
        "zscore",
        "percentile_rank",
        "fracdiff",
    ],
)
def test_every_feature_named_in_the_spec_is_registered(name):
    assert name in reg.registered_names()


def test_every_definition_declares_its_metadata():
    for name in reg.registered_names():
        definition = reg.get(name)
        assert definition.description, name
        assert definition.family in reg.FAMILIES, name
        assert definition.inputs, name
        assert definition.inputs <= reg.ALLOWED_INPUTS, name


def test_resolving_produces_a_frozen_spec_with_an_integer_lookback():
    spec = reg.resolve("sma", {"window": 20})
    assert spec.name == "sma"
    assert spec.params == {"window": 20}
    assert spec.lookback == 19
    with pytest.raises(AttributeError):
        spec.lookback = 1


def test_lookback_tracks_the_parameters():
    assert (
        reg.resolve("sma", {"window": 100}).lookback > reg.resolve("sma", {"window": 10}).lookback
    )


def test_resolving_an_unknown_feature_lists_the_known_ones():
    with pytest.raises(FeatureError, match="Unknown feature"):
        reg.resolve("bollingr", {"window": 20})


def test_resolving_with_an_unknown_parameter_is_rejected():
    with pytest.raises(FeatureError, match="unexpected parameter"):
        reg.resolve("sma", {"windwo": 20})


def test_no_registered_feature_reads_a_label_column():
    # I5: a feature whose computation graph touches the label is a leak by
    # construction, so the registry cannot express one.
    for name in reg.registered_names():
        assert not (reg.get(name).inputs & reg.FORBIDDEN_INPUTS), name


def test_registering_a_feature_that_reads_a_label_column_is_rejected():
    with pytest.raises(LeakageError, match="label"):

        @reg.register(
            name="cheat",
            family="momentum",
            description="Reads the answer.",
            inputs=frozenset({"close", "label"}),
            lookback=lambda _params: 0,
        )
        def _cheat(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
            return pd.DataFrame(index=bars.index)


def test_registering_a_duplicate_name_is_rejected():
    with pytest.raises(FeatureError, match="already registered"):

        @reg.register(
            name="sma",
            family="trend",
            description="Duplicate.",
            inputs=frozenset({"close"}),
            lookback=lambda _params: 0,
        )
        def _duplicate(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
            return pd.DataFrame(index=bars.index)


# --- param expansion -------------------------------------------------------


def test_expand_params_takes_the_cartesian_product():
    expanded = expand_params({"window": [10, 20], "k": [2.0]})
    assert expanded == [{"k": 2.0, "window": 10}, {"k": 2.0, "window": 20}]


def test_expand_params_leaves_scalars_alone():
    assert expand_params({"window": 14}) == [{"window": 14}]


def test_expand_params_of_nothing_is_one_empty_set():
    assert expand_params({}) == [{}]


def test_expand_params_is_order_independent():
    # Iteration order must never reach a column name (§4, I8).
    assert expand_params({"a": [1, 2], "b": [3]}) == expand_params({"b": [3], "a": [1, 2]})


# --- pipeline --------------------------------------------------------------


def test_build_produces_one_column_per_expanded_spec(bars):
    frame, manifest = build_features(bars, features_config(), interval=Interval.D1)
    assert "sma_window10" in frame.columns
    assert "sma_window20" in frame.columns
    assert "rsi_window14" in frame.columns
    assert set(manifest.columns) == set(frame.columns)


def test_multi_output_features_get_suffixed_columns(bars):
    config = features_config(
        specs=(FeatureSpecConfig(name="bollinger", params={"window": 20, "k": 2.0}),)
    )
    frame, _ = build_features(bars, config, interval=Interval.D1)
    assert "bollinger_k2.0_window20_upper" in frame.columns
    assert "bollinger_k2.0_window20_pct_b" in frame.columns


def test_column_names_are_deterministic(bars):
    first, _ = build_features(bars, features_config(), interval=Interval.D1)
    second, _ = build_features(bars, features_config(), interval=Interval.D1)
    assert list(first.columns) == list(second.columns)
    pd.testing.assert_frame_equal(first, second)


def test_the_warm_up_region_is_dropped_not_filled(bars):
    frame, manifest = build_features(bars, features_config(), interval=Interval.D1)
    # Rows are removed, never back-filled (§7.1).
    assert len(frame) == len(bars) - manifest.max_lookback
    assert frame.notna().all().all()


def test_lags_are_applied_and_named(bars):
    frame, _ = build_features(bars, features_config(lags=(1, 5)), interval=Interval.D1)
    assert "sma_window10_lag1" in frame.columns
    assert "sma_window10_lag5" in frame.columns


def test_a_lagged_column_equals_the_shifted_original(bars):
    frame, _ = build_features(bars, features_config(lags=(2,)), interval=Interval.D1)
    np.testing.assert_allclose(
        frame["sma_window10_lag2"].to_numpy()[2:],
        frame["sma_window10"].to_numpy()[:-2],
        atol=1e-12,
    )


def test_lags_extend_the_warm_up(bars):
    without, manifest_without = build_features(bars, features_config(), interval=Interval.D1)
    with_lags, manifest_with = build_features(
        bars, features_config(lags=(20,)), interval=Interval.D1
    )
    assert manifest_with.max_lookback == manifest_without.max_lookback + 20
    assert len(with_lags) < len(without)


def test_manifest_records_provenance_for_every_column(bars):
    _, manifest = build_features(bars, features_config(), interval=Interval.D1)
    entry = manifest.columns["sma_window10"]
    assert entry.feature == "sma"
    assert entry.params == {"window": 10}
    assert entry.family == "trend"
    assert entry.lookback == 9
    assert entry.description


def test_manifest_records_lag_provenance(bars):
    _, manifest = build_features(bars, features_config(lags=(3,)), interval=Interval.D1)
    assert manifest.columns["sma_window10_lag3"].lag == 3
    assert manifest.columns["sma_window10"].lag == 0


def test_manifest_serialises_to_json(bars):
    _, manifest = build_features(bars, features_config(), interval=Interval.D1)
    payload = manifest.model_dump(mode="json")
    assert payload["n_columns"] == len(manifest.columns)


def test_build_rejects_a_spec_naming_an_unknown_feature(bars):
    config = features_config(specs=(FeatureSpecConfig(name="nope", params={}),))
    with pytest.raises(FeatureError, match="Unknown feature"):
        build_features(bars, config, interval=Interval.D1)


def test_build_refuses_to_produce_an_empty_frame(bars):
    config = features_config(specs=(FeatureSpecConfig(name="sma", params={"window": 5000}),))
    with pytest.raises(FeatureError, match="no rows"):
        build_features(bars, config, interval=Interval.D1)


def test_realised_vol_annualisation_comes_from_the_interval(bars):
    daily, _ = build_features(
        bars,
        features_config(specs=(FeatureSpecConfig(name="realised_vol", params={"window": 20}),)),
        interval=Interval.D1,
    )
    hourly, _ = build_features(
        bars,
        features_config(specs=(FeatureSpecConfig(name="realised_vol", params={"window": 20}),)),
        interval=Interval.H1,
        calendar="24/7",
    )
    # Same bars, different declared interval: the annualisation factor moves,
    # so the numbers must differ. Hardcoding 252 is banned (§11.1).
    assert not np.allclose(daily.to_numpy(), hourly.to_numpy())


# --- transforms ------------------------------------------------------------


def test_add_lags_returns_original_and_lagged_columns():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    lagged = add_lags(frame, (1, 2))
    assert list(lagged.columns) == ["a", "a_lag1", "a_lag2"]
    assert lagged["a_lag1"].tolist()[1:] == [1.0, 2.0, 3.0]


def test_add_lags_with_no_lags_is_the_identity():
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    pd.testing.assert_frame_equal(add_lags(frame, ()), frame)


def test_add_lags_rejects_a_non_positive_lag():
    with pytest.raises(FeatureError, match="lag"):
        add_lags(pd.DataFrame({"a": [1.0]}), (0,))


def test_drop_warmup_removes_exactly_the_requested_rows():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert len(drop_warmup(frame, 2)) == 1


def test_drop_warmup_of_zero_is_the_identity():
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    pd.testing.assert_frame_equal(drop_warmup(frame, 0), frame)


# --- selection -------------------------------------------------------------


def test_correlation_prune_drops_a_duplicated_column():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(200)
    frame = pd.DataFrame(
        {"a": base, "b": base + rng.standard_normal(200) * 1e-6, "c": rng.standard_normal(200)}
    )
    kept, dropped = correlation_prune(frame, threshold=0.95)
    assert len(kept.columns) == 2
    assert len(dropped) == 1
    assert "c" in kept.columns


def test_correlation_prune_keeps_everything_below_the_threshold():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({c: rng.standard_normal(300) for c in "abcd"})
    kept, dropped = correlation_prune(frame, threshold=0.95)
    assert len(kept.columns) == 4
    assert dropped == {}


def test_prune_requires_an_explicit_fit_mask():
    # Pruning on all rows fits a selection decision on the test set (I3).
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({c: rng.standard_normal(100) for c in "ab"})
    with pytest.raises(LeakageError, match="fit"):
        prune_correlated(frame, PruneConfig(method="correlation", threshold=0.95), fit_mask=None)


def test_prune_with_method_none_is_a_no_op():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    kept, dropped = prune_correlated(frame, PruneConfig(method="none"), fit_mask=None)
    assert list(kept.columns) == ["a", "b"]
    assert dropped == {}


def test_prune_only_looks_at_the_fit_rows():
    rng = np.random.default_rng(11)
    n = 400
    a = rng.standard_normal(n)
    b = a.copy()
    # b decorrelates from a only in the second half, which the fit mask excludes.
    b[n // 2 :] = rng.standard_normal(n // 2)
    frame = pd.DataFrame({"a": a, "b": b, "c": rng.standard_normal(n)})
    mask = np.zeros(n, dtype=bool)
    mask[: n // 2] = True

    kept, dropped = prune_correlated(
        frame, PruneConfig(method="correlation", threshold=0.95), fit_mask=mask
    )
    assert len(dropped) == 1
    assert set(kept.columns) | set(dropped) == {"a", "b", "c"}


def test_clustered_prune_keeps_one_representative_per_cluster():
    rng = np.random.default_rng(13)
    n = 400
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    frame = pd.DataFrame(
        {
            "x1": x,
            "x2": x + rng.standard_normal(n) * 0.01,
            "x3": x + rng.standard_normal(n) * 0.01,
            "y1": y,
            "y2": y + rng.standard_normal(n) * 0.01,
        }
    )
    kept, dropped = prune_correlated(
        frame,
        PruneConfig(method="clustered_correlation", threshold=0.95),
        fit_mask=np.ones(n, dtype=bool),
    )
    assert len(kept.columns) == 2
    assert len(dropped) == 3


def test_drop_list_records_why_each_column_went():
    # Columns are visited in name order, so of a correlated pair the
    # alphabetically first survives. That rule exists so the survivor does not
    # depend on dict iteration order (§4, I8) — hence the deliberate names.
    rng = np.random.default_rng(17)
    base = rng.standard_normal(300)
    frame = pd.DataFrame({"aaa": base, "zzz": base + rng.standard_normal(300) * 1e-6})
    _, dropped = prune_correlated(
        frame,
        PruneConfig(method="correlation", threshold=0.95),
        fit_mask=np.ones(300, dtype=bool),
    )
    ((column, reason),) = dropped.items()
    assert column == "zzz"
    assert "aaa" in reason
    assert "corr" in reason


def test_the_survivor_of_a_correlated_pair_does_not_depend_on_column_order():
    rng = np.random.default_rng(19)
    base = rng.standard_normal(300)
    partner = base + rng.standard_normal(300) * 1e-6
    forwards = pd.DataFrame({"aaa": base, "zzz": partner})
    backwards = pd.DataFrame({"zzz": partner, "aaa": base})
    assert (
        correlation_prune(forwards, threshold=0.95)[1].keys()
        == correlation_prune(backwards, threshold=0.95)[1].keys()
    )
