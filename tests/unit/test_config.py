"""Configuration schema and loading (§5)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alphacore.config import (
    AlphaConfig,
    Interval,
    config_hash,
    deep_merge,
    env_overrides,
    load_config,
    load_defaults,
    parse_override,
    to_serialisable,
)
from alphacore.errors import ConfigError

MINIMAL = {
    "meta": {"name": "unit", "seed": 1},
    "data": {"symbols": ["SPY"], "start": "2020-01-01", "end": "2021-01-01"},
    "models": {"candidates": ["logistic"]},
}


def write_config(tmp_path: Path, payload: dict, name: str = "cfg.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


# --- merging ---------------------------------------------------------------


def test_deep_merge_recurses_into_mappings():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    assert deep_merge(base, {"a": {"y": 9}}) == {"a": {"x": 1, "y": 9}, "b": 3}


def test_deep_merge_replaces_sequences_wholesale():
    assert deep_merge({"lags": [1, 2, 3]}, {"lags": [5]}) == {"lags": [5]}


def test_deep_merge_does_not_mutate_the_base():
    base = {"a": {"x": 1}}
    deep_merge(base, {"a": {"x": 2}})
    assert base == {"a": {"x": 1}}


# --- loading ---------------------------------------------------------------


def test_packaged_defaults_parse():
    assert "backtest" in load_defaults()


def test_load_config_applies_defaults(tmp_path):
    config = load_config(write_config(tmp_path, MINIMAL), use_env=False)
    assert config.backtest.costs.commission_bps == 0.5
    assert config.cv.scheme == "walkforward"
    assert config.data.interval is Interval.D1


def test_experiment_overrides_defaults(tmp_path):
    payload = {**MINIMAL, "backtest": {"costs": {"commission_bps": 3.0}}}
    config = load_config(write_config(tmp_path, payload), use_env=False)
    assert config.backtest.costs.commission_bps == 3.0
    # Sibling defaults inside the same nested mapping survive the merge.
    assert config.backtest.costs.half_spread_bps == 1.0


def test_cli_overrides_win(tmp_path):
    config = load_config(
        write_config(tmp_path, MINIMAL),
        overrides=["backtest.costs.commission_bps=2.5", "meta.seed=7"],
        use_env=False,
    )
    assert config.backtest.costs.commission_bps == 2.5
    assert config.meta.seed == 7


def test_env_override_reaches_a_nested_field(tmp_path):
    config = load_config(
        write_config(tmp_path, MINIMAL),
        environ={"ALPHA__BACKTEST__COSTS__COMMISSION_BPS": "9.0"},
    )
    assert config.backtest.costs.commission_bps == 9.0


def test_cli_override_beats_env(tmp_path):
    config = load_config(
        write_config(tmp_path, MINIMAL),
        overrides=["backtest.costs.commission_bps=1.0"],
        environ={"ALPHA__BACKTEST__COSTS__COMMISSION_BPS": "9.0"},
    )
    assert config.backtest.costs.commission_bps == 1.0


# Every `ALPHA__*` variable that configures the process rather than the
# experiment. `AlphaConfig` is `extra="forbid"`, so one missing from the
# loader's exclusion set does not fall through harmlessly — it becomes a config
# key and fails the load. `ALPHA__JOBS_DB` did exactly that: setting the
# variable the API service reads made every `alpha run` in that shell fail.
PROCESS_ENV = {
    "ALPHA__RUNS_DIR": "/tmp/runs",
    "ALPHA__DATA_DIR": "/tmp/data",
    "ALPHA__JOBS_DB": "/tmp/runs/jobs.sqlite",
    "ALPHA__WEB_DIST": "/app/web/dist",
    "ALPHA__LOG_LEVEL": "DEBUG",
    "ALPHA__API_TOKEN": "unused",
    "ALPHA__ALPHAVANTAGE_API_KEY": "unused",
}


def test_process_env_vars_never_enter_the_config_tree():
    # Where the artifact store lives must not change config_hash.
    assert env_overrides(PROCESS_ENV) == {}


def test_process_env_vars_do_not_break_a_load(tmp_path):
    config = load_config(write_config(tmp_path, MINIMAL), environ=PROCESS_ENV)
    assert config.meta.name == "unit"


def test_the_service_reads_no_env_var_the_loader_would_treat_as_config():
    # The two halves are maintained in different packages, so the coupling is
    # asserted rather than remembered.
    from alphaservice.api import settings as service_settings  # noqa: PLC0415

    service_vars = {
        value
        for name, value in vars(service_settings).items()
        if name.endswith("_ENV_VAR") and isinstance(value, str)
    }
    assert service_vars, "the service should declare its environment variables as _ENV_VAR names"
    assert env_overrides(dict.fromkeys(service_vars, "x")) == {}


def test_env_overrides_ignore_unprefixed_variables():
    assert env_overrides({"PATH": "/usr/bin", "ALPHA__CV__EMBARGO_PCT": "0.05"}) == {
        "cv": {"embargo_pct": 0.05}
    }


def test_parse_override_builds_a_nested_fragment():
    assert parse_override("a.b.c=1.5") == {"a": {"b": {"c": 1.5}}}


@pytest.mark.parametrize("bad", ["nokey", "=1", " =2"])
def test_parse_override_rejects_malformed_input(bad):
    with pytest.raises(ConfigError):
        parse_override(bad)


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml", use_env=False)


def test_malformed_yaml_is_a_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("a: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Could not parse YAML"):
        load_config(path, use_env=False)


def test_non_mapping_yaml_is_a_config_error(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping at the top level"):
        load_config(path, use_env=False)


def test_empty_yaml_leaves_defaults_in_place(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="meta"):
        load_config(path, use_env=False)


# --- validation ------------------------------------------------------------


def test_unknown_key_is_rejected_and_named(tmp_path):
    payload = {**MINIMAL, "backtest": {"costs": {"comission_bps": 1.0}}}
    with pytest.raises(ConfigError, match="comission_bps"):
        load_config(write_config(tmp_path, payload), use_env=False)


def test_end_must_follow_start(tmp_path):
    payload = {
        **MINIMAL,
        "data": {"symbols": ["SPY"], "start": "2021-01-01", "end": "2020-01-01"},
    }
    with pytest.raises(ConfigError, match="must be after"):
        load_config(write_config(tmp_path, payload), use_env=False)


def test_csv_provider_requires_a_directory(tmp_path):
    payload = {
        **MINIMAL,
        "data": {
            "provider": "csv",
            "symbols": ["SPY"],
            "start": "2020-01-01",
            "end": "2021-01-01",
        },
    }
    with pytest.raises(ConfigError, match="csv_dir"):
        load_config(write_config(tmp_path, payload), use_env=False)


def test_short_threshold_must_sit_below_long(tmp_path):
    payload = {**MINIMAL, "signal": {"long_threshold": 0.4, "short_threshold": 0.6}}
    with pytest.raises(ConfigError, match="short_threshold"):
        load_config(write_config(tmp_path, payload), use_env=False)


def test_negative_lags_are_rejected_as_a_leak(tmp_path):
    payload = {**MINIMAL, "features": {"lags": [1, -1]}}
    with pytest.raises(ConfigError, match="leak"):
        load_config(write_config(tmp_path, payload), use_env=False)


def test_execution_lag_below_one_is_rejected(tmp_path):
    payload = {**MINIMAL, "backtest": {"execution_lag_bars": 0}}
    with pytest.raises(ConfigError, match="execution_lag_bars"):
        load_config(write_config(tmp_path, payload), use_env=False)


def test_cpcv_test_groups_must_leave_a_train_set(tmp_path):
    payload = {**MINIMAL, "cv": {"cpcv": {"n_groups": 4, "n_test_groups": 4}}}
    with pytest.raises(ConfigError, match="no training data remains"):
        load_config(write_config(tmp_path, payload), use_env=False)


def test_config_is_frozen(tmp_path):
    config = load_config(write_config(tmp_path, MINIMAL), use_env=False)
    with pytest.raises(ValueError, match="frozen"):
        config.meta.name = "changed"


# --- hashing and warnings --------------------------------------------------


def test_config_hash_is_stable_and_sensitive(tmp_path):
    a = load_config(write_config(tmp_path, MINIMAL, "a.yaml"), use_env=False)
    b = load_config(write_config(tmp_path, MINIMAL, "b.yaml"), use_env=False)
    assert config_hash(a) == config_hash(b)

    changed = load_config(
        write_config(tmp_path, MINIMAL, "c.yaml"),
        overrides=["backtest.costs.commission_bps=2.0"],
        use_env=False,
    )
    assert config_hash(changed) != config_hash(a)


def test_serialisable_round_trips(tmp_path):
    config = load_config(write_config(tmp_path, MINIMAL), use_env=False)
    assert AlphaConfig.model_validate(to_serialisable(config)) == config


def test_zero_costs_earn_the_uncosted_flag(tmp_path):
    payload = {
        **MINIMAL,
        "backtest": {
            "costs": {
                "commission_bps": 0.0,
                "min_commission": 0.0,
                "half_spread_bps": 0.0,
                "impact_model": "none",
            }
        },
    }
    config = load_config(write_config(tmp_path, payload), use_env=False)
    assert config.backtest.costs.is_frictionless
    assert "UNCOSTED" in config.warnings


def test_default_costs_are_not_flagged(tmp_path):
    config = load_config(write_config(tmp_path, MINIMAL), use_env=False)
    assert not config.backtest.costs.is_frictionless
    assert "UNCOSTED" not in config.warnings


def test_unused_short_threshold_is_flagged(tmp_path):
    payload = {**MINIMAL, "signal": {"allow_short": False}}
    config = load_config(write_config(tmp_path, payload), use_env=False)
    assert "SHORT_THRESHOLD_UNUSED" in config.warnings


# --- intervals -------------------------------------------------------------


@pytest.mark.parametrize(
    ("interval", "minutes", "freq"),
    [
        (Interval.M1, 1, "1min"),
        (Interval.M15, 15, "15min"),
        (Interval.H1, 60, "60min"),
        (Interval.H4, 240, "240min"),
        (Interval.D1, 1440, "1440min"),
    ],
)
def test_interval_properties(interval, minutes, freq):
    assert interval.minutes == minutes
    assert interval.pandas_freq == freq


def test_only_the_daily_interval_is_not_intraday():
    assert not Interval.D1.is_intraday
    assert all(i.is_intraday for i in Interval if i is not Interval.D1)


# --- the committed experiment configs --------------------------------------


@pytest.mark.parametrize("name", ["smoke.yaml", "spy_daily_tb.yaml", "btc_1h_meta.yaml"])
def test_committed_experiment_configs_are_valid(repo_root, name):
    config = load_config(repo_root / "configs" / "experiments" / name, use_env=False)
    assert config.meta.name
    assert config.data.symbols
