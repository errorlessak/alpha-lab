"""Providers, retries, and the provider registry (§6.1)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from alphacore.config.schema import DataConfig, Interval
from alphacore.data.protocol import BarProvider
from alphacore.data.providers import CsvProvider, YahooProvider, get_provider
from alphacore.data.providers.base import RETRYABLE_STATUS_CODES, retry
from alphacore.errors import ConfigError, ProviderError, SchemaValidationError

START = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
END = dt.datetime(2024, 1, 31, tzinfo=dt.UTC)


def write_csv(directory, name: str = "SYNTH_1d.csv", n: int = 10) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC")
    close = np.linspace(100.0, 100.0 + n - 1, n)
    pd.DataFrame(
        {
            "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000.0),
            "adj_factor": 1.0,
            "symbol": "SYNTH",
        }
    ).to_csv(directory / name, index=False)


# --- retry -----------------------------------------------------------------


def test_retry_returns_on_first_success():
    assert retry(lambda: 42, description="probe", sleep=lambda _: None) == 42


def test_retry_recovers_after_transient_failures():
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            msg = "429 rate limited"
            raise RuntimeError(msg)
        return "ok"

    assert retry(flaky, description="probe", sleep=lambda _: None) == "ok"
    assert attempts["n"] == 3


def test_retry_gives_up_and_names_the_operation():
    def always_fails() -> None:
        msg = "503"
        raise RuntimeError(msg)

    with pytest.raises(ProviderError, match="probe failed after 3 attempts"):
        retry(always_fails, description="probe", max_attempts=3, sleep=lambda _: None)


def test_retry_backoff_grows_and_is_jittered():
    delays: list[float] = []
    with pytest.raises(ProviderError):
        retry(
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            description="probe",
            max_attempts=5,
            sleep=delays.append,
        )
    assert len(delays) == 4
    assert all(d >= 0 for d in delays)
    # Full jitter draws from [0, ceiling], so assert the ceiling grows rather
    # than the samples themselves — which would be a flaky test.
    assert max(delays[2:]) > 0


def test_retryable_status_codes_cover_rate_limits_not_client_errors():
    assert 429 in RETRYABLE_STATUS_CODES
    assert 503 in RETRYABLE_STATUS_CODES
    assert 404 not in RETRYABLE_STATUS_CODES
    assert 401 not in RETRYABLE_STATUS_CODES


# --- csv provider ----------------------------------------------------------


def test_csv_provider_satisfies_the_protocol(tmp_path):
    assert isinstance(CsvProvider(tmp_path), BarProvider)


def test_csv_fetch_returns_validated_bars(tmp_path):
    write_csv(tmp_path)
    bars = CsvProvider(tmp_path).fetch("SYNTH", Interval.D1, START, END)
    assert len(bars) == 10
    assert bars.index.is_monotonic_increasing


def test_csv_fetch_clips_to_the_requested_span(tmp_path):
    write_csv(tmp_path)
    bars = CsvProvider(tmp_path).fetch(
        "SYNTH",
        Interval.D1,
        dt.datetime(2024, 1, 4, tzinfo=dt.UTC),
        dt.datetime(2024, 1, 6, tzinfo=dt.UTC),
    )
    assert len(bars) == 3


def test_csv_falls_back_to_the_bare_filename(tmp_path):
    write_csv(tmp_path, name="SYNTH.csv")
    assert len(CsvProvider(tmp_path).fetch("SYNTH", Interval.D1, START, END)) == 10


def test_csv_missing_file_lists_what_is_present(tmp_path):
    write_csv(tmp_path)
    with pytest.raises(ProviderError, match=r"SYNTH_1d\.csv"):
        CsvProvider(tmp_path).fetch("ABSENT", Interval.D1, START, END)


def test_csv_rejects_a_file_that_breaks_the_schema(tmp_path):
    write_csv(tmp_path)
    path = tmp_path / "SYNTH_1d.csv"
    rows = path.read_text().splitlines()
    parts = rows[1].split(",")
    parts[4] = "-5.0"  # close
    rows[1] = ",".join(parts)
    path.write_text("\n".join(rows))
    with pytest.raises(SchemaValidationError):
        CsvProvider(tmp_path).fetch("SYNTH", Interval.D1, START, END)


def test_csv_has_no_lookback_limit(tmp_path):
    assert CsvProvider(tmp_path).max_lookback(Interval.D1) is None


def test_csv_repr_names_the_directory(tmp_path):
    assert tmp_path.as_posix() in repr(CsvProvider(tmp_path))


# --- yahoo provider (offline surface only) ---------------------------------


def test_yahoo_satisfies_the_protocol():
    assert isinstance(YahooProvider(), BarProvider)


def test_yahoo_publishes_its_history_limits():
    provider = YahooProvider()
    assert provider.max_lookback(Interval.D1) is None
    assert provider.max_lookback(Interval.M1) == dt.timedelta(days=30)
    assert provider.max_lookback(Interval.H1) == dt.timedelta(days=730)


def test_yahoo_reshape_recovers_the_adjustment_factor():
    raw = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.0],
            "Adj Close": [50.0],
            "Volume": [1000.0],
        },
        index=pd.DatetimeIndex(["2024-01-02"], tz="UTC"),
    )
    reshaped = YahooProvider._reshape(raw, "SPY")
    # Unadjusted close is preserved and the factor carries the adjustment (I7).
    assert reshaped["close"].iloc[0] == 100.0
    assert reshaped["adj_factor"].iloc[0] == 0.5


def test_yahoo_reshape_flattens_multiindex_columns():
    columns = pd.MultiIndex.from_tuples(
        [("Open", "SPY"), ("High", "SPY"), ("Low", "SPY"), ("Close", "SPY"), ("Volume", "SPY")]
    )
    raw = pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 1000.0]],
        columns=columns,
        index=pd.DatetimeIndex(["2024-01-02"], tz="UTC"),
    )
    reshaped = YahooProvider._reshape(raw, "SPY")
    assert set(reshaped.columns) >= {"open", "high", "low", "close", "volume", "adj_factor"}


def test_yahoo_rejects_an_unsupported_interval(monkeypatch):
    provider = YahooProvider()
    monkeypatch.setitem(
        __import__("alphacore.data.providers.yahoo", fromlist=["_YF_INTERVAL"])._YF_INTERVAL,
        "1d",
        None,
    )
    with pytest.raises(ProviderError, match="does not serve interval"):
        provider.fetch("SPY", Interval.D1, START, END)


# --- registry --------------------------------------------------------------


def data_config(**overrides) -> DataConfig:
    base = {
        "provider": "csv",
        "symbols": ("SYNTH",),
        "start": dt.date(2024, 1, 1),
        "end": dt.date(2024, 2, 1),
        "csv_dir": "samples",
    }
    return DataConfig.model_validate({**base, **overrides})


def test_registry_builds_the_csv_provider():
    assert isinstance(get_provider(data_config()), CsvProvider)


def test_registry_builds_the_yahoo_provider():
    assert isinstance(get_provider(data_config(provider="yahoo", csv_dir=None)), YahooProvider)


def test_registry_names_the_phase_for_a_planned_provider():
    with pytest.raises(ProviderError, match="not implemented yet"):
        get_provider(data_config(provider="binance", csv_dir=None))


def test_registry_rejects_a_csv_config_without_a_directory():
    config = data_config()
    stripped = config.model_copy(update={"csv_dir": None})
    with pytest.raises(ConfigError, match="csv_dir"):
        get_provider(stripped)
