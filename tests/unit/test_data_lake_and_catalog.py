"""The parquet lake and the duckdb catalog (§6.3)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from alphacore.config.schema import Interval
from alphacore.data.catalog import CATALOG_FILENAME, Catalog, DatasetRecord
from alphacore.data.lake import (
    dataset_dir,
    dataset_id,
    parse_dataset_id,
    read_bars,
    write_bars,
)
from alphacore.errors import DataError
from alphacore.schemas.bars import empty_bars, normalise_bars, validate_bars


def make_bars(start: str = "2023-12-28", n: int = 10, symbol: str = "SYNTH") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="D", tz="UTC")
    close = np.linspace(100.0, 100.0 + n - 1, n)
    frame = pd.DataFrame(
        {
            "ts": ts,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000.0),
            "adj_factor": np.ones(n),
            "symbol": symbol,
        }
    )
    return validate_bars(normalise_bars(frame))


def record(**overrides) -> DatasetRecord:
    base = {
        "dataset_id": "csv:SYNTH:1d",
        "provider": "csv",
        "symbol": "SYNTH",
        "interval": "1d",
        "start_ts": dt.datetime(2015, 1, 2, tzinfo=dt.UTC),
        "end_ts": dt.datetime(2024, 12, 31, tzinfo=dt.UTC),
        "n_rows": 2516,
        "content_hash": "a3f91c7e2b4d5f60",
        "ingested_at": dt.datetime(2026, 7, 27, 14, 22, 1, tzinfo=dt.UTC),
        "provider_version": None,
        "calendar": "XNYS",
        "adjustment": "unadjusted_plus_factor",
        "quality_status": "PASS",
        "quality_summary": None,
    }
    return DatasetRecord.model_validate({**base, **overrides})


# --- ids -------------------------------------------------------------------


def test_dataset_id_shape():
    assert dataset_id("yahoo", "SPY", Interval.D1) == "yahoo:SPY:1d"


def test_parse_dataset_id_round_trips():
    assert parse_dataset_id("yahoo:SPY:1d") == ("yahoo", "SPY", "1d")


@pytest.mark.parametrize("bad", ["yahoo:SPY", "yahoo::1d", "spy"])
def test_parse_dataset_id_rejects_malformed(bad):
    with pytest.raises(DataError, match="Malformed dataset id"):
        parse_dataset_id(bad)


# --- lake ------------------------------------------------------------------


def test_write_partitions_by_year(tmp_path):
    paths, _ = write_bars(
        make_bars(), root=tmp_path, provider="csv", symbol="SYNTH", interval=Interval.D1
    )
    partitions = sorted(p.parent.name for p in paths)
    assert partitions == ["year=2023", "year=2024"]


def test_round_trip_preserves_values_and_dtypes(tmp_path):
    original = make_bars()
    write_bars(original, root=tmp_path, provider="csv", symbol="SYNTH", interval=Interval.D1)
    restored = read_bars(root=tmp_path, provider="csv", symbol="SYNTH", interval=Interval.D1)
    pd.testing.assert_frame_equal(original, restored)


def test_content_hash_is_stable_across_identical_writes(tmp_path):
    bars = make_bars()
    _, first = write_bars(
        bars, root=tmp_path / "a", provider="csv", symbol="SYNTH", interval=Interval.D1
    )
    _, second = write_bars(
        bars, root=tmp_path / "b", provider="csv", symbol="SYNTH", interval=Interval.D1
    )
    assert first == second


def test_content_hash_moves_when_a_price_moves(tmp_path):
    bars = make_bars()
    _, before = write_bars(
        bars, root=tmp_path / "a", provider="csv", symbol="SYNTH", interval=Interval.D1
    )
    changed = bars.copy()
    changed.iloc[0, changed.columns.get_loc("close")] = 123.45
    _, after = write_bars(
        changed, root=tmp_path / "b", provider="csv", symbol="SYNTH", interval=Interval.D1
    )
    assert before != after


def test_rewrite_replaces_stale_partitions(tmp_path):
    write_bars(
        make_bars("2019-01-01", 5),
        root=tmp_path,
        provider="csv",
        symbol="SYNTH",
        interval=Interval.D1,
    )
    write_bars(
        make_bars("2023-12-28", 10),
        root=tmp_path,
        provider="csv",
        symbol="SYNTH",
        interval=Interval.D1,
    )
    directory = dataset_dir(tmp_path, "csv", "SYNTH", Interval.D1)
    assert sorted(p.name for p in directory.iterdir()) == ["year=2023", "year=2024"]


def test_reading_span_bounds(tmp_path):
    write_bars(make_bars(), root=tmp_path, provider="csv", symbol="SYNTH", interval=Interval.D1)
    bars = read_bars(
        root=tmp_path,
        provider="csv",
        symbol="SYNTH",
        interval=Interval.D1,
        start="2024-01-01",
        end="2024-01-03",
    )
    assert len(bars) == 3


def test_writing_nothing_is_refused(tmp_path):
    with pytest.raises(DataError, match="empty dataset"):
        write_bars(
            empty_bars(), root=tmp_path, provider="csv", symbol="SYNTH", interval=Interval.D1
        )


def test_reading_an_absent_dataset_says_what_to_run(tmp_path):
    with pytest.raises(DataError, match="alpha data pull"):
        read_bars(root=tmp_path, provider="csv", symbol="NOPE", interval=Interval.D1)


# --- catalog ---------------------------------------------------------------


def test_catalog_creates_its_file(tmp_path):
    path = tmp_path / "nested" / CATALOG_FILENAME
    catalog = Catalog(path)
    assert path.is_file()
    assert catalog.list_datasets() == []
    assert "catalog" in repr(catalog)


def test_upsert_and_get_round_trip(tmp_path):
    catalog = Catalog(tmp_path / CATALOG_FILENAME)
    original = record()
    catalog.upsert(original)
    assert catalog.get("csv:SYNTH:1d") == original


def test_upsert_replaces_rather_than_appends(tmp_path):
    catalog = Catalog(tmp_path / CATALOG_FILENAME)
    catalog.upsert(record())
    catalog.upsert(record(content_hash="deadbeef", n_rows=99))
    rows = catalog.list_datasets()
    assert len(rows) == 1
    assert rows[0].content_hash == "deadbeef"


def test_get_returns_none_for_an_unknown_dataset(tmp_path):
    assert Catalog(tmp_path / CATALOG_FILENAME).get("nope:NOPE:1d") is None


def test_require_lists_what_is_known(tmp_path):
    catalog = Catalog(tmp_path / CATALOG_FILENAME)
    catalog.upsert(record())
    with pytest.raises(DataError, match="csv:SYNTH:1d"):
        catalog.require("yahoo:SPY:1d")


def test_list_datasets_is_ordered(tmp_path):
    catalog = Catalog(tmp_path / CATALOG_FILENAME)
    catalog.upsert(record(dataset_id="yahoo:SPY:1d", symbol="SPY", provider="yahoo"))
    catalog.upsert(record())
    assert [r.dataset_id for r in catalog.list_datasets()] == ["csv:SYNTH:1d", "yahoo:SPY:1d"]


def test_delete_reports_whether_a_row_existed(tmp_path):
    catalog = Catalog(tmp_path / CATALOG_FILENAME)
    catalog.upsert(record())
    assert catalog.delete("csv:SYNTH:1d") is True
    assert catalog.delete("csv:SYNTH:1d") is False


def test_catalog_survives_reopening(tmp_path):
    path = tmp_path / CATALOG_FILENAME
    Catalog(path).upsert(record())
    assert Catalog(path).get("csv:SYNTH:1d") is not None


def test_catalog_timestamps_read_back_in_utc(tmp_path):
    # duckdb renders TIMESTAMPTZ in the session timezone. Without the UTC pin
    # the same row reads back with a local offset, and that string reaches the
    # API and the UI.
    catalog = Catalog(tmp_path / CATALOG_FILENAME)
    catalog.upsert(record())
    stored = catalog.require("csv:SYNTH:1d")

    assert stored.start_ts.utcoffset() == dt.timedelta(0)
    assert stored.ingested_at.utcoffset() == dt.timedelta(0)
    assert stored.start_ts.isoformat() == "2015-01-02T00:00:00+00:00"


def test_catalog_timestamps_are_utc_regardless_of_local_timezone(tmp_path, monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    catalog = Catalog(tmp_path / CATALOG_FILENAME)
    catalog.upsert(record())
    assert catalog.require("csv:SYNTH:1d").start_ts.utcoffset() == dt.timedelta(0)
