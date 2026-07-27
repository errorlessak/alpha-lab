"""End-to-end ingestion against the committed fixture (§6, Phase 1 exit).

Offline by construction: everything here runs through the `csv` provider and
the committed `samples/` fixture, so the suite passes with no network.
"""

from __future__ import annotations

import json

import pytest

from alphacore.config import load_config
from alphacore.config.schema import Interval
from alphacore.data import CATALOG_FILENAME, Catalog, QualityStatus, read_bars
from alphacore.errors import DataQualityError, ProviderError
from alphacore.pipelines import ingest_dataset, ingest_symbols, resolve_calendar


@pytest.fixture
def smoke_config(repo_root):
    return load_config(repo_root / "configs" / "experiments" / "smoke.yaml", use_env=False)


@pytest.fixture
def rooted_config(smoke_config, repo_root):
    # The committed config points at a relative samples/ dir; make it absolute
    # so the test does not depend on the working directory.
    data = smoke_config.data.model_copy(update={"csv_dir": str(repo_root / "samples")})
    return smoke_config.model_copy(update={"data": data})


def test_fixture_ingests_cleanly(rooted_config, tmp_path):
    results = ingest_symbols(rooted_config, data_dir=tmp_path)

    assert len(results) == 1
    result = results[0]
    assert result.dataset_id == "csv:SYNTH:1d"
    assert result.n_rows > 2_000
    assert result.quality.status is QualityStatus.PASS
    assert result.trusted
    assert result.content_hash


def test_quality_report_is_written_beside_the_dataset(rooted_config, tmp_path):
    result = ingest_symbols(rooted_config, data_dir=tmp_path)[0]
    assert result.quality_path.is_file()

    report = json.loads(result.quality_path.read_text("utf-8"))
    assert report["status"] == "PASS"
    assert report["dataset_id"] == "csv:SYNTH:1d"
    assert report["n_rows"] == result.n_rows


def test_quality_report_does_not_perturb_the_content_hash(rooted_config, tmp_path):
    # The hash describes the bars, not the verdict on them.
    first = ingest_symbols(rooted_config, data_dir=tmp_path / "a")[0]
    second = ingest_symbols(rooted_config, data_dir=tmp_path / "b")[0]
    assert first.content_hash == second.content_hash
    assert first.quality_path != second.quality_path


def test_ingested_bars_are_readable_from_the_lake(rooted_config, tmp_path):
    result = ingest_symbols(rooted_config, data_dir=tmp_path)[0]
    bars = read_bars(root=tmp_path, provider="csv", symbol="SYNTH", interval=Interval.D1)
    assert len(bars) == result.n_rows
    assert bars.index.is_monotonic_increasing


def test_catalog_row_records_the_content_hash(rooted_config, tmp_path):
    result = ingest_symbols(rooted_config, data_dir=tmp_path)[0]
    record = Catalog(tmp_path / CATALOG_FILENAME).require("csv:SYNTH:1d")

    assert record.content_hash == result.content_hash
    assert record.n_rows == result.n_rows
    assert record.quality_status == "PASS"
    assert record.calendar == "XNYS"
    assert record.adjustment == "unadjusted_plus_factor"


def test_ingestion_is_reproducible(rooted_config, tmp_path):
    first = ingest_symbols(rooted_config, data_dir=tmp_path / "a")[0]
    second = ingest_symbols(rooted_config, data_dir=tmp_path / "b")[0]
    assert first.content_hash == second.content_hash


def test_reingestion_replaces_rather_than_duplicates(rooted_config, tmp_path):
    ingest_symbols(rooted_config, data_dir=tmp_path)
    ingest_symbols(rooted_config, data_dir=tmp_path)
    assert len(Catalog(tmp_path / CATALOG_FILENAME).list_datasets()) == 1


def test_benchmark_is_ingested_when_it_is_not_already_a_symbol(rooted_config, tmp_path):
    data = rooted_config.data.model_copy(update={"benchmark": "OTHER"})
    config = rooted_config.model_copy(update={"data": data})
    # OTHER has no fixture, so the attempt must fail loudly rather than skip it:
    # an information ratio against a benchmark nobody ingested is not a number.
    with pytest.raises(ProviderError, match="OTHER"):
        ingest_symbols(config, data_dir=tmp_path)


def test_resolve_calendar_uses_the_configured_value(rooted_config):
    assert resolve_calendar(rooted_config.data) == "XNYS"


def test_resolve_calendar_falls_back_to_the_provider_default(rooted_config):
    data = rooted_config.data.model_copy(update={"calendar": None})
    assert resolve_calendar(data) == "XNYS"


def test_a_quality_failure_aborts_and_explains(rooted_config, tmp_path):
    quality = rooted_config.data.quality.model_copy(update={"max_missing_bar_pct": 0.0})
    # 2015-2024 on the NYSE grid, but the fixture is a subset once we shorten
    # the window, so bars go missing relative to the expected grid.
    data = rooted_config.data.model_copy(
        update={"quality": quality, "end": rooted_config.data.end.replace(year=2025)}
    )
    config = rooted_config.model_copy(update={"data": data})

    with pytest.raises(DataQualityError, match="allow-dirty-data"):
        ingest_symbols(config, data_dir=tmp_path)


def test_allow_dirty_proceeds_but_marks_the_result_untrusted(rooted_config, tmp_path):
    quality = rooted_config.data.quality.model_copy(update={"max_missing_bar_pct": 0.0})
    data = rooted_config.data.model_copy(
        update={"quality": quality, "end": rooted_config.data.end.replace(year=2025)}
    )
    config = rooted_config.model_copy(update={"data": data})

    result = ingest_dataset("SYNTH", config.data, data_dir=tmp_path, allow_dirty=True)
    assert result.quality.status is QualityStatus.FAIL
    assert result.trusted is False
    # The failure is recorded in the catalog, not just in the exception path.
    record = Catalog(tmp_path / CATALOG_FILENAME).require("csv:SYNTH:1d")
    assert record.quality_status == "FAIL"
