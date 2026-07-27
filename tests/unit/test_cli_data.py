"""The `alpha data` and `alpha config` commands (Phase 1 surface)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from alphaservice.cli import app

runner = CliRunner()


@pytest.fixture
def smoke_config(repo_root):
    return repo_root / "configs" / "experiments" / "smoke.yaml"


def test_help_lists_the_data_and_config_groups():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "data" in result.stdout
    assert "config" in result.stdout


def test_config_show_prints_yaml_and_a_hash(smoke_config):
    result = runner.invoke(app, ["config", "show", "-c", str(smoke_config)])
    assert result.exit_code == 0, result.output
    assert "config_hash" in result.stdout
    assert "triple_barrier" in result.stdout


def test_config_show_applies_an_override(smoke_config):
    result = runner.invoke(
        app,
        ["config", "show", "-c", str(smoke_config), "--set", "backtest.costs.commission_bps=0"],
    )
    assert result.exit_code == 0, result.output


def test_config_show_reports_an_invalid_override(smoke_config):
    result = runner.invoke(
        app, ["config", "show", "-c", str(smoke_config), "--set", "data.interval=3d"]
    )
    assert result.exit_code == 1
    assert "config error" in result.stdout


def test_data_list_on_an_empty_catalog_offers_the_next_command(tmp_path):
    result = runner.invoke(app, ["data", "list", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "alpha data pull" in result.stdout


def test_data_pull_then_list_and_inspect(smoke_config, tmp_path, repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)

    pull = runner.invoke(
        app, ["data", "pull", "-c", str(smoke_config), "--data-dir", str(tmp_path)]
    )
    assert pull.exit_code == 0, pull.output
    assert "csv:SYNTH:1d" in pull.stdout
    assert "PASS" in pull.stdout

    listing = runner.invoke(app, ["data", "list", "--data-dir", str(tmp_path)])
    assert listing.exit_code == 0
    assert "csv:SYNTH:1d" in listing.stdout

    inspect = runner.invoke(
        app, ["data", "inspect", "--dataset", "csv:SYNTH:1d", "--data-dir", str(tmp_path)]
    )
    assert inspect.exit_code == 0
    assert "content_hash" in inspect.stdout
    assert "XNYS" in inspect.stdout


def test_data_inspect_on_an_unknown_dataset_fails_cleanly(tmp_path):
    result = runner.invoke(
        app, ["data", "inspect", "--dataset", "yahoo:SPY:1d", "--data-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "not in the catalog" in result.stdout


def test_data_pull_reports_a_quality_failure_without_a_traceback(
    smoke_config, tmp_path, repo_root, monkeypatch
):
    monkeypatch.chdir(repo_root)
    result = runner.invoke(
        app,
        [
            "data",
            "pull",
            "-c",
            str(smoke_config),
            "--data-dir",
            str(tmp_path),
            "--set",
            "data.quality.max_missing_bar_pct=0.0",
            "--set",
            "data.end=2025-12-31",
        ],
    )
    assert result.exit_code == 1
    assert "ingest failed" in result.stdout
    assert "allow-dirty-data" in result.stdout
    assert "Traceback" not in result.stdout


def test_data_pull_with_allow_dirty_marks_the_dataset_untrusted(
    smoke_config, tmp_path, repo_root, monkeypatch
):
    monkeypatch.chdir(repo_root)
    result = runner.invoke(
        app,
        [
            "data",
            "pull",
            "-c",
            str(smoke_config),
            "--data-dir",
            str(tmp_path),
            "--allow-dirty-data",
            "--set",
            "data.quality.max_missing_bar_pct=0.0",
            "--set",
            "data.end=2025-12-31",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "FAIL" in result.stdout
    assert "NO" in result.stdout
