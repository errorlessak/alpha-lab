"""The research verbs added in Phase 10 (§15).

`alpha features build`, `alpha train`, `alpha backtest`, `alpha run`,
`alpha report`, and `alpha runs list|show|compare`.

The run these share is produced once per module: a golden-path run takes a few
seconds, and producing one per test would make the suite slow enough that
people stop running it.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from alphaservice.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def smoke_config_path(repo_root):
    return str(repo_root / "configs" / "experiments" / "smoke.yaml")


@pytest.fixture(scope="module")
def lake(tmp_path_factory, smoke_config_path):
    """A lake with the smoke fixture already ingested."""
    root = tmp_path_factory.mktemp("cli-lake")
    result = runner.invoke(app, ["data", "pull", "-c", smoke_config_path, "--data-dir", str(root)])
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory, smoke_config_path, lake):
    """One completed run, and the store it lives in."""
    runs = tmp_path_factory.mktemp("cli-runs")
    result = runner.invoke(
        app,
        [
            "run",
            "-c",
            smoke_config_path,
            "--data-dir",
            str(lake),
            "--runs-dir",
            str(runs),
            "--skip-ingest",
        ],
    )
    assert result.exit_code == 0, result.output
    return runs, result.output


def test_help_lists_every_phase_10_verb():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for verb in ("features", "train", "backtest", "run", "report", "runs"):
        assert verb in result.stdout


def test_features_build_reports_the_frame_it_built(smoke_config_path, lake):
    result = runner.invoke(
        app, ["features", "build", "-c", smoke_config_path, "--data-dir", str(lake)]
    )
    assert result.exit_code == 0, result.output
    assert "annualisation" in result.stdout
    assert "warm-up dropped" in result.stdout


def test_features_build_says_what_to_do_when_the_lake_is_empty(smoke_config_path, tmp_path):
    result = runner.invoke(
        app, ["features", "build", "-c", smoke_config_path, "--data-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "alpha data pull" in result.stdout


def test_train_prints_an_economic_leaderboard(smoke_config_path, lake):
    result = runner.invoke(app, ["train", "-c", smoke_config_path, "--data-dir", str(lake)])
    assert result.exit_code == 0, result.output
    assert "net Sharpe" in result.stdout
    assert "baseline_s" in result.stdout, "the shuffled-label baseline must always be shown"
    assert "purged" in result.stdout


def test_backtest_prints_the_cost_sensitivity_curve(smoke_config_path, lake):
    result = runner.invoke(app, ["backtest", "-c", smoke_config_path, "--data-dir", str(lake)])
    assert result.exit_code == 0, result.output
    assert "cost sensitivity" in result.stdout
    assert "break-even round trip" in result.stdout
    assert "assumed round trip" in result.stdout


def test_run_reports_the_headline_and_where_it_wrote_it(completed_run):
    _, output = completed_run
    assert "run directory" in output
    assert "deflated Sharpe" in output
    assert "probability of overfitting" in output


def test_runs_list_shows_the_completed_run(completed_run):
    runs, _ = completed_run
    result = runner.invoke(app, ["runs", "list", "--runs-dir", str(runs)])
    assert result.exit_code == 0, result.output
    assert "completed" in result.stdout
    assert "config hash" in result.stdout


def test_runs_list_says_how_to_produce_one_when_empty(tmp_path):
    result = runner.invoke(app, ["runs", "list", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "alpha run" in result.stdout


def test_runs_show_accepts_the_short_suffix(completed_run):
    runs, _ = completed_run
    run_id = sorted(p.name for p in runs.iterdir() if p.is_dir())[0]
    result = runner.invoke(app, ["runs", "show", run_id[-6:], "--runs-dir", str(runs)])
    assert result.exit_code == 0, result.output
    assert "selected model" in result.stdout
    assert "data hash" in result.stdout


def test_runs_show_reports_an_unknown_run(completed_run):
    runs, _ = completed_run
    result = runner.invoke(app, ["runs", "show", "nosuchrun", "--runs-dir", str(runs)])
    assert result.exit_code == 1
    assert "No run matching" in result.stdout


def test_runs_compare_needs_two_runs(completed_run):
    runs, _ = completed_run
    run_id = sorted(p.name for p in runs.iterdir() if p.is_dir())[0]
    result = runner.invoke(app, ["runs", "compare", run_id, "--runs-dir", str(runs)])
    assert result.exit_code == 1
    assert "at least two" in result.stdout


def test_runs_compare_flags_a_shared_config_hash(completed_run, smoke_config_path, lake):
    runs, _ = completed_run
    second = runner.invoke(
        app,
        [
            "run",
            "-c",
            smoke_config_path,
            "--data-dir",
            str(lake),
            "--runs-dir",
            str(runs),
            "--skip-ingest",
        ],
    )
    assert second.exit_code == 0, second.output

    ids = sorted(p.name for p in runs.iterdir() if p.is_dir())
    result = runner.invoke(app, ["runs", "compare", *ids, "--runs-dir", str(runs)])
    assert result.exit_code == 0, result.output
    # Same config, same data, same code: the comparison should say so rather
    # than leave a reader hunting for a difference that cannot exist.
    assert "share a config hash" in result.stdout


def test_report_renders_the_research_note(completed_run):
    runs, _ = completed_run
    run_id = sorted(p.name for p in runs.iterdir() if p.is_dir())[0]
    result = runner.invoke(app, ["report", run_id, "--runs-dir", str(runs)])
    assert result.exit_code == 0, result.output
    assert "Research note" in result.stdout
    assert "Not investment advice" in result.stdout


def test_report_writes_to_a_file_when_asked(completed_run, tmp_path):
    runs, _ = completed_run
    run_id = sorted(p.name for p in runs.iterdir() if p.is_dir())[0]
    destination = tmp_path / "note.md"
    result = runner.invoke(
        app, ["report", run_id, "--runs-dir", str(runs), "--output", str(destination)]
    )
    assert result.exit_code == 0, result.output
    assert "# Research note" in destination.read_text(encoding="utf-8")


def test_report_reports_a_missing_run(tmp_path):
    result = runner.invoke(app, ["report", "ghost", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "No run matching" in result.stdout
