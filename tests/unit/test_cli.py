"""The `alpha` CLI (Phase 0 surface)."""

from __future__ import annotations

from typer.testing import CliRunner

from alphacore.runtime import ArtifactStore, RunStatus
from alphaservice.cli import app

runner = CliRunner()


def test_help_lists_the_phase_0_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.stdout
    assert "doctor" in result.stdout


def test_version_reports_both_packages():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "alphacore" in result.stdout
    assert "alphaservice" in result.stdout


def test_doctor_passes_in_a_pinned_deterministic_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("ALPHA__RUNS_DIR", str(tmp_path / "runs"))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "PASS" in result.stdout


def test_doctor_fails_when_hash_randomisation_is_on(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setenv("ALPHA__RUNS_DIR", str(tmp_path / "runs"))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "FAIL" in result.stdout


def test_scratch_run_writes_a_valid_completed_run(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    monkeypatch.setenv("ALPHA__RUNS_DIR", str(runs))
    result = runner.invoke(app, ["dev", "scratch-run", "--name", "phase-0"])
    assert result.exit_code == 0, result.output

    store = ArtifactStore(runs)
    (run_id,) = store.list_runs()
    manifest = store.open_run(run_id).read_manifest()
    assert manifest.status is RunStatus.COMPLETED
    assert manifest.name == "phase-0"
    assert manifest.config_hash
    assert "scratch" in manifest.phase_durations_s
    assert store.open_run(run_id).exists("scratch.json")
    assert store.open_run(run_id).exists("config.resolved.yaml")
