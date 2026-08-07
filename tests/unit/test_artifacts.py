"""The immutable artifact store (§4, I9)."""

from __future__ import annotations

import json

import pytest

from alphacore.errors import ArtifactExistsError, RunCancelledError, RunNotFoundError
from alphacore.runtime.artifacts import ArtifactStore, run_context
from alphacore.runtime.logging import LOG_FILENAME
from alphacore.runtime.manifest import MANIFEST_FILENAME, RunStatus

CONFIG = {"meta": {"name": "smoke", "seed": 1}}


def test_create_run_makes_a_writable_directory(store):
    run = store.create_run("run-a")
    assert run.path.is_dir()
    assert run.run_id == "run-a"
    assert "rw" in repr(run)


def test_create_run_refuses_to_reuse_a_run_id(store):
    store.create_run("run-a")
    with pytest.raises(ArtifactExistsError, match="never reused"):
        store.create_run("run-a")


@pytest.mark.parametrize("run_id", ["..", "../elsewhere", "nested/run", "", "."])
def test_open_run_refuses_an_id_that_is_not_a_child_of_the_store(store, run_id):
    # `artifact_path` already refuses to escape a *run* directory, which is
    # worth nothing if the run directory itself can be the store's parent. The
    # API reaches this through a percent-encoded path segment: `GET
    # /api/v1/runs/%2E%2E/export/README.md` served the repository's own README
    # until this check existed.
    with pytest.raises(RunNotFoundError):
        store.open_run(run_id)


@pytest.mark.parametrize("run_id", ["..", "../elsewhere", "nested/run", "", "."])
def test_create_run_refuses_an_id_that_is_not_a_child_of_the_store(store, run_id):
    with pytest.raises(ArtifactExistsError, match="single directory name"):
        store.create_run(run_id)


def test_write_and_read_json_round_trip(store):
    run = store.create_run("run-a")
    run.write_json("tearsheet.json", {"sharpe": 0.71, "model": "lightgbm"})
    assert run.read_json("tearsheet.json") == {"sharpe": 0.71, "model": "lightgbm"}


def test_write_json_is_canonical_on_disk(store):
    run = store.create_run("run-a")
    path = run.write_json("tearsheet.json", {"b": 1, "a": 2})
    assert path.read_bytes() == b'{"a":2,"b":1}\n'


def test_write_text_normalises_newlines(store):
    run = store.create_run("run-a")
    path = run.write_text("research.md", "line one\nline two\n")
    assert b"\r\n" not in path.read_bytes()


def test_results_are_write_once(store):
    run = store.create_run("run-a")
    run.write_json("tearsheet.json", {"sharpe": 0.71})
    with pytest.raises(ArtifactExistsError, match="immutable"):
        run.write_json("tearsheet.json", {"sharpe": 2.9})
    assert run.read_json("tearsheet.json") == {"sharpe": 0.71}


@pytest.mark.parametrize("name", [MANIFEST_FILENAME, LOG_FILENAME])
def test_control_files_are_the_only_rewritable_artifacts(store, name):
    run = store.create_run("run-a")
    run.write_text(name, "first")
    run.write_text(name, "second")
    assert run.read_bytes(name) == b"second"


def test_nested_artifact_paths_are_created(store):
    run = store.create_run("run-a")
    run.write_text("models/lightgbm/model.txt", "tree {}")
    assert run.exists("models/lightgbm/model.txt")


def test_artifact_names_cannot_escape_the_run_directory(store):
    run = store.create_run("run-a")
    with pytest.raises(ArtifactExistsError, match="escapes"):
        run.write_text("../../escaped.txt", "nope")


def test_reading_a_missing_artifact_names_the_run(store):
    run = store.create_run("run-a")
    with pytest.raises(RunNotFoundError, match="run-a"):
        run.read_json("tearsheet.json")


def test_open_run_is_read_only(store):
    store.create_run("run-a").write_json("tearsheet.json", {"sharpe": 0.1})
    reader = store.open_run("run-a")
    assert "ro" in repr(reader)
    assert reader.read_json("tearsheet.json") == {"sharpe": 0.1}
    with pytest.raises(ArtifactExistsError, match="read-only"):
        reader.write_json("tearsheet.json", {"sharpe": 9.9})


def test_open_run_raises_for_an_unknown_run(store):
    with pytest.raises(RunNotFoundError, match="no-such-run"):
        store.open_run("no-such-run")


def test_list_runs_is_newest_first(store):
    for run_id in ("2026-01-01T00-00-00Z_a_aaaaaa", "2026-06-01T00-00-00Z_b_bbbbbb"):
        store.create_run(run_id)
    assert store.list_runs() == [
        "2026-06-01T00-00-00Z_b_bbbbbb",
        "2026-01-01T00-00-00Z_a_aaaaaa",
    ]


def test_list_runs_on_an_absent_store_is_empty(tmp_path):
    assert ArtifactStore(tmp_path / "never-created").list_runs() == []


def test_store_root_falls_back_to_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHA__RUNS_DIR", str(tmp_path / "elsewhere"))
    assert ArtifactStore().root == (tmp_path / "elsewhere").resolve()
    assert "elsewhere" in repr(ArtifactStore())


def test_run_context_finalises_a_successful_run(store, frozen_now):
    with (
        run_context(name="smoke", seed=1, config=CONFIG, store=store, now=frozen_now) as (
            run,
            timings,
        ),
        timings.phase("train"),
    ):
        run.write_json("tearsheet.json", {"sharpe": 0.1})

    manifest = store.open_run(run.run_id).read_manifest()
    assert manifest.status is RunStatus.COMPLETED
    assert manifest.duration_s is not None
    assert "train" in manifest.phase_durations_s
    assert manifest.error is None


def test_run_context_writes_a_log_artifact(store, frozen_now):
    with run_context(name="smoke", seed=1, config=CONFIG, store=store, now=frozen_now) as (run, _):
        pass
    lines = run.read_bytes(LOG_FILENAME).decode("utf-8").splitlines()
    assert lines
    assert json.loads(lines[0])["record"]["message"].startswith("run started")


def test_run_context_marks_a_crashed_run_failed(store, frozen_now):
    captured = {}
    with (
        pytest.raises(ValueError, match="engine exploded"),
        run_context(name="smoke", seed=1, config=CONFIG, store=store, now=frozen_now) as (run, _),
    ):
        captured["run_id"] = run.run_id
        msg = "engine exploded"
        raise ValueError(msg)

    manifest = store.open_run(captured["run_id"]).read_manifest()
    assert manifest.status is RunStatus.FAILED
    assert "engine exploded" in (manifest.error or "")


def test_run_context_marks_an_interrupted_run_cancelled(store, frozen_now):
    captured = {}
    with (
        pytest.raises(KeyboardInterrupt),
        run_context(name="smoke", seed=1, config=CONFIG, store=store, now=frozen_now) as (run, _),
    ):
        captured["run_id"] = run.run_id
        raise KeyboardInterrupt

    manifest = store.open_run(captured["run_id"]).read_manifest()
    assert manifest.status is RunStatus.CANCELLED


def test_run_context_marks_a_cancelled_run_cancelled(store, frozen_now):
    # A job cancelled through the API arrives as RunCancelledError, not as a
    # keyboard interrupt. §14.2 requires the run directory to end up marked
    # rather than left half-written and presented as valid.
    captured = {}
    with (
        pytest.raises(RunCancelledError, match="asked to stop"),
        run_context(name="smoke", seed=1, config=CONFIG, store=store, now=frozen_now) as (run, _),
    ):
        captured["run_id"] = run.run_id
        msg = "the job was asked to stop"
        raise RunCancelledError(msg)

    manifest = store.open_run(captured["run_id"]).read_manifest()
    assert manifest.status is RunStatus.CANCELLED
    assert "asked to stop" in (manifest.error or "")


def test_run_context_generates_the_documented_run_id(store, frozen_now):
    with run_context(name="spy daily tb", seed=1, config=CONFIG, store=store, now=frozen_now) as (
        run,
        _,
    ):
        pass
    assert run.run_id.startswith("2026-07-27T14-22-01Z_spy-daily-tb_")
