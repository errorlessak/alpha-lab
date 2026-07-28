"""The in-process job runner (§14.2, ADR 0008).

A process pool, SQLite for state, cooperative cancellation, and a run directory
that is never left half-written and presented as valid.

Most of these use a thread executor so the logic is tested in milliseconds. One
test uses the real `ProcessPoolExecutor` on the real smoke config, because a
worker that only ever ran in-thread would not prove that the state written by a
different process is visible to the one answering `GET /jobs/{id}`.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from alphaservice.errors import JobNotFoundError
from alphaservice.jobs.runner import JobRunner
from alphaservice.jobs.store import JobStatus, JobStore, JobType


def _await_terminal(runner: JobRunner, job_id: str, *, timeout: float = 180.0) -> JobStatus:
    """Block until a job reaches a terminal state.

    Args:
        runner: The runner.
        job_id: The job id.
        timeout: Seconds to wait before giving up.

    Returns:
        The terminal status.

    Raises:
        AssertionError: If the job is still running when the timeout expires.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = runner.get(job_id).status
        if status.is_terminal:
            return status
        time.sleep(0.05)
    msg = f"job {job_id} did not finish within {timeout}s"
    raise AssertionError(msg)


@pytest.fixture
def runner(tmp_path, repo_root):
    """A runner backed by a thread executor, for fast logic tests."""
    made = JobRunner(
        store=JobStore(tmp_path / "jobs.sqlite"),
        data_dir=tmp_path / "lake",
        runs_dir=tmp_path / "runs",
        executor_factory=lambda: ThreadPoolExecutor(max_workers=1),
    )
    made.repo_root = repo_root
    yield made
    made.shutdown()


@pytest.fixture
def smoke_config_path(repo_root):
    return repo_root / "configs" / "experiments" / "smoke.yaml"


def test_submitting_a_job_returns_it_queued_or_running(runner, smoke_config_path):
    job = runner.submit(job_type=JobType.INGEST, config_ref=smoke_config_path)
    assert job.job_type is JobType.INGEST
    assert job.status in {JobStatus.QUEUED, JobStatus.RUNNING}
    assert job.job_id


def test_an_ingest_job_completes_and_fills_the_lake(runner, smoke_config_path, tmp_path):
    job = runner.submit(job_type=JobType.INGEST, config_ref=smoke_config_path)
    assert _await_terminal(runner, job.job_id) is JobStatus.COMPLETED
    assert (tmp_path / "lake").is_dir()


def test_a_job_records_its_log_lines(runner, smoke_config_path):
    job = runner.submit(job_type=JobType.INGEST, config_ref=smoke_config_path)
    _await_terminal(runner, job.job_id)
    lines = runner.store.read_logs(job.job_id)
    assert lines, "a job that produced no log line has nothing to stream"
    assert any("SYNTH" in line.message for line in lines)


def test_a_job_reports_structured_progress(runner, smoke_config_path):
    job = runner.submit(job_type=JobType.INGEST, config_ref=smoke_config_path)
    _await_terminal(runner, job.job_id)
    progress = runner.get(job.job_id).progress
    assert progress.phase
    assert progress.fraction == pytest.approx(1.0)


def test_a_job_with_a_broken_config_fails_with_its_error(runner, tmp_path):
    missing = tmp_path / "nope.yaml"
    job = runner.submit(job_type=JobType.INGEST, config_ref=missing)
    assert _await_terminal(runner, job.job_id) is JobStatus.FAILED
    record = runner.get(job.job_id)
    assert record.error
    assert str(missing.name) in record.error


def test_a_failing_job_does_not_take_the_runner_down(runner, smoke_config_path, tmp_path):
    broken = runner.submit(job_type=JobType.INGEST, config_ref=tmp_path / "nope.yaml")
    _await_terminal(runner, broken.job_id)

    healthy = runner.submit(job_type=JobType.INGEST, config_ref=smoke_config_path)
    assert _await_terminal(runner, healthy.job_id) is JobStatus.COMPLETED


def test_cancelling_a_queued_job_marks_it_cancelled(runner, smoke_config_path):
    runner.pause()
    job = runner.submit(job_type=JobType.FULL_RUN, config_ref=smoke_config_path)
    runner.cancel(job.job_id)
    runner.resume()
    assert _await_terminal(runner, job.job_id) is JobStatus.CANCELLED


def test_cancelling_an_unknown_job_raises(runner):
    with pytest.raises(JobNotFoundError):
        runner.cancel("ghost")


def test_jobs_are_listed_newest_first(runner, smoke_config_path):
    first = runner.submit(job_type=JobType.INGEST, config_ref=smoke_config_path)
    second = runner.submit(job_type=JobType.INGEST, config_ref=smoke_config_path)
    _await_terminal(runner, second.job_id)
    assert [job.job_id for job in runner.list_jobs()][:2] == [second.job_id, first.job_id]


@pytest.mark.slow
def test_a_full_run_in_a_real_process_pool_is_visible_from_this_process(
    tmp_path, repo_root, smoke_config_path
):
    """The property the whole design rests on.

    The job executes in a different process. If its state were not visible here,
    every `GET /jobs/{id}` would report a job that never starts and never ends.
    """
    store = JobStore(tmp_path / "jobs.sqlite")
    runner = JobRunner(
        store=store, data_dir=tmp_path / "lake", runs_dir=tmp_path / "runs", max_workers=1
    )
    try:
        job = runner.submit(job_type=JobType.FULL_RUN, config_ref=smoke_config_path)
        assert _await_terminal(runner, job.job_id, timeout=300.0) is JobStatus.COMPLETED

        finished = runner.get(job.job_id)
        assert finished.run_id, "a completed full run must name the directory it wrote"
        assert (tmp_path / "runs" / finished.run_id / "tearsheet.json").is_file()
        assert finished.duration_s and finished.duration_s > 0
        assert store.read_logs(job.job_id), "the worker's log lines must reach this process"
    finally:
        runner.shutdown()
