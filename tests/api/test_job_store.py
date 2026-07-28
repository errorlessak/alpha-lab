"""The SQLite job store (§14.2).

Job state and a rolling log table, both readable from a different process than
the one writing them — the job runner hands work to a process pool, and the API
answering `GET /jobs/{id}` is not that process.
"""

from __future__ import annotations

import datetime as dt

import pytest

from alphaservice.errors import JobNotFoundError
from alphaservice.jobs.store import JobStatus, JobStore, JobType

NOW = dt.datetime(2026, 7, 28, 12, 0, 0, tzinfo=dt.UTC)


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs.sqlite")


def test_a_new_job_starts_queued(store):
    job = store.create(job_type=JobType.FULL_RUN, config_ref="configs/smoke.yaml", now=NOW)
    assert job.status is JobStatus.QUEUED
    assert job.job_id
    assert job.created_at == NOW
    assert job.run_id is None


def test_a_job_can_be_read_back_by_id(store):
    created = store.create(job_type=JobType.INGEST, config_ref="c.yaml", now=NOW)
    assert store.get(created.job_id) == created


def test_an_unknown_job_raises_rather_than_returning_none(store):
    # The API turns this into a 404. Returning None would make every caller
    # responsible for remembering to check.
    with pytest.raises(JobNotFoundError, match="ghost"):
        store.get("ghost")


def test_a_job_records_when_it_started_and_finished(store):
    job = store.create(job_type=JobType.FULL_RUN, config_ref="c.yaml", now=NOW)
    started = NOW + dt.timedelta(seconds=1)
    finished = NOW + dt.timedelta(seconds=11)

    store.mark_running(job.job_id, now=started)
    assert store.get(job.job_id).status is JobStatus.RUNNING
    assert store.get(job.job_id).started_at == started

    store.mark_finished(job.job_id, JobStatus.COMPLETED, run_id="run-1", now=finished)
    done = store.get(job.job_id)
    assert done.status is JobStatus.COMPLETED
    assert done.finished_at == finished
    assert done.run_id == "run-1"
    assert done.duration_s == pytest.approx(10.0)


def test_a_failed_job_keeps_its_traceback(store):
    job = store.create(job_type=JobType.FULL_RUN, config_ref="c.yaml", now=NOW)
    store.mark_finished(job.job_id, JobStatus.FAILED, error="Traceback...\nBoom", now=NOW)
    assert "Boom" in store.get(job.job_id).error


def test_finishing_a_job_as_running_is_refused(store):
    # `running` is not a terminal state. Allowing it would let a crashed job be
    # recorded as still in flight forever.
    job = store.create(job_type=JobType.FULL_RUN, config_ref="c.yaml", now=NOW)
    with pytest.raises(ValueError, match="terminal"):
        store.mark_finished(job.job_id, JobStatus.RUNNING, now=NOW)


def test_progress_is_structured_not_a_string(store):
    # §14.2 asks for phase, step, total and message as separate fields, so a UI
    # can draw a progress bar without parsing prose.
    job = store.create(job_type=JobType.FULL_RUN, config_ref="c.yaml", now=NOW)
    store.update_progress(job.job_id, phase="train", step=2, total=4, message="scoring models")
    progress = store.get(job.job_id).progress
    assert progress.phase == "train"
    assert progress.step == 2
    assert progress.total == 4
    assert progress.message == "scoring models"
    assert progress.fraction == pytest.approx(0.5)


def test_progress_without_a_total_has_no_fraction(store):
    job = store.create(job_type=JobType.FULL_RUN, config_ref="c.yaml", now=NOW)
    store.update_progress(job.job_id, phase="ingest", step=0, total=0, message="fetching")
    assert store.get(job.job_id).progress.fraction is None


def test_cancellation_is_a_request_the_worker_observes(store):
    # A process pool cannot kill one of its workers, so cancellation is
    # cooperative: the flag is set here and the worker checks it between phases.
    job = store.create(job_type=JobType.FULL_RUN, config_ref="c.yaml", now=NOW)
    assert store.get(job.job_id).cancel_requested is False
    store.request_cancel(job.job_id)
    assert store.get(job.job_id).cancel_requested is True
    assert store.cancel_requested(job.job_id) is True


def test_cancelling_a_finished_job_is_refused(store):
    job = store.create(job_type=JobType.FULL_RUN, config_ref="c.yaml", now=NOW)
    store.mark_finished(job.job_id, JobStatus.COMPLETED, run_id="r", now=NOW)
    with pytest.raises(ValueError, match="already finished"):
        store.request_cancel(job.job_id)


def test_log_lines_are_returned_in_the_order_they_arrived(store):
    job = store.create(job_type=JobType.FULL_RUN, config_ref="c.yaml", now=NOW)
    for i in range(5):
        store.append_log(job.job_id, level="INFO", phase="train", message=f"line {i}", now=NOW)

    lines = store.read_logs(job.job_id)
    assert [line.message for line in lines] == [f"line {i}" for i in range(5)]
    assert [line.seq for line in lines] == [1, 2, 3, 4, 5]


def test_logs_can_be_read_from_a_cursor(store):
    # This is what the SSE stream does: it remembers the last sequence number it
    # sent and asks for what came after it.
    job = store.create(job_type=JobType.FULL_RUN, config_ref="c.yaml", now=NOW)
    for i in range(5):
        store.append_log(job.job_id, level="INFO", phase="run", message=str(i), now=NOW)

    assert [line.message for line in store.read_logs(job.job_id, after_seq=3)] == ["3", "4"]
    assert store.read_logs(job.job_id, after_seq=5) == []


def test_the_log_table_rolls_rather_than_growing_without_bound(tmp_path):
    # §14.2 calls it a rolling log table. A run that logs for an hour must not
    # be able to fill the disk through the job database.
    store = JobStore(tmp_path / "jobs.sqlite", max_log_lines=10)
    job = store.create(job_type=JobType.FULL_RUN, config_ref="c.yaml", now=NOW)
    for i in range(25):
        store.append_log(job.job_id, level="INFO", phase="run", message=str(i), now=NOW)

    lines = store.read_logs(job.job_id)
    assert len(lines) == 10
    assert [line.message for line in lines] == [str(i) for i in range(15, 25)]


def test_jobs_are_listed_newest_first(store):
    first = store.create(job_type=JobType.INGEST, config_ref="a.yaml", now=NOW)
    second = store.create(
        job_type=JobType.FULL_RUN, config_ref="b.yaml", now=NOW + dt.timedelta(seconds=1)
    )
    assert [job.job_id for job in store.list_jobs()] == [second.job_id, first.job_id]


def test_jobs_can_be_filtered_by_status(store):
    queued = store.create(job_type=JobType.INGEST, config_ref="a.yaml", now=NOW)
    done = store.create(job_type=JobType.FULL_RUN, config_ref="b.yaml", now=NOW)
    store.mark_finished(done.job_id, JobStatus.COMPLETED, run_id="r", now=NOW)

    assert [j.job_id for j in store.list_jobs(status=JobStatus.QUEUED)] == [queued.job_id]
    assert [j.job_id for j in store.list_jobs(status=JobStatus.COMPLETED)] == [done.job_id]


def test_a_second_connection_sees_what_the_first_wrote(tmp_path):
    """The whole point: the writer and the reader are different processes.

    The job runs in a process-pool worker; the request handler answering
    `GET /jobs/{id}` runs in the API process. If a second connection could not
    see the first's writes, every status would be stale.
    """
    path = tmp_path / "jobs.sqlite"
    writer = JobStore(path)
    job = writer.create(job_type=JobType.FULL_RUN, config_ref="c.yaml", now=NOW)
    writer.append_log(job.job_id, level="INFO", phase="run", message="hello", now=NOW)

    reader = JobStore(path)
    assert reader.get(job.job_id).job_id == job.job_id
    assert [line.message for line in reader.read_logs(job.job_id)] == ["hello"]
