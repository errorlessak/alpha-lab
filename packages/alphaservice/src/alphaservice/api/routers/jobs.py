"""Job submission, status, cancellation, and the SSE log stream (§14.1, §14.2).

The SSE stream is hand-rolled over `StreamingResponse` rather than pulled in as
a dependency. It is a poll of the job's rolling log table plus a status frame,
and the two things that make it correct — resuming from `Last-Event-ID`, and
noticing that the client went away — are ten lines each.

Unlike every run resource, nothing here is cacheable. A job's representation
changes by design, so these responses carry `Cache-Control: no-store`.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final

from fastapi import APIRouter, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from alphaservice.api.deps import JobStoreDep, RunnerDep, SettingsDep
from alphaservice.api.problems import Problem, problem_responses
from alphaservice.errors import JobNotFoundError
from alphaservice.jobs.store import JobRecord, JobStatus, JobType

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from alphaservice.jobs.store import JobStore

__all__ = ["router"]

router = APIRouter(tags=["jobs"])

_POLL_INTERVAL_S: Final[float] = 0.25
"""How often the stream looks for new log lines. Fast enough to read as live,
slow enough that an idle job is not a busy loop."""

_HEARTBEAT_EVERY: Final[int] = 60
"""Polls between keep-alive comments. Proxies close idle connections, and a
comment line costs nothing."""


class JobSubmission(BaseModel):
    """What `POST /jobs` accepts."""

    type: JobType = Field(description="Which stage to run.")
    config_ref: str = Field(
        min_length=1,
        description="Path to the experiment YAML, resolved on the server's filesystem.",
    )


class JobProgressOut(BaseModel):
    """Structured progress (§14.2)."""

    phase: str
    step: int
    total: int
    message: str
    fraction: float | None = Field(
        default=None, description="Null when the total is unknown — not zero."
    )


class JobOut(BaseModel):
    """A job as the API reports it."""

    job_id: str
    type: JobType
    status: JobStatus
    config_ref: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    run_id: str | None = None
    error: str | None = None
    progress: JobProgressOut
    cancel_requested: bool


class JobList(BaseModel):
    """Jobs, newest first."""

    jobs: list[JobOut]
    total: int = Field(ge=0)


def _out(record: JobRecord) -> JobOut:
    """Render a job record for the wire.

    Args:
        record: The stored record.

    Returns:
        The response model.
    """
    return JobOut(
        job_id=record.job_id,
        type=record.job_type,
        status=record.status,
        config_ref=record.config_ref,
        created_at=record.created_at.isoformat(),
        started_at=record.started_at.isoformat() if record.started_at else None,
        finished_at=record.finished_at.isoformat() if record.finished_at else None,
        duration_s=record.duration_s,
        run_id=record.run_id,
        error=record.error,
        progress=JobProgressOut(
            phase=record.progress.phase,
            step=record.progress.step,
            total=record.progress.total,
            message=record.progress.message,
            fraction=record.progress.fraction,
        ),
        cancel_requested=record.cancel_requested,
    )


def _lookup(store: JobStore, job_id: str) -> JobRecord:
    """Read a job or raise a 404 problem.

    Args:
        store: The job store.
        job_id: The job id.

    Returns:
        The record.

    Raises:
        Problem: With 404 if no such job exists.
    """
    try:
        return store.get(job_id)
    except JobNotFoundError as exc:
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="job-not-found",
            title="Job not found",
            detail=f"No job {job_id!r}. List them with GET /api/v1/jobs.",
        ) from exc


@router.post(
    "/jobs",
    operation_id="createJob",
    summary="Submit a job",
    status_code=status.HTTP_202_ACCEPTED,
    responses=problem_responses(422, 503),
)
def create_job(
    submission: JobSubmission, runner: RunnerDep, settings: SettingsDep, response: Response
) -> JobOut:
    """Queue a job.

    Args:
        submission: The job type and the config to run it with.
        runner: The job runner.
        settings: Service settings.
        response: The response, for `Location` and cache headers.

    Returns:
        The queued job, with `202 Accepted` and a `Location` pointing at it.

    Raises:
        Problem: With 422 if the config file does not exist. Failing here rather
            than in the worker means the caller learns about a typo now instead
            of from a job that dies two seconds later.
    """
    config_path = Path(submission.config_ref)
    if not config_path.is_file():
        raise Problem(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            problem_type="config-not-found",
            title="Configuration file not found",
            detail=(
                f"{submission.config_ref!r} does not exist on the server. Paths are resolved "
                f"relative to the service's working directory ({Path.cwd().as_posix()})."
            ),
        )

    record = runner.submit(job_type=submission.type, config_ref=str(config_path))
    response.status_code = status.HTTP_202_ACCEPTED
    response.headers["Location"] = f"/api/v1/jobs/{record.job_id}"
    response.headers["Cache-Control"] = "no-store"
    _ = settings
    return _out(record)


@router.get(
    "/jobs",
    operation_id="listJobs",
    summary="List jobs",
    responses=problem_responses(422, 503),
)
def list_jobs(
    store: JobStoreDep,
    response: Response,
    job_status: Annotated[JobStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> JobList:
    """List jobs, newest first.

    Args:
        store: The job store.
        response: The response, for cache headers.
        job_status: Optional status filter.
        limit: Page size.

    Returns:
        The jobs.
    """
    response.headers["Cache-Control"] = "no-store"
    records = store.list_jobs(status=job_status, limit=limit)
    return JobList(jobs=[_out(record) for record in records], total=len(records))


@router.get(
    "/jobs/{job_id}",
    operation_id="getJob",
    summary="One job's state",
    responses=problem_responses(404, 422, 503),
)
def get_job(job_id: str, store: JobStoreDep, response: Response) -> JobOut:
    """Read one job.

    Args:
        job_id: The job id.
        store: The job store.
        response: The response, for cache headers.

    Returns:
        The job.
    """
    response.headers["Cache-Control"] = "no-store"
    return _out(_lookup(store, job_id))


@router.post(
    "/jobs/{job_id}/cancel",
    operation_id="cancelJob",
    summary="Ask a running job to stop",
    responses=problem_responses(404, 409, 422, 503),
)
def cancel_job(job_id: str, runner: RunnerDep, response: Response) -> JobOut:
    """Request cancellation.

    Cancellation is cooperative: a running job stops at its next phase boundary
    and its run directory is finalised as `cancelled` rather than killed
    mid-write (§14.2).

    Args:
        job_id: The job id.
        runner: The job runner.
        response: The response, for cache headers.

    Returns:
        The updated job.

    Raises:
        Problem: With 404 if no such job exists, or 409 if it already finished.
    """
    response.headers["Cache-Control"] = "no-store"
    try:
        return _out(runner.cancel(job_id))
    except JobNotFoundError as exc:
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="job-not-found",
            title="Job not found",
            detail=f"No job {job_id!r}.",
        ) from exc
    except ValueError as exc:
        raise Problem(
            status_code=status.HTTP_409_CONFLICT,
            problem_type="job-already-finished",
            title="Job has already finished",
            detail=str(exc),
        ) from exc


def _sse(event: str, data: dict[str, Any], *, event_id: int | None = None) -> str:
    """Format one server-sent event.

    Args:
        event: Event name.
        data: JSON-serialisable payload.
        event_id: Optional id, which the browser sends back as `Last-Event-ID`
            when it reconnects.

    Returns:
        The wire representation, terminated by a blank line.
    """
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, sort_keys=True, default=str)}")
    return "\n".join(lines) + "\n\n"


@router.get(
    "/jobs/{job_id}/events",
    operation_id="streamJobEvents",
    summary="Live log and progress stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
            "description": "An SSE stream of `log`, `progress`, and one terminal `status` event.",
        },
        **problem_responses(404, 422, 503),
    },
)
async def stream_events(job_id: str, store: JobStoreDep, request: Request) -> StreamingResponse:
    """Stream a job's log lines and progress until it finishes.

    Resumable: the browser replays `Last-Event-ID` on reconnect and the stream
    picks up from the line after it, so a dropped connection does not lose the
    middle of a run's log.

    Args:
        job_id: The job id.
        store: The job store.
        request: The request, for `Last-Event-ID` and disconnect detection.

    Returns:
        The event stream.

    Raises:
        Problem: With 404 if no such job exists.
    """
    _lookup(store, job_id)
    resume_from = _last_event_id(request)

    async def events() -> AsyncIterator[str]:
        cursor = resume_from
        last_progress: tuple[str, int, int, str] | None = None
        polls = 0

        while True:
            if await request.is_disconnected():
                return

            for line in store.read_logs(job_id, after_seq=cursor):
                cursor = line.seq
                yield _sse(
                    "log",
                    {
                        "seq": line.seq,
                        "ts": line.ts.isoformat(),
                        "level": line.level,
                        "phase": line.phase,
                        "message": line.message,
                    },
                    event_id=line.seq,
                )

            record = store.get(job_id)
            progress = (
                record.progress.phase,
                record.progress.step,
                record.progress.total,
                record.progress.message,
            )
            if progress != last_progress:
                last_progress = progress
                yield _sse(
                    "progress",
                    {
                        "phase": record.progress.phase,
                        "step": record.progress.step,
                        "total": record.progress.total,
                        "message": record.progress.message,
                        "fraction": record.progress.fraction,
                    },
                )

            if record.status.is_terminal:
                # One last sweep: the worker's final lines can land between the
                # read above and the status flipping.
                for line in store.read_logs(job_id, after_seq=cursor):
                    cursor = line.seq
                    yield _sse(
                        "log",
                        {
                            "seq": line.seq,
                            "ts": line.ts.isoformat(),
                            "level": line.level,
                            "phase": line.phase,
                            "message": line.message,
                        },
                        event_id=line.seq,
                    )
                yield _sse(
                    "status",
                    {
                        "job_id": record.job_id,
                        "status": record.status.value,
                        "run_id": record.run_id,
                        "error": record.error,
                        "finished_at": record.finished_at.isoformat()
                        if record.finished_at
                        else dt.datetime.now(dt.UTC).isoformat(),
                    },
                )
                return

            polls += 1
            if polls % _HEARTBEAT_EVERY == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(_POLL_INTERVAL_S)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            # Nginx buffers by default, which turns a live stream into one
            # delivery at the end. This header is how you ask it not to.
            "X-Accel-Buffering": "no",
        },
    )


def _last_event_id(request: Request) -> int:
    """Read the resume point from a reconnecting client.

    Args:
        request: The request.

    Returns:
        The sequence number to resume after, or 0 to start from the beginning.
    """
    raw = request.headers.get("last-event-id") or request.query_params.get("after")
    try:
        return max(int(raw), 0) if raw else 0
    except ValueError:
        return 0
