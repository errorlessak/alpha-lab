"""The in-process job runner (§14.2, ADR 0008).

A `ProcessPoolExecutor` with one or two workers, SQLite for state, and no
broker. ADR 0008 explains why that is the right call for a single-user research
tool and what changes at scale.

**Why a process and not a thread.** A run holds the GIL for minutes at a time
inside numpy and LightGBM, and it calls `set_global_seed`, which mutates
interpreter-wide state. Two runs sharing an interpreter would fight over both,
and the second one's results would depend on the first — which is the one thing
this project cannot allow.

**Cancellation is cooperative.** A process pool cannot kill an individual
worker: `Future.cancel` only works before the job starts, and there is no
supported way to signal one member of the pool. So `POST /jobs/{id}/cancel`
sets a flag in SQLite, and the worker checks it at each phase boundary through
`run_experiment(should_cancel=...)`. The cost is latency — a cancel lands at the
end of the current phase, not instantly. The benefit is that the run directory
is finalised as `cancelled` with everything it managed to write intact, rather
than being killed mid-parquet-write and left looking valid.

**Log lines reach the API through the database.** The worker attaches a loguru
sink that appends to the job's rolling log table. The API's SSE endpoint polls
that table. The complete, authoritative log is still `runs/{run_id}/logs.jsonl`;
this is the live tail.
"""

from __future__ import annotations

import datetime as dt
import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from alphaservice.jobs.store import JobStatus, JobStore, JobType

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Executor

    from alphaservice.jobs.store import JobRecord

__all__ = ["JobRunner", "execute_job"]

_ORPHAN_SCAN_LIMIT: Final[int] = 1_000
"""How far back a starting runner looks for jobs stranded `queued`. Generous
enough that a real store is covered in one pass, bounded so a corrupted database
cannot make startup unbounded."""

DEFAULT_MAX_WORKERS: Final[int] = 1
"""§14.2 says one to two. One by default: a second concurrent run doubles peak
memory for a tool whose bottleneck is a single researcher's attention."""

_PHASES: Final[dict[JobType, tuple[str, ...]]] = {
    JobType.INGEST: ("ingest",),
    JobType.FEATURES: ("ingest", "features"),
    JobType.TRAIN: ("ingest", "features", "train"),
    JobType.BACKTEST: ("ingest", "features", "train", "backtest"),
    JobType.FULL_RUN: ("ingest", "features", "train", "report"),
}
"""Phases each job type passes through, so `step`/`total` progress is a real
fraction rather than a spinner (§14.2)."""


class JobRunner:
    """Submits jobs to a pool and reports their state.

    Attributes:
        store: The SQLite job store. Shared with the API process.
    """

    def __init__(
        self,
        *,
        store: JobStore,
        data_dir: Path | str,
        runs_dir: Path | str,
        max_workers: int = DEFAULT_MAX_WORKERS,
        executor_factory: Callable[[], Executor] | None = None,
    ) -> None:
        """Build a runner.

        Args:
            store: The job store.
            data_dir: Lake root passed to every job.
            runs_dir: Artifact store root passed to every job.
            max_workers: Pool size.
            executor_factory: Override the executor. Tests inject a thread pool
                so the logic can be exercised in milliseconds; production uses
                the process pool, and one test uses the real thing so the
                cross-process path is not left unproven.
        """
        self.store = store
        self.data_dir = Path(data_dir)
        self.runs_dir = Path(runs_dir)
        self._executor = (
            executor_factory() if executor_factory else ProcessPoolExecutor(max_workers=max_workers)
        )
        self._futures: dict[str, Future[None]] = {}
        self._paused = False
        self._pending: list[tuple[str, JobType, str]] = []
        self._reap_orphans()

    def _reap_orphans(self) -> None:
        """Close out jobs left `queued` by a process that is no longer running.

        `submit` dispatches immediately, so a job is `queued` only in the window
        between the row being written and the executor accepting it. Nothing
        ever picks a queued job up later — not this runner, not any other. A row
        still in that state when a runner starts therefore belongs to a process
        that died holding it, and it will stay there forever, indistinguishable
        in the jobs list from one that is about to run.

        ADR 0008 says jobs do not survive a restart. This does not change that:
        it makes the record say so instead of lying.

        `running` is deliberately left alone. A worker may be mid-write on a run
        directory, and marking its job failed from another process would
        contradict the artifact it is about to finish.
        """
        for record in self.store.list_jobs(status=JobStatus.QUEUED, limit=_ORPHAN_SCAN_LIMIT):
            self.store.mark_finished(
                record.job_id,
                JobStatus.FAILED,
                error=(
                    "The service restarted before this job started. Nothing resumes a queued "
                    "job (ADR 0008) — submit it again."
                ),
            )

    def __repr__(self) -> str:
        """Return a debug representation naming the store."""
        return f"JobRunner({self.store.path.as_posix()!r})"

    def pause(self) -> None:
        """Hold submissions in a queue instead of dispatching them.

        Exists so a test can observe a job in the `queued` state, which is
        otherwise a race: with a free worker, a job starts before `submit`
        returns.
        """
        self._paused = True

    def resume(self) -> None:
        """Dispatch everything held while paused."""
        self._paused = False
        pending, self._pending = self._pending, []
        for job_id, job_type, config_ref in pending:
            self._dispatch(job_id, job_type, config_ref)

    def submit(
        self, *, job_type: JobType, config_ref: Path | str, job_id: str | None = None
    ) -> JobRecord:
        """Queue a job and hand it to the pool.

        Args:
            job_type: What to run.
            config_ref: Path to the experiment YAML.
            job_id: Override the generated id.

        Returns:
            The record as it was queued. Its status may already have advanced
            by the time the caller reads it — the worker starts immediately.
        """
        reference = str(config_ref)
        record = self.store.create(job_type=job_type, config_ref=reference, job_id=job_id)
        if self._paused:
            self._pending.append((record.job_id, job_type, reference))
        else:
            self._dispatch(record.job_id, job_type, reference)
        return record

    def _dispatch(self, job_id: str, job_type: JobType, config_ref: str) -> None:
        """Hand one job to the executor.

        Args:
            job_id: The job id.
            job_type: What to run.
            config_ref: Path to the experiment YAML.
        """
        self._futures[job_id] = self._executor.submit(
            execute_job,
            job_id,
            str(self.store.path),
            job_type.value,
            config_ref,
            str(self.data_dir),
            str(self.runs_dir),
            self.store.max_log_lines,
        )

    def get(self, job_id: str) -> JobRecord:
        """Read one job's state.

        Args:
            job_id: The job id.

        Returns:
            The record.

        Raises:
            JobNotFoundError: If no such job exists.
        """
        return self.store.get(job_id)

    def list_jobs(self, *, status: JobStatus | None = None, limit: int = 100) -> list[JobRecord]:
        """List jobs, newest first.

        Args:
            status: Optional status filter.
            limit: Maximum rows.

        Returns:
            The records.
        """
        return self.store.list_jobs(status=status, limit=limit)

    def cancel(self, job_id: str) -> JobRecord:
        """Ask a job to stop.

        A job that has not started is cancelled outright. A running one is
        asked, and stops at its next phase boundary.

        Args:
            job_id: The job id.

        Returns:
            The updated record.

        Raises:
            JobNotFoundError: If no such job exists.
            ValueError: If the job has already finished.
        """
        record = self.store.request_cancel(job_id)
        self._pending = [entry for entry in self._pending if entry[0] != job_id]

        future = self._futures.get(job_id)
        if future is not None and future.cancel():
            # Never started, so no worker will ever observe the flag.
            self.store.mark_finished(job_id, JobStatus.CANCELLED, error="Cancelled before it ran.")
        elif record.status is JobStatus.QUEUED and future is None:
            self.store.mark_finished(job_id, JobStatus.CANCELLED, error="Cancelled while queued.")
        return self.store.get(job_id)

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop the executor.

        Args:
            wait: Whether to block until running jobs finish.
        """
        self._executor.shutdown(wait=wait)


def _failure_detail(exc: BaseException) -> str:
    """Render a failure for the `error` field the UI shows.

    The full traceback is written to the job's log either way — that is what the
    log is for. This is the one line a reader sees first, and §13.5 asks it to
    name the cause and the fix.

    Args:
        exc: The exception that ended the job.

    Returns:
        The exception's own message when it is an `AlphaError`, whose messages
        are written to be read — "Could not parse YAML in pyproject.toml:
        expected '<document start>'" is the whole answer, and forty frames of
        virtualenv paths in front of it are not. Anything else is a bug rather
        than a bad request, and for those the traceback is the most useful thing
        anyone can be shown.
    """
    from alphacore.errors import AlphaError  # noqa: PLC0415 - worker-side import
    from alphacore.runtime.logging import logger  # noqa: PLC0415 - worker-side import

    logger.opt(exception=exc).error("the job failed")
    if isinstance(exc, AlphaError):
        return str(exc)
    return "".join(traceback.format_exception(exc))


def execute_job(
    job_id: str,
    store_path: str,
    job_type: str,
    config_ref: str,
    data_dir: str,
    runs_dir: str,
    max_log_lines: int,
) -> None:
    """Run one job in a pool worker rather than in the API process.

    **This function crosses a process boundary.**

    Every argument is a primitive because it has to survive pickling to another
    interpreter. The worker opens its own store connection, attaches its own log
    sink, and reports its own terminal state — the parent only reads.

    Args:
        job_id: The job id.
        store_path: Path to the job database.
        job_type: A `JobType` value.
        config_ref: Path to the experiment YAML.
        data_dir: Lake root.
        runs_dir: Artifact store root.
        max_log_lines: Rolling log tail size.
    """
    store = JobStore(store_path, max_log_lines=max_log_lines)
    kind = JobType(job_type)
    sink_id: int | None = None

    try:
        store.mark_running(job_id)
        sink_id = _attach_log_sink(store, job_id)
        run_id = _dispatch_job(store, job_id, kind, config_ref, data_dir, runs_dir)
    except _CancelledError:
        store.mark_finished(job_id, JobStatus.CANCELLED, error="The job was cancelled.")
    except BaseException as exc:
        if _is_cancellation(exc):
            store.mark_finished(job_id, JobStatus.CANCELLED, error=str(exc))
        else:
            # The traceback goes to the log, which is what it is for; the
            # `error` field is what the UI puts in front of a reader, and §13.5
            # asks that to name the cause and the fix. Every library error here
            # is an `AlphaError` whose message already does — "Could not parse
            # YAML in pyproject.toml: expected '<document start>'" is the whole
            # answer, and forty frames of venv paths in front of it are not.
            #
            # Anything else is a bug rather than a bad request, and for those
            # the traceback *is* the most useful thing anyone can be shown.
            store.mark_finished(job_id, JobStatus.FAILED, error=_failure_detail(exc))
    else:
        store.mark_finished(job_id, JobStatus.COMPLETED, run_id=run_id)
    finally:
        if sink_id is not None:
            _remove_log_sink(sink_id)


class _CancelledError(Exception):
    """Raised inside a worker when the cancel flag is observed."""


def _is_cancellation(exc: BaseException) -> bool:
    """Whether an exception means "somebody stopped this", not "this broke".

    Args:
        exc: The exception.

    Returns:
        True for a cancellation.
    """
    from alphacore.errors import RunCancelledError  # noqa: PLC0415 - worker-side import

    return isinstance(exc, RunCancelledError | _CancelledError | KeyboardInterrupt)


def _attach_log_sink(store: JobStore, job_id: str) -> int:
    """Route this worker's log lines into the job's rolling log table.

    Args:
        store: The job store.
        job_id: The job id.

    Returns:
        The loguru sink id, for removal afterwards.
    """
    from alphacore.runtime.logging import logger  # noqa: PLC0415 - worker-side import

    def sink(message: Any) -> None:  # noqa: ANN401 - loguru's Message type
        record = message.record
        store.append_log(
            job_id,
            level=record["level"].name,
            phase=str(record["extra"].get("phase", "")),
            message=record["message"],
            now=record["time"].astimezone(dt.UTC),
        )

    return int(logger.add(sink, level="INFO", enqueue=False))


def _remove_log_sink(sink_id: int) -> None:
    """Detach a log sink.

    Args:
        sink_id: The id returned by `_attach_log_sink`.
    """
    from alphacore.runtime.logging import remove_sink  # noqa: PLC0415 - worker-side import

    remove_sink(sink_id)


def _dispatch_job(
    store: JobStore,
    job_id: str,
    job_type: JobType,
    config_ref: str,
    data_dir: str,
    runs_dir: str,
) -> str | None:
    """Run the stage the job asked for.

    Args:
        store: The job store.
        job_id: The job id.
        job_type: What to run.
        config_ref: Path to the experiment YAML.
        data_dir: Lake root.
        runs_dir: Artifact store root.

    Returns:
        The run id, for a `full_run`; None for the stage jobs, which write
        nothing (a half-populated run directory would satisfy §16's filenames
        without satisfying its meaning).

    Raises:
        _CancelledError: If cancellation is observed before the work starts.
        AlphaError: Any stage failure.
    """
    # Worker-side imports: the parent process never needs alphacore's heavy
    # modules, and paying for them at API start-up would make `--reload` painful.
    from alphacore.config import load_config  # noqa: PLC0415
    from alphacore.pipelines import (  # noqa: PLC0415
        build_feature_set,
        ingest_symbols,
        load_bars,
        run_experiment,
        train_models,
    )
    from alphacore.runtime.artifacts import ArtifactStore  # noqa: PLC0415

    phases = _PHASES[job_type]
    total = len(phases)

    def advance(index: int, message: str) -> None:
        store.update_progress(
            job_id, phase=phases[min(index, total - 1)], step=index, total=total, message=message
        )

    def cancelled() -> bool:
        return store.cancel_requested(job_id)

    if cancelled():
        raise _CancelledError

    config = load_config(Path(config_ref), use_env=False)

    if job_type is JobType.FULL_RUN:
        advance(0, "starting the golden path")
        summary = run_experiment(
            config,
            data_dir=data_dir,
            store=ArtifactStore(runs_dir),
            should_cancel=cancelled,
        )
        advance(total, f"wrote {summary.run_id}")
        return summary.run_id

    advance(0, "ingesting")
    ingest_symbols(config, data_dir=data_dir)
    if job_type is JobType.INGEST:
        advance(total, "ingest complete")
        return None

    advance(1, "building features")
    feature_set = build_feature_set(config, load_bars(config, data_dir=data_dir))
    if job_type is JobType.FEATURES:
        advance(total, f"{feature_set.manifest.n_columns} columns")
        return None

    if cancelled():
        raise _CancelledError
    advance(2, "training")
    training = train_models(config, feature_set)
    advance(total, f"best model {training.selected}")
    return None
