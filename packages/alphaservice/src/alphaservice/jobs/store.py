"""Job state and the rolling log table, in SQLite (§14.2).

SQLite rather than an in-memory dictionary for one reason: **the writer and the
reader are different processes.** A job executes in a process-pool worker; the
request handler answering `GET /jobs/{id}` runs in the API process. A dictionary
would be per-process and every status would be stale.

Write-ahead logging is on, which is what makes concurrent readers see a writer's
committed rows without blocking. Timestamps are stored as ISO-8601 UTC strings
so the file is legible with the `sqlite3` shell — a job database nobody can read
during an incident is a job database that does not help.

The log table rolls. A run that logs for an hour must not be able to fill the
disk through this file, so each job keeps its most recent `max_log_lines` and
older lines are dropped as new ones arrive. The authoritative, complete log is
always `runs/{run_id}/logs.jsonl`; this table is the live tail.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from alphaservice.errors import JobNotFoundError

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "DEFAULT_MAX_LOG_LINES",
    "JobLogLine",
    "JobProgress",
    "JobRecord",
    "JobStatus",
    "JobStore",
    "JobType",
]

DEFAULT_MAX_LOG_LINES: Final[int] = 5_000
"""Lines kept per job. The complete log lives in the run directory; this is the
live tail the SSE stream reads from."""


class JobType(StrEnum):
    """What a job runs (§14.1).

    The set is closed: `POST /jobs` with an unknown type is a client error, not
    an invitation to run something the service cannot describe.
    """

    INGEST = "ingest"
    FEATURES = "features"
    TRAIN = "train"
    BACKTEST = "backtest"
    FULL_RUN = "full_run"


class JobStatus(StrEnum):
    """Lifecycle state of a job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether no further transition is possible from this state."""
        return self in _TERMINAL


_TERMINAL: Final[frozenset[JobStatus]] = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
)


@dataclass(frozen=True)
class JobProgress:
    """Structured progress, as §14.2 requires it.

    Separate fields rather than a formatted string, so a progress bar can be
    drawn without parsing prose.

    Attributes:
        phase: Pipeline phase, e.g. `train`.
        step: Steps completed.
        total: Steps expected, or 0 when the phase cannot say.
        message: Human-readable detail.
    """

    phase: str = ""
    step: int = 0
    total: int = 0
    message: str = ""

    @property
    def fraction(self) -> float | None:
        """Completed fraction, or None when the total is unknown.

        Returns:
            A value in `[0, 1]`, or None. Deliberately not 0.0 for an unknown
            total: a progress bar sitting at zero says something different from
            a progress bar that is not drawn.
        """
        if self.total <= 0:
            return None
        return min(max(self.step / self.total, 0.0), 1.0)


@dataclass(frozen=True)
class JobRecord:
    """One row of the jobs table.

    Attributes:
        job_id: Opaque identifier, generated here.
        job_type: What the job runs.
        status: Lifecycle state.
        config_ref: Path to the experiment YAML the job was submitted with.
        created_at: When it was queued.
        started_at: When a worker picked it up.
        finished_at: When it reached a terminal state.
        run_id: The run directory it produced, once it has one.
        error: Traceback text, when it failed.
        progress: Structured progress.
        cancel_requested: Whether a cancellation has been asked for.
    """

    job_id: str
    job_type: JobType
    status: JobStatus
    config_ref: str
    created_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    run_id: str | None = None
    error: str | None = None
    progress: JobProgress = JobProgress()
    cancel_requested: bool = False

    @property
    def duration_s(self) -> float | None:
        """Wall-clock seconds from start to finish, or None if unfinished."""
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class JobLogLine:
    """One line of a job's live log tail.

    Attributes:
        seq: Monotonic sequence number within the job. The SSE stream uses it
            as its cursor and as the event id, so a reconnecting client can
            resume from where it stopped.
        ts: When the line was emitted.
        level: Log level.
        phase: Pipeline phase the line came from.
        message: The line itself.
    """

    seq: int
    ts: dt.datetime
    level: str
    phase: str
    message: str


_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT PRIMARY KEY,
    job_type         TEXT NOT NULL,
    status           TEXT NOT NULL,
    config_ref       TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT,
    run_id           TEXT,
    error            TEXT,
    progress_phase   TEXT NOT NULL DEFAULT '',
    progress_step    INTEGER NOT NULL DEFAULT 0,
    progress_total   INTEGER NOT NULL DEFAULT 0,
    progress_message TEXT NOT NULL DEFAULT '',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    sort_key         INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS job_logs (
    job_id  TEXT NOT NULL,
    seq     INTEGER NOT NULL,
    ts      TEXT NOT NULL,
    level   TEXT NOT NULL,
    phase   TEXT NOT NULL,
    message TEXT NOT NULL,
    PRIMARY KEY (job_id, seq)
);
CREATE INDEX IF NOT EXISTS job_logs_by_seq ON job_logs (job_id, seq);
CREATE INDEX IF NOT EXISTS jobs_by_sort_key ON jobs (sort_key DESC);
"""


def _iso(value: dt.datetime | None) -> str | None:
    """Render a datetime as UTC ISO-8601, or None.

    Args:
        value: The datetime.

    Returns:
        The ISO string, or None.
    """
    return None if value is None else value.astimezone(dt.UTC).isoformat()


def _parse(value: str | None) -> dt.datetime | None:
    """Parse a stored ISO-8601 timestamp.

    Args:
        value: The stored string.

    Returns:
        A timezone-aware datetime, or None.
    """
    return None if value is None else dt.datetime.fromisoformat(value)


class JobStore:
    """The jobs database.

    Safe to open from several processes at once: each instance holds its own
    connection, and WAL lets readers proceed while a writer commits.
    """

    def __init__(self, path: Path | str, *, max_log_lines: int = DEFAULT_MAX_LOG_LINES) -> None:
        """Open or create the database.

        Args:
            path: Database file. Parent directories are created.
            max_log_lines: Lines retained per job before the tail rolls.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_log_lines = max_log_lines
        self._sequence = 0
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def __repr__(self) -> str:
        """Return a debug representation naming the database file."""
        return f"JobStore({self.path.as_posix()!r})"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection with WAL and a busy timeout.

        A short timeout rather than none: two processes committing at the same
        instant is normal here, and failing the request instead of waiting 50
        milliseconds would be the wrong trade.

        Yields:
            The connection, committed on clean exit and closed either way.
        """
        connection = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            yield connection
        finally:
            connection.close()

    def create(
        self,
        *,
        job_type: JobType,
        config_ref: str,
        job_id: str | None = None,
        now: dt.datetime | None = None,
    ) -> JobRecord:
        """Queue a new job.

        Args:
            job_type: What the job runs.
            config_ref: Path to the experiment YAML.
            job_id: Override the generated id. Used by tests.
            now: Creation time; defaults to the current UTC time.

        Returns:
            The queued record.
        """
        created = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
        record = JobRecord(
            job_id=job_id or uuid.uuid4().hex[:16],
            job_type=job_type,
            status=JobStatus.QUEUED,
            config_ref=config_ref,
            created_at=created,
        )
        with self._connect() as connection:
            # `sort_key` is a monotonic counter rather than the timestamp,
            # because two jobs submitted in the same millisecond still have an
            # order and a list that reshuffles them is a list nobody trusts.
            next_key = connection.execute(
                "SELECT COALESCE(MAX(sort_key), 0) + 1 FROM jobs"
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO jobs (job_id, job_type, status, config_ref, created_at, sort_key) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.job_id,
                    record.job_type.value,
                    record.status.value,
                    record.config_ref,
                    _iso(created),
                    next_key,
                ),
            )
        return record

    def get(self, job_id: str) -> JobRecord:
        """Read one job.

        Args:
            job_id: The job id.

        Returns:
            The record.

        Raises:
            JobNotFoundError: If no such job exists.
        """
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            msg = f"No job {job_id!r}."
            raise JobNotFoundError(msg)
        return _record(row)

    def list_jobs(self, *, status: JobStatus | None = None, limit: int = 100) -> list[JobRecord]:
        """List jobs, newest first.

        Args:
            status: Optional status filter.
            limit: Maximum rows returned.

        Returns:
            The records.
        """
        query = "SELECT * FROM jobs"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY sort_key DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_record(row) for row in rows]

    def mark_running(self, job_id: str, *, now: dt.datetime | None = None) -> None:
        """Record that a worker picked the job up.

        Args:
            job_id: The job id.
            now: Start time; defaults to the current UTC time.

        Raises:
            JobNotFoundError: If no such job exists.
        """
        started = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
        self._update(
            job_id,
            "status = ?, started_at = ?",
            (JobStatus.RUNNING.value, _iso(started)),
        )

    def mark_finished(
        self,
        job_id: str,
        status: JobStatus,
        *,
        run_id: str | None = None,
        error: str | None = None,
        now: dt.datetime | None = None,
    ) -> None:
        """Record a terminal state.

        Args:
            job_id: The job id.
            status: A terminal status.
            run_id: The run directory produced, if any.
            error: Traceback text, when the job failed.
            now: Finish time; defaults to the current UTC time.

        Raises:
            ValueError: If `status` is not terminal. A job recorded as finishing
                into `running` would stay in flight forever.
            JobNotFoundError: If no such job exists.
        """
        if not status.is_terminal:
            msg = f"{status.value!r} is not a terminal status; finishing needs one that is."
            raise ValueError(msg)
        finished = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
        self._update(
            job_id,
            "status = ?, finished_at = ?, run_id = COALESCE(?, run_id), error = ?",
            (status.value, _iso(finished), run_id, error),
        )

    def update_progress(
        self, job_id: str, *, phase: str, step: int, total: int, message: str
    ) -> None:
        """Record structured progress.

        Args:
            job_id: The job id.
            phase: Pipeline phase.
            step: Steps completed.
            total: Steps expected, or 0 when unknown.
            message: Human-readable detail.

        Raises:
            JobNotFoundError: If no such job exists.
        """
        self._update(
            job_id,
            "progress_phase = ?, progress_step = ?, progress_total = ?, progress_message = ?",
            (phase, int(step), int(total), message),
        )

    def request_cancel(self, job_id: str) -> JobRecord:
        """Ask a job to stop.

        A process pool cannot kill one of its workers, so this sets a flag the
        worker checks between phases. The job ends as `cancelled` with its run
        directory finalised rather than half-written (§14.2).

        Args:
            job_id: The job id.

        Returns:
            The updated record.

        Raises:
            JobNotFoundError: If no such job exists.
            ValueError: If the job has already finished.
        """
        record = self.get(job_id)
        if record.status.is_terminal:
            msg = f"Job {job_id!r} has already finished as {record.status.value!r}."
            raise ValueError(msg)
        self._update(job_id, "cancel_requested = 1", ())
        return self.get(job_id)

    def cancel_requested(self, job_id: str) -> bool:
        """Whether a cancellation has been asked for.

        This is what the worker polls between phases.

        Args:
            job_id: The job id.

        Returns:
            True if cancellation was requested. A job that has vanished from the
            database counts as cancelled — there is nobody left to report to.
        """
        try:
            return self.get(job_id).cancel_requested
        except JobNotFoundError:
            return True

    def append_log(
        self,
        job_id: str,
        *,
        level: str,
        phase: str,
        message: str,
        now: dt.datetime | None = None,
    ) -> int:
        """Append one line to a job's live log tail.

        Args:
            job_id: The job id.
            level: Log level.
            phase: Pipeline phase.
            message: The line.
            now: Timestamp; defaults to the current UTC time.

        Returns:
            The line's sequence number.
        """
        ts = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
        with self._connect() as connection:
            seq = (
                connection.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM job_logs WHERE job_id = ?", (job_id,)
                ).fetchone()[0]
                or 1
            )
            connection.execute(
                "INSERT INTO job_logs (job_id, seq, ts, level, phase, message) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, seq, _iso(ts), level, phase, message),
            )
            connection.execute(
                "DELETE FROM job_logs WHERE job_id = ? AND seq <= ?",
                (job_id, seq - self.max_log_lines),
            )
        return int(seq)

    def read_logs(self, job_id: str, *, after_seq: int = 0, limit: int = 1_000) -> list[JobLogLine]:
        """Read log lines after a cursor.

        Args:
            job_id: The job id.
            after_seq: Return lines with a sequence number above this. The SSE
                stream passes the last id it sent.
            limit: Maximum lines returned.

        Returns:
            The lines, oldest first.
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT seq, ts, level, phase, message FROM job_logs "
                "WHERE job_id = ? AND seq > ? ORDER BY seq LIMIT ?",
                (job_id, after_seq, limit),
            ).fetchall()
        return [
            JobLogLine(
                seq=int(row["seq"]),
                ts=_parse(row["ts"]) or dt.datetime.now(dt.UTC),
                level=row["level"],
                phase=row["phase"],
                message=row["message"],
            )
            for row in rows
        ]

    def _update(self, job_id: str, assignments: str, params: tuple[Any, ...]) -> None:
        """Apply an UPDATE to one job row.

        Args:
            job_id: The job id.
            assignments: The SET clause.
            params: Its parameters.

        Raises:
            JobNotFoundError: If no row was updated.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {assignments} WHERE job_id = ?",
                (*params, job_id),
            )
        if cursor.rowcount == 0:
            msg = f"No job {job_id!r}."
            raise JobNotFoundError(msg)


def _record(row: sqlite3.Row) -> JobRecord:
    """Build a record from a database row.

    Args:
        row: A row from the jobs table.

    Returns:
        The record.
    """
    return JobRecord(
        job_id=row["job_id"],
        job_type=JobType(row["job_type"]),
        status=JobStatus(row["status"]),
        config_ref=row["config_ref"],
        created_at=_parse(row["created_at"]) or dt.datetime.now(dt.UTC),
        started_at=_parse(row["started_at"]),
        finished_at=_parse(row["finished_at"]),
        run_id=row["run_id"],
        error=row["error"],
        progress=JobProgress(
            phase=row["progress_phase"],
            step=int(row["progress_step"]),
            total=int(row["progress_total"]),
            message=row["progress_message"],
        ),
        cancel_requested=bool(row["cancel_requested"]),
    )
