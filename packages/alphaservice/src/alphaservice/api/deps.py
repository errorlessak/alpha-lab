"""Shared request dependencies and caching helpers.

The caching rules in §14.1 are the interesting part. A run directory is
immutable, so its representations can be cached forever — `Cache-Control:
immutable` plus an `ETag` derived from the run's own config and data hashes.
That is not a performance trick: it is the same claim the artifact store makes,
expressed in HTTP. A resource that could change must never carry these headers,
so the job endpoints deliberately do not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import Depends, Request, Response, status

from alphacore.errors import RunNotFoundError
from alphacore.runtime.artifacts import ArtifactStore
from alphaservice.api.problems import Problem
from alphaservice.api.settings import Settings

if TYPE_CHECKING:
    from alphacore.runtime.artifacts import RunDirectory
    from alphaservice.jobs.runner import JobRunner
    from alphaservice.jobs.store import JobStore

__all__ = [
    "IMMUTABLE_CACHE_CONTROL",
    "RunDir",
    "RunnerDep",
    "SettingsDep",
    "StoreDep",
    "apply_immutable_headers",
    "not_modified",
]

IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"
"""A year, and `immutable`. Run directories are never rewritten (§4, I9), so a
client that has one representation has it forever."""


def get_settings(request: Request) -> Settings:
    """Read the service settings off the application.

    Args:
        request: The request.

    Returns:
        The settings.
    """
    settings = getattr(request.app.state, "settings", None)
    return settings if isinstance(settings, Settings) else Settings.from_env()


def get_artifact_store(settings: Annotated[Settings, Depends(get_settings)]) -> ArtifactStore:
    """Open the artifact store.

    Args:
        settings: Service settings.

    Returns:
        A store rooted at the configured runs directory.
    """
    return ArtifactStore(settings.runs_dir)


def get_job_store(request: Request) -> JobStore:
    """Read the job store off the application.

    Args:
        request: The request.

    Returns:
        The store.

    Raises:
        Problem: With 503 if the application was built without a lifespan, which
            means the job runner does not exist.
    """
    store = getattr(request.app.state, "job_store", None)
    if store is None:
        msg = "The job runner is not available; the service was started without its lifespan."
        raise Problem(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            problem_type="jobs-unavailable",
            title="Job runner unavailable",
            detail=msg,
        )
    return cast("JobStore", store)


def get_job_runner(request: Request) -> JobRunner:
    """Read the job runner off the application.

    Args:
        request: The request.

    Returns:
        The runner.

    Raises:
        Problem: With 503 if the runner does not exist.
    """
    runner = getattr(request.app.state, "job_runner", None)
    if runner is None:
        msg = "The job runner is not available; the service was started without its lifespan."
        raise Problem(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            problem_type="jobs-unavailable",
            title="Job runner unavailable",
            detail=msg,
        )
    return cast("JobRunner", runner)


def get_run(run_id: str, store: Annotated[ArtifactStore, Depends(get_artifact_store)]) -> Any:  # noqa: ANN401
    """Open a run directory for reading, or raise a 404 problem.

    Args:
        run_id: The run id from the path.
        store: The artifact store.

    Returns:
        A read-only `RunDirectory`.

    Raises:
        Problem: With 404 if no such run exists.
    """
    try:
        return store.open_run(run_id)
    except RunNotFoundError as exc:
        raise Problem(
            status_code=status.HTTP_404_NOT_FOUND,
            problem_type="run-not-found",
            title="Run not found",
            detail=(
                f"No run {run_id!r} in {store.root.as_posix()}. "
                "List what is there with GET /api/v1/runs."
            ),
        ) from exc


SettingsDep = Annotated[Settings, Depends(get_settings)]
StoreDep = Annotated[ArtifactStore, Depends(get_artifact_store)]
RunDir = Annotated["RunDirectory", Depends(get_run)]
JobStoreDep = Annotated["JobStore", Depends(get_job_store)]
RunnerDep = Annotated["JobRunner", Depends(get_job_runner)]


def run_etag(run: RunDirectory, *, suffix: str = "") -> str:
    """Build an entity tag for a run resource.

    Derived from the run id and the artifact being served rather than from a
    file hash: the directory is immutable, so the identity of the run *is* the
    identity of its content, and hashing megabytes of parquet to prove it would
    be work with a known answer.

    Args:
        run: The run directory.
        suffix: Distinguishes representations of the same run, e.g. a series
            name and pixel width.

    Returns:
        A strong entity tag, quoted.
    """
    return f'"{run.run_id}{("-" + suffix) if suffix else ""}"'


def apply_immutable_headers(response: Response, etag: str) -> None:
    """Mark a response as an immutable, cacheable representation.

    Args:
        response: The response to annotate.
        etag: The entity tag.
    """
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = IMMUTABLE_CACHE_CONTROL


def not_modified(request: Request, etag: str) -> Response | None:
    """Return a 304 when the client already has this representation.

    Args:
        request: The request, for `If-None-Match`.
        etag: The current entity tag.

    Returns:
        A 304 response, or None if the client should get the body.
    """
    header = request.headers.get("if-none-match")
    if not header:
        return None
    candidates = {value.strip().removeprefix("W/") for value in header.split(",")}
    if etag not in candidates and "*" not in candidates:
        return None
    response = Response(status_code=status.HTTP_304_NOT_MODIFIED)
    apply_immutable_headers(response, etag)
    return response
