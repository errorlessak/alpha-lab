"""RFC 9457 problem documents (§14.1).

Every error this API returns is `application/problem+json` with `type`, `title`,
`status` and `detail`. The reason is §13.5's rule about error states, applied one
layer down: an error must name the cause and the fix, and "Internal Server Error"
names neither.

`type` is a stable URI fragment a client can branch on — `about:blank` is
permitted by the RFC and useless to a UI that wants to say something specific
about a run with no backtest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from alphacore.errors import (
    AlphaError,
    ArtifactExistsError,
    ConfigError,
    DataError,
    EvaluationError,
    RunNotFoundError,
)
from alphaservice.errors import JobNotFoundError, JobStateError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastapi import FastAPI

__all__ = [
    "PROBLEM_CONTENT_TYPE",
    "Problem",
    "ProblemDocument",
    "install_error_handlers",
    "problem_response",
    "problem_responses",
]

PROBLEM_CONTENT_TYPE: Final[str] = "application/problem+json"

_TYPE_BASE: Final[str] = "https://alpha-lab.invalid/problems/"
"""Problem type URIs. A reserved-by-RFC-2606 host: these identify a kind of
error, they are not addresses anybody should fetch."""


class ProblemDocument(BaseModel):
    """An RFC 9457 problem, as it appears in the OpenAPI document.

    Declared as a model so the generated TypeScript client has a type for the
    error branch. A client that types errors as `unknown` pushes every caller
    into `catch (e: any)`.
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(description="Stable URI identifying the kind of problem.")
    title: str = Field(description="Short summary of the kind of problem.")
    status: int = Field(description="HTTP status code, repeated in the body per RFC 9457.")
    detail: str = Field(description="What went wrong this time, and where possible what to do.")
    instance: str | None = Field(default=None, description="The request path.")


def problem_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    """Document the problem responses a route can return.

    Every error this API produces is a problem document, so a route that can
    404 must say so in the schema — otherwise the generated client has no type
    for the failure and the contract tests flag an undocumented status.

    Args:
        *status_codes: Statuses the route can answer with.

    Returns:
        A `responses` mapping for a FastAPI route decorator.
    """
    return {
        code: {
            "model": ProblemDocument,
            "description": _TITLES.get(code, "Request failed"),
            "content": {PROBLEM_CONTENT_TYPE: {"schema": ProblemDocument.model_json_schema()}},
        }
        for code in status_codes
    }


class Problem(HTTPException):
    """An error carrying everything a problem document needs.

    Attributes:
        problem_type: Slug appended to the problem type base URI.
        title: Short, human-readable summary of the *kind* of error.
        detail: What went wrong this time, and where possible what to do.
        extra: Additional members merged into the document.
    """

    def __init__(
        self,
        *,
        status_code: int,
        problem_type: str,
        title: str,
        detail: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Build a problem.

        Args:
            status_code: HTTP status.
            problem_type: Slug for the type URI.
            title: Summary of the kind of error.
            detail: What went wrong this time.
            extra: Additional members.
        """
        super().__init__(status_code=status_code, detail=detail)
        self.problem_type = problem_type
        self.title = title
        self.extra = extra or {}


def problem_response(
    *,
    status_code: int,
    problem_type: str,
    title: str,
    detail: str,
    instance: str | None = None,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    """Render a problem document.

    Args:
        status_code: HTTP status.
        problem_type: Slug for the type URI.
        title: Summary of the kind of error.
        detail: What went wrong this time.
        instance: The request path this happened on.
        extra: Additional members.

    Returns:
        A `application/problem+json` response.
    """
    body: dict[str, Any] = {
        "type": f"{_TYPE_BASE}{problem_type}",
        "title": title,
        "status": status_code,
        "detail": detail,
    }
    if instance:
        body["instance"] = instance
    body.update(extra or {})
    return JSONResponse(status_code=status_code, content=body, media_type=PROBLEM_CONTENT_TYPE)


_ALPHA_ERROR_MAP: Final[tuple[tuple[type[AlphaError], int, str, str], ...]] = (
    (RunNotFoundError, status.HTTP_404_NOT_FOUND, "run-not-found", "Run not found"),
    (JobNotFoundError, status.HTTP_404_NOT_FOUND, "job-not-found", "Job not found"),
    (JobStateError, status.HTTP_409_CONFLICT, "job-state", "Job is in the wrong state"),
    (ArtifactExistsError, status.HTTP_409_CONFLICT, "artifact-exists", "Artifact already exists"),
    (ConfigError, status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid-config", "Invalid configuration"),
    (
        EvaluationError,
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "malformed-artifact",
        "Artifact does not match its schema",
    ),
    (DataError, status.HTTP_422_UNPROCESSABLE_CONTENT, "data-error", "Data problem"),
)
"""Ordered most specific first: `RunNotFoundError` is an `ArtifactError`, and a
missing run is a 404 rather than a 409."""


def install_error_handlers(app: FastAPI) -> None:
    """Register the handlers that make every error a problem document.

    Args:
        app: The application.
    """

    @app.exception_handler(Problem)
    async def _problem(request: Request, exc: Problem) -> JSONResponse:
        return problem_response(
            status_code=exc.status_code,
            problem_type=exc.problem_type,
            title=exc.title,
            detail=str(exc.detail),
            instance=request.url.path,
            extra=exc.extra,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # The field-level errors are kept as an extra member: RFC 9457 allows
        # them and a form cannot highlight the offending input without them.
        return problem_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            problem_type="invalid-request",
            title="Request parameters are invalid",
            detail=_summarise(exc),
            instance=request.url.path,
            extra={"errors": _serialisable(exc.errors())},
        )

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException) -> JSONResponse:
        return problem_response(
            status_code=exc.status_code,
            problem_type="http-error",
            title=_TITLES.get(exc.status_code, "Request failed"),
            detail=str(exc.detail),
            instance=request.url.path,
        )

    @app.exception_handler(AlphaError)
    async def _alpha(request: Request, exc: AlphaError) -> JSONResponse:
        for kind, code, slug, title in _ALPHA_ERROR_MAP:
            if isinstance(exc, kind):
                return problem_response(
                    status_code=code,
                    problem_type=slug,
                    title=title,
                    detail=str(exc),
                    instance=request.url.path,
                )
        # An alphacore error nobody mapped is a bug in this table, not in the
        # request. It is reported as a server fault with the library's own
        # message, which is always specific.
        return problem_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            problem_type="internal-error",
            title="The service failed to complete the request",
            detail=str(exc),
            instance=request.url.path,
        )


_TITLES: Final[dict[int, str]] = {
    status.HTTP_400_BAD_REQUEST: "Bad request",
    status.HTTP_404_NOT_FOUND: "Not found",
    status.HTTP_409_CONFLICT: "Conflict",
    status.HTTP_422_UNPROCESSABLE_CONTENT: "Unprocessable request",
    status.HTTP_500_INTERNAL_SERVER_ERROR: "Internal error",
}


def _summarise(exc: RequestValidationError) -> str:
    """Turn pydantic's error list into one readable sentence.

    Args:
        exc: The validation error.

    Returns:
        A summary naming the offending fields.
    """
    parts = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()) if item != "query")
        parts.append(f"{location or 'request'}: {error.get('msg', 'is invalid')}")
    return "; ".join(parts) or "The request could not be validated."


def _serialisable(errors: Sequence[Any]) -> list[dict[str, Any]]:
    """Strip pydantic error entries down to JSON-safe members.

    Args:
        errors: Raw pydantic errors.

    Returns:
        Entries carrying only `loc`, `msg` and `type`. The `ctx` member can hold
        arbitrary exception objects, which do not survive serialisation.
    """
    return [
        {
            "loc": [str(item) for item in error.get("loc", ())],
            "msg": str(error.get("msg", "")),
            "type": str(error.get("type", "")),
        }
        for error in errors
    ]
