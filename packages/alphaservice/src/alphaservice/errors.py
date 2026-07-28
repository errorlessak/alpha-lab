"""Typed errors for the service layer.

Everything here derives from `alphacore.errors.AlphaError`, so a handler can
catch one type and still tell "the config is wrong" from "no such job" without
string matching (§18). The service adds its own because the failures it has are
not the library's: a job id that does not exist is not a data problem, a
research problem, or a bug in a metric.
"""

from __future__ import annotations

from alphacore.errors import AlphaError

__all__ = ["JobNotFoundError", "JobStateError", "ServiceError"]


class ServiceError(AlphaError):
    """The service could not satisfy a request."""


class JobNotFoundError(ServiceError):
    """No job exists with the requested id."""


class JobStateError(ServiceError):
    """A job was asked to do something its current state does not allow."""
