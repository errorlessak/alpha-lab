"""Health and the config schema (§14.1)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

import alphacore
import alphaservice
from alphacore.config.schema import AlphaConfig
from alphacore.runtime.manifest import git_revision
from alphaservice.api.deps import StoreDep

__all__ = ["router"]

router = APIRouter(tags=["system"])


class Health(BaseModel):
    """What the service is and what it is reading."""

    status: str = Field(description="`ok` when the service can read its artifact store.")
    alphacore_version: str
    alphaservice_version: str
    git_sha: str | None = Field(description="Code revision, `-dirty` suffixed when uncommitted.")
    artifact_store: str = Field(description="Absolute path to the runs directory.")
    n_runs: int = Field(ge=0, description="Runs currently on disk.")


@router.get("/health", operation_id="getHealth", summary="Service health and provenance")
def health(store: StoreDep) -> Health:
    """Report the service version, revision, and artifact store.

    Args:
        store: The artifact store.

    Returns:
        The health document. The git revision is included because a number
        served by an unknown build is a number nobody can reproduce.
    """
    return Health(
        status="ok",
        alphacore_version=alphacore.__version__,
        alphaservice_version=alphaservice.__version__,
        git_sha=git_revision(),
        artifact_store=store.root.as_posix(),
        n_runs=len(store.list_runs()),
    )


@router.get(
    "/system/config/schema",
    operation_id="getConfigSchema",
    summary="JSON schema for an experiment configuration",
)
def config_schema() -> dict[str, Any]:
    """Serve the JSON schema for `AlphaConfig`.

    Returns:
        The schema pydantic generates from the config models themselves, so the
        UI cannot render a form that disagrees with what the loader accepts.
    """
    return AlphaConfig.model_json_schema()
