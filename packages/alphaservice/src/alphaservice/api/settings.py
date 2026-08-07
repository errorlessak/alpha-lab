"""Where the service reads its state from.

One object rather than module-level globals, so a test can point the app at a
throwaway directory without mutating the environment and hoping nothing else
read it first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

__all__ = ["API_PREFIX", "WEB_DIST_ENV_VAR", "Settings"]

API_PREFIX: Final[str] = "/api/v1"
"""§14.1 mounts everything under this. Versioned from the first commit: the UI
is generated from this document, and an unversioned path makes the first
breaking change everybody's problem at once."""

_DATA_DIR_ENV_VAR: Final[str] = "ALPHA__DATA_DIR"
_RUNS_DIR_ENV_VAR: Final[str] = "ALPHA__RUNS_DIR"
_JOBS_DB_ENV_VAR: Final[str] = "ALPHA__JOBS_DB"

WEB_DIST_ENV_VAR: Final[str] = "ALPHA__WEB_DIST"
"""Public because `alpha serve --web-dist` sets it: with `--reload`, uvicorn
re-imports the application in a fresh process that can only be told anything
through the environment."""


@dataclass(frozen=True)
class Settings:
    """Service configuration.

    Attributes:
        data_dir: Lake root, holding the parquet datasets and the catalog.
        runs_dir: Artifact store root. Read-only to the API — runs are created
            by jobs and never modified afterwards (§4, I9).
        jobs_db: SQLite file for job state and the rolling log table.
        max_workers: Process-pool size for the job runner.
        default_series_width: Pixel width used when a series request omits one.
        web_dist: Built web bundle to serve from this origin, or None to serve
            the API alone. Set in the container; unset in development, where
            Vite serves the app and proxies `/api` to this service.
    """

    data_dir: Path = field(default_factory=lambda: Path("data"))
    runs_dir: Path = field(default_factory=lambda: Path("runs"))
    jobs_db: Path = field(default_factory=lambda: Path("runs") / "jobs.sqlite")
    max_workers: int = 1
    default_series_width: int = 1200
    web_dist: Path | None = None

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the environment the CLI already uses.

        Returns:
            Settings honouring `ALPHA__DATA_DIR`, `ALPHA__RUNS_DIR`,
            `ALPHA__JOBS_DB` and `ALPHA__WEB_DIST`, so `alpha run` and
            `just serve` disagree about nothing.
        """
        runs = Path(os.environ.get(_RUNS_DIR_ENV_VAR) or "runs")
        dist = os.environ.get(WEB_DIST_ENV_VAR)
        return cls(
            data_dir=Path(os.environ.get(_DATA_DIR_ENV_VAR) or "data"),
            runs_dir=runs,
            jobs_db=Path(os.environ.get(_JOBS_DB_ENV_VAR) or runs / "jobs.sqlite"),
            web_dist=Path(dist) if dist else None,
        )
