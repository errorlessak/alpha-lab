"""The run manifest: what was run, against what data, with which code.

`runs/{run_id}/manifest.json` is the provenance record for a run. It is written
once when the run starts and finalised once when it ends; nothing else in the
run directory is ever rewritten (§4, I9).

A manifest answers exactly one question: *could I reproduce this?* If the
config hash, data hash, code version and seed all match, invariant I8 says the
artifacts must be byte-identical. If they do not match, the manifest says
precisely which one moved.
"""

from __future__ import annotations

import datetime as dt
import platform
import re
import socket
import subprocess
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field

from alphacore.runtime.hashing import hash_json, short_hash

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "EnvironmentInfo",
    "RunManifest",
    "RunStatus",
    "capture_environment",
    "git_revision",
    "new_run_id",
    "slugify",
]

MANIFEST_FILENAME: Final[str] = "manifest.json"
MANIFEST_SCHEMA_VERSION: Final[str] = "1.0"

RUN_ID_TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%dT%H-%M-%SZ"
RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z_[a-z0-9][a-z0-9-]*_[0-9a-f]{4,}$"
)

_TRACKED_PACKAGES: Final[tuple[str, ...]] = (
    "alphacore",
    "alphaservice",
    "catboost",
    "duckdb",
    "lightgbm",
    "numpy",
    "pandas",
    "pandera",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "shap",
    "torch",
    "xgboost",
)
"""Packages whose versions materially change results. A version bump here is a
legitimate reason for two runs with the same config hash to differ, so the
manifest records them rather than leaving the difference unexplained."""

_GIT_TIMEOUT_S: Final[float] = 5.0
_SLUG_INVALID = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Reduce a config name to a filesystem- and URL-safe slug.

    Args:
        name: A human-written experiment name, e.g. `"SPY daily / triple barrier"`.

    Returns:
        Lowercase, hyphen-separated slug, e.g. `"spy-daily-triple-barrier"`.
        Returns `"run"` if nothing usable survives.
    """
    slug = _SLUG_INVALID.sub("-", name.strip().lower()).strip("-")
    return slug or "run"


def new_run_id(name: str, config_hash: str, *, now: dt.datetime | None = None) -> str:
    """Build a run id of the form `2026-07-27T14-22-01Z_spy-daily-tb_a3f91c`.

    The timestamp makes runs sort chronologically in a directory listing; the
    slug makes them recognisable; the config hash prefix makes it obvious at a
    glance when two runs share a configuration.

    Args:
        name: Experiment name from `meta.name`.
        config_hash: Full hex `config_hash`.
        now: Creation time; defaults to the current UTC time. Injectable so
            tests do not depend on the clock.

    Returns:
        The run id.

    Raises:
        SerialisationError: If `config_hash` is too short to truncate.
    """
    moment = now or dt.datetime.now(dt.UTC)
    stamp = moment.astimezone(dt.UTC).strftime(RUN_ID_TIMESTAMP_FORMAT)
    return f"{stamp}_{slugify(name)}_{short_hash(config_hash)}"


def git_revision(repo: Path | None = None) -> str | None:
    """Read the current git commit, marking a dirty working tree.

    Args:
        repo: Repository root. Defaults to the current working directory.

    Returns:
        The short SHA, suffixed `-dirty` when there are uncommitted changes, or
        None when git is unavailable or this is not a repository. A missing
        revision is recorded honestly rather than faked.
    """
    cwd = repo or Path.cwd()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def _package_versions() -> dict[str, str]:
    """Collect installed versions of the packages that can change results.

    Returns:
        Mapping of package name to version, sorted by name, omitting packages
        that are not installed.
    """
    found: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return dict(sorted(found.items()))


class EnvironmentInfo(BaseModel):
    """The machine and interpreter a run executed on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    python: str = Field(description="Interpreter version, e.g. '3.12.13'.")
    implementation: str = Field(description="Interpreter implementation, e.g. 'CPython'.")
    platform: str = Field(description="OS and architecture summary.")
    host: str = Field(description="Hostname of the machine that produced the run.")
    packages: dict[str, str] = Field(
        default_factory=dict,
        description="Versions of result-affecting packages, sorted by name.",
    )


def capture_environment() -> EnvironmentInfo:
    """Snapshot the current interpreter and machine.

    Returns:
        An `EnvironmentInfo` describing this process.
    """
    return EnvironmentInfo(
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        platform=platform.platform(),
        host=socket.gethostname(),
        packages=_package_versions(),
    )


class RunStatus(StrEnum):
    """Lifecycle state of a run directory.

    A directory is only ever `completed` if every phase finished. A crashed job
    is marked `failed` with its traceback rather than left half-written and
    presented as valid (§14.2).
    """

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunManifest(BaseModel):
    """Provenance record for a single run.

    Attributes:
        schema_version: Version of this manifest's own schema.
        run_id: Directory name under `runs/`.
        name: Experiment name from `meta.name`.
        status: Lifecycle state.
        created_at: UTC time the run directory was created.
        finished_at: UTC time the run reached a terminal state.
        duration_s: Wall-clock seconds from creation to termination.
        seed: The run seed.
        config_hash: blake2b over the canonical JSON of the resolved config.
        data_hash: Content hash of the input dataset, once ingestion has run.
        code_version: Short git SHA, `-dirty` suffixed when uncommitted.
        environment: Interpreter and machine snapshot.
        phase_durations_s: Wall-clock seconds per pipeline phase.
        error: Traceback text when `status` is `failed`.
        notes: Free-form flags, e.g. `{"data_quality": "FAIL"}`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = MANIFEST_SCHEMA_VERSION
    run_id: str
    name: str
    status: RunStatus = RunStatus.RUNNING
    created_at: dt.datetime
    finished_at: dt.datetime | None = None
    duration_s: float | None = None
    seed: int
    config_hash: str
    data_hash: str | None = None
    code_version: str | None = None
    environment: EnvironmentInfo
    phase_durations_s: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    notes: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def start(
        cls,
        *,
        name: str,
        seed: int,
        config: Any,  # noqa: ANN401 - any canonically serialisable config tree
        run_id: str | None = None,
        data_hash: str | None = None,
        now: dt.datetime | None = None,
        repo: Path | None = None,
    ) -> Self:
        """Create a manifest for a run that is about to begin.

        Args:
            name: Experiment name from `meta.name`.
            seed: The run seed.
            config: The fully resolved config tree. Hashed, not stored — the
                config itself is written to `config.resolved.yaml` beside it.
            run_id: Override the generated run id. Used by tests.
            data_hash: Content hash of the input dataset if already known.
            now: Creation time; defaults to the current UTC time.
            repo: Repository root for the git revision lookup.

        Returns:
            A manifest with `status=running`.

        Raises:
            SerialisationError: If the config cannot be canonically serialised.
            DeterminismError: If the seed is out of range.
        """
        from alphacore.runtime.determinism import _validate_seed  # noqa: PLC0415

        _validate_seed(seed)
        created = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
        config_hash = hash_json(config)
        return cls(
            run_id=run_id or new_run_id(name, config_hash, now=created),
            name=name,
            created_at=created,
            seed=seed,
            config_hash=config_hash,
            data_hash=data_hash,
            code_version=git_revision(repo),
            environment=capture_environment(),
        )

    def finish(
        self,
        status: RunStatus,
        *,
        now: dt.datetime | None = None,
        error: str | None = None,
        phase_durations_s: dict[str, float] | None = None,
    ) -> RunManifest:
        """Return a terminal copy of this manifest.

        Returns a copy rather than mutating in place, so the transition is
        explicit at the call site and an in-flight manifest can never be
        confused with a finished one.

        Args:
            status: Terminal status to record.
            now: Finish time; defaults to the current UTC time.
            error: Traceback text, required when `status` is `failed`.
            phase_durations_s: Per-phase wall-clock seconds.

        Returns:
            A new manifest in a terminal state.

        Raises:
            ValueError: If `status` is not terminal.
        """
        if status is RunStatus.RUNNING:
            msg = "finish() requires a terminal status, not RunStatus.RUNNING."
            raise ValueError(msg)
        finished = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
        return self.model_copy(
            update={
                "status": status,
                "finished_at": finished,
                "duration_s": (finished - self.created_at).total_seconds(),
                "error": error,
                "phase_durations_s": phase_durations_s or self.phase_durations_s,
            }
        )
