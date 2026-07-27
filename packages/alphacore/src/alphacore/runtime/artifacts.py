"""The immutable artifact store.

Invariant I9 (§4): `runs/{run_id}/` is written once and never mutated. Any
change means a new run; the API only ever reads.

Enforcement is mechanical, not documentary. `RunDirectory.write_*` raises
`ArtifactExistsError` on a path that already exists. Two files are exempt and
the exemption is explicit rather than incidental:

* `manifest.json` — must transition `running` to a terminal status
* `logs.jsonl` — appended to throughout the run

Everything else — tearsheet, parquet series, model files — is write-once.
"""

from __future__ import annotations

import datetime as dt
import os
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from alphacore.errors import ArtifactExistsError, RunNotFoundError
from alphacore.runtime.hashing import canonical_json
from alphacore.runtime.logging import LOG_FILENAME, add_run_sink, logger, remove_sink
from alphacore.runtime.manifest import MANIFEST_FILENAME, RunManifest, RunStatus
from alphacore.runtime.timers import PhaseTimings

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "DEFAULT_RUNS_DIR",
    "MUTABLE_ARTIFACTS",
    "ArtifactStore",
    "RunDirectory",
    "run_context",
]

DEFAULT_RUNS_DIR: Final[Path] = Path("runs")

MUTABLE_ARTIFACTS: Final[frozenset[str]] = frozenset({MANIFEST_FILENAME, LOG_FILENAME})
"""The only files in a run directory that may be written more than once. Both
are run *control* records rather than results; no metric, series, or model is
ever rewritten."""

_RUNS_DIR_ENV_VAR: Final[str] = "ALPHA__RUNS_DIR"


class RunDirectory:
    """A handle on one run's directory.

    Attributes:
        run_id: The directory name.
        path: Absolute path to the directory.
    """

    def __init__(self, path: Path, *, writable: bool) -> None:
        """Bind a handle to an existing directory.

        Args:
            path: The run directory.
            writable: Whether writes are permitted. Handles returned by
                `ArtifactStore.open_run` are read-only, which is what the API
                uses.
        """
        self.path = path.resolve()
        self.run_id = self.path.name
        self._writable = writable

    def __repr__(self) -> str:
        """Return a debug representation naming the run and its mode."""
        mode = "rw" if self._writable else "ro"
        return f"RunDirectory({self.run_id!r}, {mode})"

    def artifact_path(self, name: str) -> Path:
        """Resolve an artifact name to a path inside this run.

        Args:
            name: Relative artifact name, e.g. `"tearsheet.json"` or
                `"models/lightgbm/model.txt"`.

        Returns:
            The absolute path.

        Raises:
            ArtifactExistsError: If the name escapes the run directory.
        """
        candidate = (self.path / name).resolve()
        if not candidate.is_relative_to(self.path):
            msg = f"Artifact name {name!r} escapes run directory {self.path}."
            raise ArtifactExistsError(msg)
        return candidate

    def exists(self, name: str) -> bool:
        """Report whether an artifact has been written.

        Args:
            name: Relative artifact name.

        Returns:
            True if the file exists.
        """
        return self.artifact_path(name).is_file()

    def _check_writable(self, name: str) -> Path:
        """Validate a write and return the destination path.

        Args:
            name: Relative artifact name.

        Returns:
            The absolute destination path, with parent directories created.

        Raises:
            ArtifactExistsError: If this handle is read-only, or the artifact
                already exists and is not on the mutable allow-list.
        """
        if not self._writable:
            msg = f"Run {self.run_id!r} was opened read-only; artifacts are immutable (§4, I9)."
            raise ArtifactExistsError(msg)
        path = self.artifact_path(name)
        if path.exists() and name not in MUTABLE_ARTIFACTS:
            msg = (
                f"Artifact {name!r} already exists in run {self.run_id!r}. Run "
                "directories are immutable (§4, I9) — produce a new run instead of "
                "editing this one."
            )
            raise ArtifactExistsError(msg)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_bytes(self, name: str, data: bytes) -> Path:
        """Write raw bytes as an artifact.

        Args:
            name: Relative artifact name.
            data: Bytes to write.

        Returns:
            The path written.

        Raises:
            ArtifactExistsError: If the artifact already exists, or the handle
                is read-only.
        """
        path = self._check_writable(name)
        path.write_bytes(data)
        return path

    def write_text(self, name: str, text: str) -> Path:
        """Write UTF-8 text as an artifact, with LF line endings.

        Line-ending normalisation matters: a byte-for-byte determinism check
        (§12.4) would otherwise fail purely from running on a different
        operating system.

        Args:
            name: Relative artifact name.
            text: Text to write.

        Returns:
            The path written.

        Raises:
            ArtifactExistsError: If the artifact already exists, or the handle
                is read-only.
        """
        return self.write_bytes(name, text.encode("utf-8"))

    def write_json(self, name: str, value: Any) -> Path:  # noqa: ANN401
        """Write a value as canonical JSON.

        Canonical form (sorted keys, no insignificant whitespace) is what makes
        the determinism test meaningful — dict insertion order can never change
        the bytes on disk.

        Args:
            name: Relative artifact name.
            value: Any canonically serialisable value.

        Returns:
            The path written.

        Raises:
            ArtifactExistsError: If the artifact already exists, or the handle
                is read-only.
            SerialisationError: If the value cannot be canonically serialised.
        """
        return self.write_bytes(name, canonical_json(value) + b"\n")

    def read_bytes(self, name: str) -> bytes:
        """Read an artifact's raw bytes.

        Args:
            name: Relative artifact name.

        Returns:
            The file's contents.

        Raises:
            RunNotFoundError: If the artifact does not exist.
        """
        path = self.artifact_path(name)
        if not path.is_file():
            msg = f"Run {self.run_id!r} has no artifact {name!r}."
            raise RunNotFoundError(msg)
        return path.read_bytes()

    def read_json(self, name: str) -> Any:  # noqa: ANN401
        """Read and parse a JSON artifact.

        Args:
            name: Relative artifact name.

        Returns:
            The parsed value.

        Raises:
            RunNotFoundError: If the artifact does not exist.
        """
        import json  # noqa: PLC0415 - only needed on the read path

        return json.loads(self.read_bytes(name).decode("utf-8"))

    def write_manifest(self, manifest: RunManifest) -> Path:
        """Write or update `manifest.json`.

        Args:
            manifest: The manifest to persist.

        Returns:
            The path written.

        Raises:
            ArtifactExistsError: If the handle is read-only.
        """
        return self.write_json(MANIFEST_FILENAME, manifest.model_dump(mode="json"))

    def read_manifest(self) -> RunManifest:
        """Read and validate `manifest.json`.

        Returns:
            The manifest.

        Raises:
            RunNotFoundError: If the run has no manifest.
            pydantic.ValidationError: If the manifest does not match the schema.
        """
        return RunManifest.model_validate(self.read_json(MANIFEST_FILENAME))


class ArtifactStore:
    """The `runs/` directory: an append-only collection of immutable runs."""

    def __init__(self, root: Path | str | None = None) -> None:
        """Bind a store to a root directory.

        Args:
            root: The store root. Falls back to `$ALPHA__RUNS_DIR`, then to
                `./runs`.
        """
        if root is None:
            root = os.environ.get(_RUNS_DIR_ENV_VAR) or DEFAULT_RUNS_DIR
        self.root = Path(root).resolve()

    def __repr__(self) -> str:
        """Return a debug representation naming the store root."""
        return f"ArtifactStore({self.root.as_posix()!r})"

    def create_run(self, run_id: str) -> RunDirectory:
        """Create a new, empty run directory.

        Args:
            run_id: The directory name, normally from `manifest.new_run_id`.

        Returns:
            A writable handle.

        Raises:
            ArtifactExistsError: If a directory with this id already exists.
        """
        path = self.root / run_id
        if path.exists():
            msg = f"Run {run_id!r} already exists at {path}. Run ids are never reused (§4, I9)."
            raise ArtifactExistsError(msg)
        path.mkdir(parents=True)
        return RunDirectory(path, writable=True)

    def open_run(self, run_id: str) -> RunDirectory:
        """Open an existing run for reading.

        Args:
            run_id: The run id.

        Returns:
            A read-only handle.

        Raises:
            RunNotFoundError: If no such run directory exists.
        """
        path = self.root / run_id
        if not path.is_dir():
            msg = f"No run {run_id!r} in {self.root}."
            raise RunNotFoundError(msg)
        return RunDirectory(path, writable=False)

    def list_runs(self) -> list[str]:
        """List run ids, newest first.

        Run ids begin with a UTC timestamp, so reverse lexicographic order is
        reverse chronological order.

        Returns:
            Run ids, or an empty list if the store root does not exist yet.
        """
        if not self.root.is_dir():
            return []
        return sorted(
            (p.name for p in self.root.iterdir() if p.is_dir()),
            reverse=True,
        )


@contextmanager
def run_context(  # noqa: PLR0913 - every parameter is a distinct provenance field
    *,
    name: str,
    seed: int,
    config: Any,  # noqa: ANN401 - any canonically serialisable config tree
    store: ArtifactStore | None = None,
    run_id: str | None = None,
    data_hash: str | None = None,
    log_level: str = "DEBUG",
    now: dt.datetime | None = None,
) -> Iterator[tuple[RunDirectory, PhaseTimings]]:
    """Open a run: create the directory, stamp a manifest, attach the log sink.

    On clean exit the manifest is finalised as `completed`. On an exception the
    manifest is finalised as `failed` with the traceback, and the exception is
    re-raised. A run directory is therefore never left claiming to be valid
    when it is not (§14.2).

    Args:
        name: Experiment name from `meta.name`.
        seed: The run seed.
        config: The fully resolved config tree, hashed into `config_hash`.
        store: Artifact store. Defaults to a store at `$ALPHA__RUNS_DIR`.
        run_id: Override the generated run id. Used by tests.
        data_hash: Content hash of the input dataset, if already known.
        log_level: Minimum level for the run's JSON-lines log.
        now: Creation time; defaults to the current UTC time.

    Yields:
        The writable run directory and a `PhaseTimings` recorder. Times
        recorded through the latter land in the final manifest automatically.

    Raises:
        ArtifactExistsError: If the run id is already taken.
        SerialisationError: If the config cannot be canonically serialised.
    """
    artifact_store = store or ArtifactStore()
    manifest = RunManifest.start(
        name=name,
        seed=seed,
        config=config,
        run_id=run_id,
        data_hash=data_hash,
        now=now,
    )
    run = artifact_store.create_run(manifest.run_id)
    run.write_manifest(manifest)

    timings = PhaseTimings()
    sink_id = add_run_sink(run.path, level=log_level)
    logger.bind(phase="run").info(
        "run started run_id={} config_hash={} seed={}",
        manifest.run_id,
        manifest.config_hash,
        manifest.seed,
    )
    try:
        yield run, timings
    except BaseException as exc:
        status = RunStatus.CANCELLED if isinstance(exc, KeyboardInterrupt) else RunStatus.FAILED
        logger.bind(phase="run").error("run {} — {}", status.value, exc)
        run.write_manifest(
            manifest.finish(
                status,
                error="".join(traceback.format_exception(exc)),
                phase_durations_s=timings.durations_s,
                now=dt.datetime.now(dt.UTC),
            )
        )
        raise
    else:
        final = manifest.finish(
            RunStatus.COMPLETED,
            phase_durations_s=timings.durations_s,
            now=dt.datetime.now(dt.UTC),
        )
        logger.bind(phase="run").success("run completed in {:.2f}s", final.duration_s or 0.0)
        run.write_manifest(final)
    finally:
        remove_sink(sink_id)
