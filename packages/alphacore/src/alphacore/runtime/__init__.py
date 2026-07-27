"""Run-time scaffolding: seeds, hashing, timing, logging, manifests, artifacts.

Nothing in here knows about markets. It exists so that every number the rest of
the library produces can be traced back to a config, a dataset, and a commit.
"""

from __future__ import annotations

from alphacore.runtime.artifacts import (
    DEFAULT_RUNS_DIR,
    MUTABLE_ARTIFACTS,
    ArtifactStore,
    RunDirectory,
    run_context,
)
from alphacore.runtime.determinism import (
    deterministic_rng,
    hash_seed_is_pinned,
    require_pinned_hash_seed,
    seed_sequence,
    set_global_seed,
)
from alphacore.runtime.hashing import (
    blake2b_hex,
    canonical_json,
    hash_files,
    hash_json,
    hash_path,
    short_hash,
)
from alphacore.runtime.logging import add_run_sink, configure_logging, logger, remove_sink
from alphacore.runtime.manifest import (
    EnvironmentInfo,
    RunManifest,
    RunStatus,
    capture_environment,
    git_revision,
    new_run_id,
    slugify,
)
from alphacore.runtime.timers import PhaseTimings, Timer, timed

__all__ = [
    "DEFAULT_RUNS_DIR",
    "MUTABLE_ARTIFACTS",
    "ArtifactStore",
    "EnvironmentInfo",
    "PhaseTimings",
    "RunDirectory",
    "RunManifest",
    "RunStatus",
    "Timer",
    "add_run_sink",
    "blake2b_hex",
    "canonical_json",
    "capture_environment",
    "configure_logging",
    "deterministic_rng",
    "git_revision",
    "hash_files",
    "hash_json",
    "hash_path",
    "hash_seed_is_pinned",
    "logger",
    "new_run_id",
    "remove_sink",
    "require_pinned_hash_seed",
    "run_context",
    "seed_sequence",
    "set_global_seed",
    "short_hash",
    "slugify",
    "timed",
]
