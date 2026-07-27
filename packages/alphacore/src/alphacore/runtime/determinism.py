"""Seeding and determinism guards.

Invariant I8 (§4): given `(config_hash, data_hash, seed)`, every artifact is
byte-identical across runs. That requires three things, and this module owns all
three:

1. every RNG in the process is seeded from one place
2. `PYTHONHASHSEED` is pinned, so `hash()` of a string is stable across
   processes and set iteration cannot reorder output
3. code that needs randomness takes an explicit `Generator` rather than reaching
   for the global `numpy.random` state

Point 3 is the one that actually bites in practice. `set_global_seed` exists to
tame third-party libraries; first-party code should call `deterministic_rng`.
"""

from __future__ import annotations

import os
import random
import sys
from typing import TYPE_CHECKING, Final

import numpy as np

from alphacore.errors import DeterminismError

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "HASH_SEED_ENV_VAR",
    "REQUIRED_HASH_SEED",
    "deterministic_rng",
    "hash_seed_is_pinned",
    "require_pinned_hash_seed",
    "seed_sequence",
    "set_global_seed",
]

HASH_SEED_ENV_VAR: Final[str] = "PYTHONHASHSEED"
REQUIRED_HASH_SEED: Final[str] = "0"

_MAX_SEED: Final[int] = 2**32 - 1


def _validate_seed(seed: int) -> int:
    """Check a seed is a plain non-negative 32-bit integer.

    Args:
        seed: Candidate seed.

    Returns:
        The seed, unchanged.

    Raises:
        DeterminismError: If the seed is negative or too large to be accepted by
            every downstream RNG.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        msg = f"Seed must be an int, got {type(seed).__name__!r}."
        raise DeterminismError(msg)
    if seed < 0 or seed > _MAX_SEED:
        msg = f"Seed must be in [0, {_MAX_SEED}], got {seed}."
        raise DeterminismError(msg)
    return seed


def hash_seed_is_pinned() -> bool:
    """Report whether string hashing is deterministic in this process.

    Returns:
        True if `PYTHONHASHSEED=0` was set before the interpreter started.
    """
    return os.environ.get(HASH_SEED_ENV_VAR) == REQUIRED_HASH_SEED


def require_pinned_hash_seed() -> None:
    """Fail unless string hash randomisation is disabled.

    `PYTHONHASHSEED` is read by CPython at startup, so it cannot be fixed from
    inside a running process. The only honest options are to fail loudly or to
    stop claiming determinism.

    Raises:
        DeterminismError: If `PYTHONHASHSEED` is not `"0"`.
    """
    if not hash_seed_is_pinned():
        actual = os.environ.get(HASH_SEED_ENV_VAR, "<unset>")
        msg = (
            f"{HASH_SEED_ENV_VAR}={actual!r} but determinism (§4, I8) requires "
            f"{HASH_SEED_ENV_VAR}={REQUIRED_HASH_SEED!r}. It must be set before the "
            "interpreter starts — export it in your shell, or run via `just`, which "
            "sets it for you."
        )
        raise DeterminismError(msg)


def set_global_seed(seed: int, *, strict_hash_seed: bool = False) -> None:
    """Seed every global RNG this process can reach.

    Covers `random`, `numpy`'s legacy global state, and PyTorch if — and only
    if — it is already imported. Torch is never imported here: `import
    alphacore` must work with torch uninstalled (§10.3).

    Args:
        seed: The run seed, from `meta.seed` in the config.
        strict_hash_seed: If True, also require `PYTHONHASHSEED=0`.

    Raises:
        DeterminismError: If the seed is invalid, or `strict_hash_seed` is set
            and `PYTHONHASHSEED` is not pinned.
    """
    _validate_seed(seed)
    if strict_hash_seed:
        require_pinned_hash_seed()

    random.seed(seed)
    # Legacy global state, seeded deliberately: third-party estimators still
    # reach for it. First-party code uses deterministic_rng instead.
    np.random.seed(seed)

    torch = sys.modules.get("torch")
    if torch is not None:  # pragma: no cover - exercised only in the [dl] extra
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def deterministic_rng(seed: int, *stream: object) -> np.random.Generator:
    """Build an independent, reproducible generator for one logical stream.

    Deriving a child stream from `(seed, *stream)` means two callers never
    silently share state, and adding a new caller cannot perturb an existing
    one's draws. Pass a stable label, e.g. `deterministic_rng(seed, "cpcv",
    fold)`.

    Args:
        seed: The run seed.
        *stream: Stable labels identifying this stream. Hashed canonically, so
            strings and ints are both fine.

    Returns:
        A `numpy.random.Generator` backed by PCG64.

    Raises:
        DeterminismError: If the seed is invalid.
        SerialisationError: If a stream label cannot be canonically serialised.
    """
    _validate_seed(seed)
    if not stream:
        return np.random.default_rng(seed)

    from alphacore.runtime.hashing import blake2b_hex, canonical_json  # noqa: PLC0415

    # Hash first, then truncate. Truncating the canonical JSON directly would
    # collide on any two labels sharing a prefix — ("cpcv", 0) and ("cpcv", 1)
    # agree in their first eight bytes.
    entropy = int(blake2b_hex(canonical_json(list(stream)), digest_size=8), 16)
    return np.random.default_rng(np.random.SeedSequence([seed, entropy]))


def seed_sequence(seed: int, n: int) -> Iterator[int]:
    """Yield `n` reproducible child seeds derived from one parent seed.

    Used where a third-party API insists on an int seed per unit of work — an
    Optuna trial, a bootstrap replicate — and cannot accept a `Generator`.

    Args:
        seed: The parent seed.
        n: How many child seeds to produce.

    Yields:
        Non-negative 32-bit integers.

    Raises:
        DeterminismError: If the seed is invalid or `n` is negative.
    """
    _validate_seed(seed)
    if n < 0:
        msg = f"Cannot produce {n} seeds."
        raise DeterminismError(msg)
    children = np.random.SeedSequence(seed).generate_state(n, dtype=np.uint32)
    for child in children:
        yield int(child)
