"""Canonical serialisation and content hashing.

Reproducibility (§4, I8) is proved by hashes, not by trust. Two things must be
hashable to a stable digest:

* a resolved configuration tree, giving `config_hash`
* the bytes of an ingested dataset, giving `data_hash` / `content_hash`

Both use blake2b. Canonical JSON is the intermediate representation for
structured values: keys sorted, no insignificant whitespace, ASCII-escaped, and
`NaN`/`Infinity` rejected outright.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence, Set
from enum import Enum
from pathlib import Path
from typing import Any, Final

from alphacore.errors import SerialisationError

__all__ = [
    "DEFAULT_DIGEST_SIZE",
    "SHORT_HASH_LENGTH",
    "blake2b_hex",
    "canonical_json",
    "hash_files",
    "hash_json",
    "hash_path",
    "short_hash",
]

DEFAULT_DIGEST_SIZE: Final[int] = 16
"""Digest size in bytes. 16 gives a 32-char hex string: ample for collision
avoidance across a local artifact store, short enough to read in a run id."""

SHORT_HASH_LENGTH: Final[int] = 6
"""Characters of a digest used in human-facing identifiers, e.g. `a3f91c`."""

_FILE_CHUNK_BYTES: Final[int] = 1 << 20


def _to_jsonable(value: Any) -> Any:  # noqa: ANN401, PLR0911 - a type dispatch chain
    """Convert a value into a JSON-representable form, deterministically.

    Sets and frozensets are sorted by their canonical representation so that
    iteration order can never leak into a hash (§4, I8).

    Args:
        value: Any value appearing in a config or manifest tree.

    Returns:
        A structure containing only `dict`, `list`, `str`, `int`, `float`,
        `bool` and `None`.

    Raises:
        SerialisationError: If the value has no stable JSON representation.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Enum):
        return _to_jsonable(value.value)
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, Set):
        return sorted((_to_jsonable(v) for v in value), key=repr)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_to_jsonable(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_jsonable(model_dump(mode="python"))
    msg = (
        f"Cannot canonically serialise {type(value).__name__!r} for hashing. "
        "Add an explicit conversion rather than letting repr() into a digest."
    )
    raise SerialisationError(msg)


def canonical_json(value: Any) -> bytes:  # noqa: ANN401 - accepts arbitrary config trees
    """Serialise a value to canonical JSON bytes.

    Canonical means: keys sorted, `(",", ":")` separators, ASCII-escaped, and
    non-finite floats rejected. The same logical value always produces the same
    bytes, regardless of insertion order.

    Args:
        value: The value to serialise.

    Returns:
        UTF-8 encoded canonical JSON.

    Raises:
        SerialisationError: If the value contains `NaN`/`Infinity`, or a type
            with no stable representation.
    """
    try:
        text = json.dumps(
            _to_jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except ValueError as exc:  # allow_nan=False raises ValueError
        msg = (
            "Refusing to hash a value containing NaN or Infinity. A non-finite "
            "number in a config makes the run unreproducible and every metric "
            "downstream unfalsifiable."
        )
        raise SerialisationError(msg) from exc
    return text.encode("utf-8")


def blake2b_hex(data: bytes, *, digest_size: int = DEFAULT_DIGEST_SIZE) -> str:
    """Hash raw bytes with blake2b.

    Args:
        data: Bytes to hash.
        digest_size: Digest length in bytes.

    Returns:
        Lowercase hex digest of length `2 * digest_size`.
    """
    return hashlib.blake2b(data, digest_size=digest_size).hexdigest()


def hash_json(value: Any, *, digest_size: int = DEFAULT_DIGEST_SIZE) -> str:  # noqa: ANN401
    """Hash a structured value via its canonical JSON form.

    This is how `config_hash` is computed (§5).

    Args:
        value: The value to hash.
        digest_size: Digest length in bytes.

    Returns:
        Lowercase hex digest.

    Raises:
        SerialisationError: If the value cannot be canonically serialised.
    """
    return blake2b_hex(canonical_json(value), digest_size=digest_size)


def hash_path(path: Path, *, digest_size: int = DEFAULT_DIGEST_SIZE) -> str:
    """Hash the contents of a single file, streaming.

    Args:
        path: File to read.
        digest_size: Digest length in bytes.

    Returns:
        Lowercase hex digest of the file's bytes.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    digest = hashlib.blake2b(digest_size=digest_size)
    with path.open("rb") as handle:
        while chunk := handle.read(_FILE_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: Iterable[Path], *, digest_size: int = DEFAULT_DIGEST_SIZE) -> str:
    """Hash a set of files as one content digest.

    Paths are sorted by POSIX-normalised name before hashing, and each file's
    name is mixed in alongside its bytes. Sorting is what makes this safe to
    call with a glob result: filesystem iteration order never reaches the
    digest (§4, I8).

    Args:
        paths: Files to include.
        digest_size: Digest length in bytes.

    Returns:
        Lowercase hex digest covering every file's name and contents.

    Raises:
        FileNotFoundError: If any path does not exist.
    """
    digest = hashlib.blake2b(digest_size=digest_size)
    for path in sorted(paths, key=lambda p: p.as_posix()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(hash_path(path, digest_size=digest_size)))
    return digest.hexdigest()


def short_hash(digest: str, *, length: int = SHORT_HASH_LENGTH) -> str:
    """Truncate a hex digest for display in run ids and UI labels.

    Args:
        digest: A full hex digest.
        length: Number of leading characters to keep.

    Returns:
        The first `length` characters of `digest`.

    Raises:
        SerialisationError: If `digest` is shorter than `length`.
    """
    if len(digest) < length:
        msg = f"Digest {digest!r} is shorter than the requested short length {length}."
        raise SerialisationError(msg)
    return digest[:length]
