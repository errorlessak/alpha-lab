"""Canonical serialisation and hashing (§4, I8)."""

from __future__ import annotations

import datetime as dt
import math
from enum import Enum
from pathlib import Path

import pytest
from pydantic import BaseModel

from alphacore.errors import SerialisationError
from alphacore.runtime.hashing import (
    DEFAULT_DIGEST_SIZE,
    blake2b_hex,
    canonical_json,
    hash_files,
    hash_json,
    hash_path,
    short_hash,
)


def test_canonical_json_is_key_order_independent():
    a = {"b": 1, "a": {"z": 2, "y": 3}}
    b = {"a": {"y": 3, "z": 2}, "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_has_no_insignificant_whitespace():
    assert canonical_json({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'


def test_canonical_json_sorts_sets_so_iteration_order_cannot_leak():
    # Set iteration order must never reach a digest (§4, I8).
    assert canonical_json({"x", "y", "z"}) == canonical_json({"z", "x", "y"})


def test_canonical_json_rejects_nan():
    with pytest.raises(SerialisationError, match="NaN or Infinity"):
        canonical_json({"sharpe": math.nan})


def test_canonical_json_rejects_infinity():
    with pytest.raises(SerialisationError, match="NaN or Infinity"):
        canonical_json({"sharpe": math.inf})


def test_canonical_json_rejects_unrepresentable_type():
    with pytest.raises(SerialisationError, match="Cannot canonically serialise"):
        canonical_json({"fn": object()})


def test_canonical_json_handles_datetimes_paths_and_enums():
    class Side(Enum):
        LONG = "long"

    payload = {
        "when": dt.datetime(2026, 7, 27, tzinfo=dt.UTC),
        "day": dt.date(2026, 7, 27),
        "where": Path("data/raw"),
        "side": Side.LONG,
    }
    assert canonical_json(payload) == (
        b'{"day":"2026-07-27","side":"long","when":"2026-07-27T00:00:00+00:00","where":"data/raw"}'
    )


def test_canonical_json_accepts_pydantic_models():
    class Meta(BaseModel):
        name: str
        seed: int

    assert canonical_json(Meta(name="spy", seed=7)) == b'{"name":"spy","seed":7}'


def test_canonical_json_preserves_float_precision():
    value = {"threshold": 0.1 + 0.2}
    assert canonical_json(value) == b'{"threshold":0.30000000000000004}'


def test_blake2b_hex_matches_known_vector():
    # blake2b(b"", digest_size=16) — checked against hashlib's own reference.
    assert blake2b_hex(b"") == "cae66941d9efbd404e4d88758ea67670"


def test_blake2b_hex_length_follows_digest_size():
    assert len(blake2b_hex(b"x")) == DEFAULT_DIGEST_SIZE * 2
    assert len(blake2b_hex(b"x", digest_size=32)) == 64


def test_hash_json_is_stable_across_equivalent_trees():
    assert hash_json({"a": 1, "b": 2}) == hash_json({"b": 2, "a": 1})


def test_hash_json_changes_when_a_value_changes():
    assert hash_json({"commission_bps": 0.5}) != hash_json({"commission_bps": 1.5})


def test_hash_path_reads_file_contents(tmp_path: Path):
    target = tmp_path / "bars.parquet"
    target.write_bytes(b"payload")
    assert hash_path(target) == blake2b_hex(b"payload")


def test_hash_path_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        hash_path(tmp_path / "absent.parquet")


def test_hash_files_is_independent_of_listing_order(tmp_path: Path):
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    assert hash_files([first, second]) == hash_files([second, first])


def test_hash_files_is_sensitive_to_filenames(tmp_path: Path):
    original = tmp_path / "a.parquet"
    original.write_bytes(b"one")
    renamed = tmp_path / "renamed.parquet"
    renamed.write_bytes(b"one")
    assert hash_files([original]) != hash_files([renamed])


def test_hash_files_on_empty_input_is_stable():
    assert hash_files([]) == blake2b_hex(b"")


def test_short_hash_truncates():
    assert short_hash("a3f91c7e2b4d") == "a3f91c"
    assert short_hash("a3f91c7e2b4d", length=4) == "a3f9"


def test_short_hash_refuses_to_pad():
    with pytest.raises(SerialisationError, match="shorter than"):
        short_hash("a3f", length=6)
