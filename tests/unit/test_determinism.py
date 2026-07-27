"""Seeding and determinism guards (§4, I8)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from alphacore.errors import DeterminismError
from alphacore.runtime.determinism import (
    HASH_SEED_ENV_VAR,
    deterministic_rng,
    hash_seed_is_pinned,
    require_pinned_hash_seed,
    seed_sequence,
    set_global_seed,
)


def test_set_global_seed_makes_stdlib_random_reproducible():
    set_global_seed(20260727)
    first = [random.random() for _ in range(5)]
    set_global_seed(20260727)
    assert [random.random() for _ in range(5)] == first


def test_set_global_seed_makes_numpy_global_state_reproducible():
    set_global_seed(11)
    # Reads the legacy global state on purpose: that is what is being asserted.
    first = np.random.rand(5).tolist()
    set_global_seed(11)
    assert np.random.rand(5).tolist() == first


@pytest.mark.parametrize("bad", [-1, 2**32, "7", True, 1.5])
def test_set_global_seed_rejects_invalid_seeds(bad):
    with pytest.raises(DeterminismError):
        set_global_seed(bad)


def test_deterministic_rng_is_reproducible():
    assert deterministic_rng(7).standard_normal(4).tolist() == (
        deterministic_rng(7).standard_normal(4).tolist()
    )


def test_deterministic_rng_streams_are_independent():
    a = deterministic_rng(7, "cpcv", 0).standard_normal(8)
    b = deterministic_rng(7, "cpcv", 1).standard_normal(8)
    assert not np.array_equal(a, b)


def test_deterministic_rng_streams_sharing_a_prefix_are_independent():
    # Regression: truncating the canonical JSON of the labels instead of
    # hashing it made every ("cpcv", fold) stream identical.
    draws = [tuple(deterministic_rng(7, "cpcv", fold).standard_normal(4)) for fold in range(8)]
    assert len(set(draws)) == 8


def test_deterministic_rng_stream_labels_are_stable():
    first = deterministic_rng(7, "bootstrap", 3).standard_normal(4)
    second = deterministic_rng(7, "bootstrap", 3).standard_normal(4)
    assert np.array_equal(first, second)


def test_deterministic_rng_differs_from_unlabelled_stream():
    plain = deterministic_rng(7).standard_normal(4)
    labelled = deterministic_rng(7, "shuffled_labels").standard_normal(4)
    assert not np.array_equal(plain, labelled)


def test_deterministic_rng_rejects_invalid_seed():
    with pytest.raises(DeterminismError):
        deterministic_rng(-3)


def test_seed_sequence_is_reproducible_and_distinct():
    first = list(seed_sequence(20260727, 5))
    assert first == list(seed_sequence(20260727, 5))
    assert len(set(first)) == 5
    assert all(0 <= s <= 2**32 - 1 for s in first)


def test_seed_sequence_of_zero_is_empty():
    assert list(seed_sequence(1, 0)) == []


def test_seed_sequence_rejects_negative_count():
    with pytest.raises(DeterminismError, match="Cannot produce"):
        list(seed_sequence(1, -1))


def test_hash_seed_is_pinned_reads_the_environment(monkeypatch):
    monkeypatch.setenv(HASH_SEED_ENV_VAR, "0")
    assert hash_seed_is_pinned() is True
    monkeypatch.setenv(HASH_SEED_ENV_VAR, "1")
    assert hash_seed_is_pinned() is False
    monkeypatch.delenv(HASH_SEED_ENV_VAR, raising=False)
    assert hash_seed_is_pinned() is False


def test_require_pinned_hash_seed_passes_when_pinned(monkeypatch):
    monkeypatch.setenv(HASH_SEED_ENV_VAR, "0")
    require_pinned_hash_seed()


def test_require_pinned_hash_seed_raises_when_unset(monkeypatch):
    monkeypatch.delenv(HASH_SEED_ENV_VAR, raising=False)
    with pytest.raises(DeterminismError, match=HASH_SEED_ENV_VAR):
        require_pinned_hash_seed()


def test_set_global_seed_can_demand_a_pinned_hash_seed(monkeypatch):
    monkeypatch.delenv(HASH_SEED_ENV_VAR, raising=False)
    with pytest.raises(DeterminismError):
        set_global_seed(1, strict_hash_seed=True)
