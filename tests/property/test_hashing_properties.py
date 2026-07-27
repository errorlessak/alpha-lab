"""Property tests for canonical serialisation.

The determinism guarantee in §4 (I8) is a claim about *all* configs, not the
handful in `configs/`. Hypothesis is how that claim gets tested.
"""

from __future__ import annotations

import json

from hypothesis import given
from hypothesis import strategies as st

from alphacore.runtime.hashing import canonical_json, hash_json

finite_floats = st.floats(allow_nan=False, allow_infinity=False, width=64)

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**53), max_value=2**53)
    | finite_floats
    | st.text(max_size=32),
    lambda children: (
        st.lists(children, max_size=6) | st.dictionaries(st.text(max_size=12), children, max_size=6)
    ),
    max_leaves=25,
)

config_trees = st.dictionaries(st.text(min_size=1, max_size=12), json_values, max_size=8)


@given(config_trees)
def test_canonical_json_round_trips(tree):
    assert json.loads(canonical_json(tree).decode("utf-8")) == tree


@given(config_trees)
def test_canonical_json_is_idempotent(tree):
    once = canonical_json(tree)
    assert canonical_json(json.loads(once.decode("utf-8"))) == once


@given(config_trees)
def test_canonical_json_keys_are_sorted(tree):
    decoded = json.loads(canonical_json(tree).decode("utf-8"))
    assert list(decoded) == sorted(decoded)


@given(config_trees)
def test_hash_is_a_function_of_the_value_not_its_construction(tree):
    rebuilt = dict(reversed(list(tree.items())))
    assert hash_json(tree) == hash_json(rebuilt)


@given(config_trees, config_trees)
def test_different_trees_hash_differently(a, b):
    if canonical_json(a) != canonical_json(b):
        assert hash_json(a) != hash_json(b)
