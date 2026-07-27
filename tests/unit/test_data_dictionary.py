"""The generated data dictionary stays in step with the registry (§7.1).

Phase 2's exit criteria require this as a CI check. Documentation that drifts
from the code is worse than none: a reviewer reading that `fracdiff` needs 282
bars of warm-up, when the registry now says 1,458, is being actively misled.
"""

from __future__ import annotations

import pytest

from alphacore.features import emit_data_dictionary, registered_names

DOC_PATH_PARTS = ("docs", "data-dictionary.md")


@pytest.fixture
def generated() -> str:
    return emit_data_dictionary()


def test_the_committed_file_matches_the_registry(repo_root, generated):
    path = repo_root.joinpath(*DOC_PATH_PARTS)
    assert path.is_file(), f"{path} is missing. Run `just docs`."
    committed = path.read_text("utf-8")
    assert committed == generated, (
        "docs/data-dictionary.md is out of date with the feature registry. "
        "Regenerate it with `just docs` and commit the result."
    )


def test_every_registered_feature_is_documented(generated):
    for name in registered_names():
        assert f"`{name}`" in generated, f"{name} is missing from the data dictionary"


def test_the_dictionary_has_exactly_one_row_per_feature(generated):
    rows = [line for line in generated.splitlines() if line.startswith("| `")]
    assert len(rows) == len(registered_names())


def test_generation_is_deterministic():
    # Registry iteration order must never reach a committed file (§4, I8).
    assert emit_data_dictionary() == emit_data_dictionary()


def test_the_dictionary_explains_the_lookback_cost(generated):
    assert "lookback" in generated
    assert "I5" in generated


def test_families_appear_as_sections(generated):
    assert "## Trend" in generated
    assert "## Volatility" in generated
