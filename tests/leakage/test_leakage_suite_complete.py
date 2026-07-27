"""The leakage suite must grow as the build does.

§12.3 names seven tests, and calls them the highest-priority tests in the
project. Each one becomes possible only once the machinery it inspects exists,
so this file records the full list against the phase that unlocks it and fails
if a test that *should* exist by now does not.

Without this, `just test-leak` would pass on an empty directory for eleven
phases and then quietly keep passing if a test were ever deleted. A green
leakage suite has to mean something.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.conftest import CURRENT_PHASE

LEAKAGE_DIR = Path(__file__).parent


@dataclass(frozen=True)
class RequiredTest:
    """One of the seven mandatory leakage tests.

    Attributes:
        name: The test function name required by §12.3.
        unlocked_by_phase: The §15 phase after which this test must exist.
        summary: What the test proves.
    """

    name: str
    unlocked_by_phase: int
    summary: str


REQUIRED_TESTS: tuple[RequiredTest, ...] = (
    RequiredTest(
        "test_future_poison",
        2,
        "Corrupting bars after index k must not change feature rows 0..k (I1).",
    ),
    RequiredTest(
        "test_quantile_edges_per_fold",
        3,
        "Label bin edges must differ across folds, proving they were fitted per-fold.",
    ),
    RequiredTest(
        "test_label_span_purged",
        4,
        "No training label span may intersect the test window, and the embargo gap "
        "must be present (I4).",
    ),
    RequiredTest(
        "test_pipeline_fit_on_train_only",
        5,
        "The scaler's fit must never see a test index (I3).",
    ),
    RequiredTest(
        "test_shuffled_labels_have_no_edge",
        5,
        "On shuffled labels the OOS Sharpe confidence interval must contain zero.",
    ),
    RequiredTest(
        "test_no_same_bar_execution",
        7,
        "A perfect close-of-bar predictor must yield ~0 Sharpe (I2).",
    ),
    RequiredTest(
        "test_perfect_foresight_sanity",
        7,
        "A model handed the forward return must yield an absurd Sharpe — the inverse "
        "canary that proves the same-bar test can detect anything at all.",
    ),
)


def _collected_test_names() -> set[str]:
    """Names of every test function defined in the leakage suite.

    Returns:
        Function names found by scanning `def test_*` across the directory. A
        text scan rather than an import, so a module that fails to import
        cannot be counted as present.
    """
    names: set[str] = set()
    for path in LEAKAGE_DIR.rglob("test_*.py"):
        for line in path.read_text("utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("def test_"):
                names.add(stripped.removeprefix("def ").split("(", 1)[0])
    return names


def test_required_leakage_tests_are_unique():
    names = [t.name for t in REQUIRED_TESTS]
    assert len(names) == len(set(names))
    assert len(REQUIRED_TESTS) == 7


@pytest.mark.parametrize("required", REQUIRED_TESTS, ids=lambda t: t.name)
def test_unlocked_leakage_tests_exist(required: RequiredTest):
    if required.unlocked_by_phase > CURRENT_PHASE:
        pytest.skip(
            f"unlocked by phase {required.unlocked_by_phase}; build is at phase {CURRENT_PHASE}"
        )
    assert required.name in _collected_test_names(), (
        f"{required.name} is required from phase {required.unlocked_by_phase} onwards: "
        f"{required.summary}"
    )
