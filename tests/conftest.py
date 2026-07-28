"""Shared fixtures and the build-phase marker.

`CURRENT_PHASE` is the single place recording how far the build in §15 has got.
`tests/leakage/test_leakage_suite_complete.py` reads it to decide which of the
seven mandatory leakage tests must already exist, so bumping it without wiring
the corresponding tests turns the suite red.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from alphacore.runtime import ArtifactStore, configure_logging

if TYPE_CHECKING:
    from collections.abc import Iterator

CURRENT_PHASE = 10
"""Highest fully completed phase from ARCHITECTURE.md §15. Bump only when that
phase's exit criteria pass."""

FROZEN_NOW = dt.datetime(2026, 7, 27, 14, 22, 1, tzinfo=dt.UTC)


@pytest.fixture(autouse=True)
def _quiet_logging() -> Iterator[None]:
    """Keep loguru at WARNING so test output stays readable."""
    configure_logging("WARNING", colorize=False)
    yield
    configure_logging("WARNING", colorize=False)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    """An artifact store rooted in a throwaway directory."""
    return ArtifactStore(tmp_path / "runs")


@pytest.fixture
def frozen_now() -> dt.datetime:
    """A fixed UTC instant, so run ids do not depend on the clock."""
    return FROZEN_NOW


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root.

    Session-scoped: it is a constant, and module-scoped fixtures that need it
    cannot depend on a function-scoped one.
    """
    return Path(__file__).resolve().parent.parent
