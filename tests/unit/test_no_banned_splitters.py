"""Random splitters are banned outside the test tree (§9).

`train_test_split` and plain `KFold` shuffle by default, and on a time series
that means training on rows drawn from after the test rows, next to test rows,
sharing their feature windows. §19 lists `train_test_split(shuffle=True)` on
time series as one of the things that sinks a project.

§9 asks for a CI grep. This is a parser instead, and deliberately so: a grep
over source text also matches the docstrings that *explain* the ban, and a
check that fires on its own documentation gets silenced. Walking the AST looks
at identifiers only, so prose is invisible to it and
`from sklearn.model_selection import KFold as _k` is not.

`PurgedKFold` is ours and is not matched: identifiers are compared whole, never
by substring.
"""

from __future__ import annotations

import ast

import pytest

BANNED_NAMES = frozenset(
    {
        "train_test_split",
        "KFold",
        "StratifiedKFold",
        "ShuffleSplit",
        "StratifiedShuffleSplit",
        "GroupKFold",
        "TimeSeriesSplit",
    }
)

SEARCH_ROOTS = ("packages",)
SKIP_DIRS = frozenset({".venv", "__pycache__", ".mypy_cache", ".ruff_cache"})


def source_files(repo_root):
    for root in SEARCH_ROOTS:
        for path in sorted((repo_root / root).rglob("*.py")):
            if not any(part in SKIP_DIRS for part in path.parts):
                yield path


def banned_identifiers(source: str) -> list[tuple[str, int]]:
    """Find banned splitter identifiers used as code, ignoring prose.

    Args:
        source: Python source text.

    Returns:
        `(identifier, line)` pairs for every banned name that appears as a
        reference, an attribute access, or an import.
    """
    tree = ast.parse(source)
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            found.append((node.id, node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_NAMES:
            found.append((node.attr, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            found += [(a.name, node.lineno) for a in node.names if a.name in BANNED_NAMES]
        elif isinstance(node, ast.Import):
            found += [
                (a.name.rsplit(".", 1)[-1], node.lineno)
                for a in node.names
                if a.name.rsplit(".", 1)[-1] in BANNED_NAMES
            ]
    return found


def test_no_banned_splitter_appears_in_library_code(repo_root):
    offenders = []
    for path in source_files(repo_root):
        for name, line in banned_identifiers(path.read_text("utf-8")):
            offenders.append(f"{path.relative_to(repo_root).as_posix()}:{line} {name}")

    assert offenders == [], (
        "Random or unpurged splitters found in library code:\n  "
        + "\n  ".join(offenders)
        + "\nUse alphacore.cv.WalkForward, PurgedKFold, or CPCV instead (§9)."
    )


def test_our_own_purged_kfold_is_not_matched():
    # The ban must not fire on the class that exists to replace the banned one.
    assert banned_identifiers("from alphacore.cv import PurgedKFold\ncv = PurgedKFold(5)\n") == []


def test_prose_mentioning_the_ban_is_not_matched():
    # This is why the check parses instead of grepping.
    assert banned_identifiers('"""Never use train_test_split or KFold here."""\n') == []


@pytest.mark.parametrize(
    "snippet",
    [
        "from sklearn.model_selection import train_test_split",
        "a, b = train_test_split(X, y)",
        "cv = KFold(n_splits=5, shuffle=True)",
        "from sklearn.model_selection import KFold as _k",
        "splitter = model_selection.StratifiedKFold(5)",
        "ShuffleSplit(n_splits=10)",
        "TimeSeriesSplit(n_splits=5)",
    ],
)
def test_the_ban_detects_real_violations(snippet):
    # The control: a check that matched nothing would pass forever.
    assert banned_identifiers(snippet)


def test_the_scanner_actually_reads_files(repo_root):
    assert len(list(source_files(repo_root))) > 20
