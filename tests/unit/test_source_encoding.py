"""Every tracked text file is clean UTF-8 (§4, I8).

This exists because it happened. Twice. A PowerShell `-replace` decoded a UTF-8
source file as ANSI and wrote it back as UTF-8, turning every `§` and `—` into a
two-character sequence and prepending a byte-order mark. Nothing failed: the
code still ran, the tests still passed, and the damage sat in docstrings and a
README until someone read them.

A BOM is the more dangerous of the two. Python tolerates one on a source file,
but it changes the bytes, and §12.4 compares run artifacts byte for byte.
"""

from __future__ import annotations

import pytest

TEXT_SUFFIXES = frozenset({".py", ".md", ".toml", ".yaml", ".yml", ".csv", ".json"})
SKIP_DIRS = frozenset(
    {".venv", ".git", "node_modules", "data", "runs", ".ruff_cache", ".mypy_cache", ".pytest_cache"}
)

#: Sequences that appear when UTF-8 is decoded as latin-1 and re-encoded.
MOJIBAKE = (
    "Â§",  # section sign
    "â",  # em dash
    "â",  # right single quote
    "Ã¯",  # i-diaeresis, as in "naive"
)

BOM = b"\xef\xbb\xbf"
_SELF = "test_source_encoding.py"


def text_files(root):
    for path in sorted(root.rglob("*")):
        if path.suffix in TEXT_SUFFIXES and not any(p in SKIP_DIRS for p in path.parts):
            yield path


def test_no_file_carries_a_byte_order_mark(repo_root):
    offenders = [
        p.relative_to(repo_root).as_posix()
        for p in text_files(repo_root)
        if p.read_bytes().startswith(BOM)
    ]
    assert offenders == [], f"UTF-8 BOM found in {offenders}"


def test_no_file_carries_mojibake(repo_root):
    offenders = []
    for path in text_files(repo_root):
        # This module necessarily spells the patterns it searches for, so it
        # would otherwise always be the first match.
        if path.name == _SELF:
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        hits = [m for m in MOJIBAKE if m in text]
        if hits:
            offenders.append(path.relative_to(repo_root).as_posix())
    assert offenders == [], (
        f"Mojibake found in {offenders}. A UTF-8 file was decoded as ANSI and "
        "re-encoded; restore it from git rather than patching the characters."
    )


def test_every_file_decodes_as_utf8(repo_root):
    for path in text_files(repo_root):
        try:
            path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - guarded by the assert
            pytest.fail(f"{path.relative_to(repo_root)} is not valid UTF-8: {exc}")


def test_the_scanner_actually_sees_files(repo_root):
    # A scanner that silently matched nothing would pass forever.
    assert len(list(text_files(repo_root))) > 50


def test_the_scanner_detects_damage(tmp_path):
    """The control: corrupt a file the way PowerShell did and confirm a hit."""
    clean = "Section §7.2 — the feature registry.\n"
    damaged = clean.encode("utf-8").decode("latin-1")
    assert any(pattern in damaged for pattern in MOJIBAKE)

    path = tmp_path / "damaged.md"
    path.write_bytes(BOM + damaged.encode("utf-8"))
    assert path.read_bytes().startswith(BOM)
