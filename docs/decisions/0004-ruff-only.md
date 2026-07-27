# ADR 0004 — Ruff as the only formatter and linter

- **Status:** Accepted
- **Date:** 2026-07-27
- **Phase:** 0

## Context

The conventional setup is Black for formatting plus Ruff (or flake8) for
linting. `ruff format` is a reimplementation of Black's algorithm and produces
byte-identical output on virtually all real code.

Running both means two tools with overlapping opinions and two configuration
surfaces that can drift apart — the classic symptom being a pre-commit hook that
reformats a file the CI linter then rejects.

## Decision

Ruff only: `ruff format` for formatting, `ruff check` for linting. No Black, no
isort, no flake8, no pydocstyle as a separate tool — Ruff's `I` and `D` rule
families cover the last two.

The enabled rule set is deliberately broad, and two selections are load-bearing
rather than stylistic:

- **`T20`** bans `print()`. §18 requires it, and the reason is structural: a
  `print` carries no level, no timestamp and no run id, so it cannot reach
  `runs/{run_id}/logs.jsonl` and is therefore invisible to the person debugging
  a failed run.
- **`ANN` + `D`** require annotations and docstrings on public functions, which
  §18 also requires. Enforcing them mechanically means the data dictionary and
  the API docs can be generated rather than maintained.

## Consequences

**Benefit.** One tool, one config block, one cache. Formatting and linting can
never disagree. Substantially faster, which matters for the pre-commit hook.

**Cost.** A dependency on Ruff's formatter staying Black-compatible. If it ever
diverges meaningfully, the recovery is to pin the Ruff version — the same
exposure any formatter choice carries.

**Also rejected:** per-file formatter opt-outs. A file that needs different
formatting rules is a file whose formatting is being used to hide something.
