# ADR 0003 — PyTorch behind an optional extra instead of TensorFlow / Keras

- **Status:** Accepted
- **Date:** 2026-07-27
- **Phase:** 0 (decision), 13 (implementation)

## Context

The sequence models in scope — LSTM, GRU, a small TCN — are tiny by modern
standards. The framework choice is therefore about install ergonomics and
determinism control, not about modelling power.

TensorFlow's install story on Python 3.12 is fragile, its version pinning drags
in a large transitive tree, and per-op determinism is harder to force.

## Decision

Use PyTorch 2.x, and put it behind an optional `[dl]` extra. `sequence.py`
imports torch lazily, inside function bodies.

`import alphacore` must succeed with torch uninstalled. This is enforced by
`tests/unit/test_import_isolation.py`, which imports `alphacore` in a clean
subprocess and asserts `torch` is absent from `sys.modules`.

## Consequences

**Benefit.** The base install stays small and fast — which matters because CI
runs it on every push, and because a grader following the README should not wait
on a multi-gigabyte download to see a tearsheet. Determinism is controllable via
`torch.use_deterministic_algorithms`, which `runtime.determinism.set_global_seed`
calls when — and only when — torch is already imported.

**Cost.** Two install paths to document and one extra CI dimension. The lazy
import is slightly awkward to read; the isolation test is what keeps it honest.

**Note on import-linter.** No static contract forbids `alphacore -> torch`,
because import-linter sees function-level imports too and would flag the
legitimate lazy import in `sequence.py`. The runtime subprocess test is the
correct tool for this particular constraint.

## What would change at scale

Nothing about the framework. If sequence models became the primary modelling
approach rather than one candidate on the leaderboard, `[dl]` would stop being
optional and torch would move into the base dependencies.
