# ADR 0001 — In-house indicator primitives instead of `pandas-ta` / TA-Lib

- **Status:** Accepted
- **Date:** 2026-07-27
- **Phase:** 0 (decision), 2 (implementation)

## Context

The obvious move is to depend on `pandas-ta` or TA-Lib for RSI, MACD, ATR,
Bollinger bands and the rest. Both are widely used and would save a week.

Two problems. The practical one: `pandas-ta` is effectively unmaintained and
breaks on NumPy 2 and Python 3.12; TA-Lib needs a C library that turns a
one-line install into a support burden on three operating systems.

The load-bearing one: invariant I1 says a feature at time `t` may depend only on
data at or before `t`. A third-party indicator is a black box with respect to
that claim. Its warm-up behaviour, its centring, and its handling of the first
`n` bars are implementation details nobody has certified as causal. If a
reviewer asks "how do you know your features do not see the future?", the honest
answer with a vendored library is "I assume the maintainers got it right".

## Decision

Implement every primitive in `alphacore.features.primitives` as a pure,
numpy-vectorised function, with:

- the formula in the docstring
- an explicit `lookback` declared on the `FeatureSpec`
- a `NaN` warm-up region, never zero-filled and never backfilled
- reference tests for RSI, MACD, ATR and Bollinger against hand-verified values
  from the original published definitions, asserted to `1e-9`
- a parametrised future-poison test (§12.3) run over the entire registry

## Consequences

**Cost.** More code to write and maintain. A bug in our EMA is our bug. Roughly
one phase of work (Phase 2).

**Benefit.** Causality becomes a property that is *tested*, not assumed. The
future-poison test is only meaningful because every indicator is ours to
instrument. The data dictionary can be generated from the registry, so the
documentation cannot drift from the implementation.

**Risk accepted.** Our indicators may disagree with a vendor's in the third
decimal place — usually because vendors differ on Wilder vs simple smoothing.
The golden reference tests pin us to the published definition, and the docstring
records which convention we chose.

## What would change at scale

With hundreds of features across many asset classes, hand-writing every
primitive stops paying. The right move then is to keep the registry and the
causality contract, and let specs delegate to a vetted vendor implementation —
the contract is the valuable part, not the arithmetic.
