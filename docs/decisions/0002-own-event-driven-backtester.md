# ADR 0002 — Own event-driven backtester instead of VectorBT / Backtrader

- **Status:** Accepted
- **Date:** 2026-07-27
- **Phase:** 0 (decision), 7 (implementation)

## Context

VectorBT is fast and Backtrader is mature. Either would produce an equity curve
in a fraction of the time.

But every interesting question about a backtest is a question about assumptions:
*when* did you fill, at *what* price, with *what* costs, and what happened when
the order was larger than the bar's volume? A vendored engine answers those
questions somewhere in its source, under configuration flags whose defaults
usually assume the friendliest case. The single most common way a backtest lies
is same-bar execution — signal from `close[t]`, fill at `close[t]` — and that is
a default in more libraries than it should be.

## Decision

Write `alphacore.backtest.engine` as an explicit event loop over bars, where the
engine's state object never holds a reference to future bars, and bars are
supplied by an iterator rather than a frame. The ordering within a bar is fixed
and documented (§12.1):

mark to market at the open, apply risk exits against the bar's high/low with the
**stop** winning any tie, execute orders queued at `t - execution_lag`, read the
signal available at `t.close`, size, apply limits, queue for `t + lag`, append to
the ledger.

Guard it with three things:

1. `test_no_same_bar_execution` — a synthetic series where `close[t]` perfectly
   predicts the future must produce ~0 Sharpe
2. `test_perfect_foresight_sanity` — a model handed the actual forward return
   must produce an absurd Sharpe, proving test 1 can detect anything at all
3. `backtest/reference.py` — a deliberately dumb vectorised engine that the
   event-driven engine must match to `1e-8` under a frictionless config

## Consequences

**Cost.** An event loop in Python is orders of magnitude slower than a
vectorised one, and the accounting is ours to get right. Phase 7 is the largest
single phase in the build.

**Benefit.** Every assumption is a line of code someone can read. The
cross-check against `reference.py` catches accounting bugs that neither engine
would catch alone: if the two disagree, one of them is wrong, and the frictionless
configuration removes every legitimate reason for them to differ.

**Risk accepted.** Slower backtests constrain how many configurations can be
swept, which feeds directly into the DSR trial count. That is a real cost, and
the honest framing is that it discourages exactly the kind of unbounded search
that DSR exists to punish.

## What would change at scale

For a portfolio of thousands of instruments the Python loop becomes the
bottleneck. The move then is to keep this engine as the reference oracle and
port the hot loop to a compiled implementation, holding the `1e-8` agreement test
as the acceptance criterion.
