# ADR 0011 — Sequence windows are cut on the bar grid, and the model is told what that grid is

- **Status:** Accepted
- **Date:** 2026-07-28
- **Phase:** 13

## Context

§10.3 requires windowed tensors built **inside** the fold, "with the same purge
logic applied at the window level (a window that straddles the boundary is
dropped)". Deciding whether a window straddles a boundary turns out to be the
whole problem.

`AlphaModel.fit` receives one fold's training rows *after* purge and embargo, so
the frame has a hole in it where the test block was. Nothing in the frame says
so. Three ways of detecting the hole were considered:

1. **Row position inside the frame.** Useless: after the purged rows are
   removed, the last bar before the gap and the first bar after it are adjacent
   positions. This is precisely the leak.
2. **Elapsed time between timestamps.** A gap in *time* is not a gap in *bars*.
   Friday and Monday are adjacent bars in a daily equity index, as are the two
   sides of a holiday, an early close, or a halt. Any threshold that separates a
   weekend from a purge gap is a heuristic, and a leakage check that depends on a
   tuned threshold is not a check.
3. **Position on the full bar grid.** Exact. The purged rows exist in the
   complete feature index; the fold's rows are a subset of it; "consecutive"
   means "adjacent in that index".

Option 3 needs information `fit(X, y)` does not carry. The signature could have
grown a parameter, but it is the shared contract for eleven other models that
have no use for it, and a parameter that ten implementations ignore is a
parameter someone will forget to pass.

## Decision

Sequence models expose `set_bar_grid(index)`, and **raise from `fit` if it was
never called**. The error names the method and says why.

`alphacore.models.oof.fit_predict_oof` calls it once, before the fold loop, on
any model that exposes it — that function is the only place which knows both the
complete index and which rows each fold kept. `alphacore.pipelines.full_run`
does the same where it refits the winning model for export.
`EnsembleModel.set_bar_grid` forwards to whichever legs want it.

Windowing itself lives in `alphacore.models.windows`, which imports no torch:
`grid_positions` locates rows on the grid, `contiguous_runs` cuts the fold into
maximal unbroken runs, and `build_windows` never lets a window span two runs.

Training and prediction differ in one way, deliberately. Training uses
`pad=False`: a row becomes a training sample only if a full `seq_len` of real
history sits behind it inside its own run, so the rows at a fold boundary are
dropped rather than fabricated. Prediction uses `pad=True`, because the OOF loop
needs one score per row and a missing row would misalign every downstream metric;
the short history at the head of a run is filled by repeating that run's first
bar, which is a bar the model may legitimately see.

## Consequences

**Benefit.** The check is exact and needs no threshold.
`tests/leakage/test_window_purge.py` proves it against a control — the obvious
positional windower — which does straddle the gap and fails the same assertion.

**Benefit.** The requirement is enforced rather than documented. A sequence model
fitted by hand without the grid raises a message naming the fix; it cannot
silently train across a test block.

**Cost.** `AlphaModel` is no longer the complete contract for every model. There
is a second, optional capability discovered with `getattr`, which is duck typing
in a codebase that otherwise prefers explicit protocols. The alternative was
widening `fit` for everyone.

**Cost.** Sequence models lose `seq_len - 1` training rows at the head of every
run — two runs per fold with a middle test block, so twice that per fold. On a
2,000-row sample with a 32-bar window that is about 3% of the training rows.
Padding them would have cost nothing and invented bars, which is worse.
