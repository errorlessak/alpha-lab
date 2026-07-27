# ADR 0009 — `just verify` gains `smoke` and `e2e` as their phases land

- **Status:** Accepted
- **Date:** 2026-07-27
- **Phase:** 0

## Context

§17 defines the gate as:

```
verify: just fmt lint types test test-leak smoke
```

`smoke` runs `alpha run -c configs/experiments/smoke.yaml`, which requires the
full pipeline — data, features, labels, CV, models, backtest, report. That does
not exist until Phase 10. A Phase 0 `verify` that includes it can only be red.

§15's Phase 0 exit criterion says `just verify` must be green "on an empty test
suite", so the spec already assumes `smoke` is not gating at Phase 0. The two
sections are consistent in intent; only the recipe needs sequencing.

## Decision

`verify` covers `fmt lint types test test-leak` from Phase 0 onward. `smoke`
joins it at Phase 10, `e2e` at Phase 12. `verify-full` exists from day one and
names all seven, so the final shape of the gate is visible in the justfile
rather than living in someone's head.

Two guards stop this becoming a way to stay green by omission:

1. **`tests/leakage/test_leakage_suite_complete.py`** holds the seven mandatory
   leakage tests from §12.3 against the phase that unlocks each one, reading
   `CURRENT_PHASE` from `tests/conftest.py`. Bumping the phase without wiring
   the corresponding tests turns the suite red. Without it, `just test-leak`
   would pass on an empty directory for eleven phases — and keep passing if a
   test were later deleted.
2. **`PROGRESS.md`** records, per phase, which recipes are actually in `verify`.

## Consequences

**Benefit.** The gate is honest at every phase: green means everything that
*can* be checked has been.

**Cost.** `CURRENT_PHASE` is self-reported. The leakage registry converts that
from a comment into something that fails loudly when it is wrong, but it cannot
stop someone from simply not bumping it. `PROGRESS.md` and the ADR trail are the
backstop.

**Superseded when:** Phase 12 completes. At that point `verify` and
`verify-full` are the same recipe and this ADR is history.
