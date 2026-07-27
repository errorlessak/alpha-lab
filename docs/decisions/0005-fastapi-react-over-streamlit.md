# ADR 0005 — FastAPI + React instead of Streamlit

- **Status:** Accepted
- **Date:** 2026-07-27
- **Phase:** 0 (decision), 11–12 (implementation)

## Context

Streamlit would put charts on screen in an afternoon. For a research tool whose
entire product is *how the evidence is presented*, that afternoon buys the wrong
thing.

Three concrete limits decide it:

1. **Performance.** Equity series here run to 100k+ points. Streamlit re-runs
   the script on every interaction and has no route to server-side LTTB
   downsampling behind a pixel-width parameter.
2. **Layout control.** The three devices this UI is built around — the Fold
   Strip, the Provenance Stamp, and the margin critique gutter (§13.2) — are
   custom SVG and custom layout. None is expressible as a widget.
3. **Separation.** A Streamlit app calls research code in-process. That makes it
   easy to accidentally *compute* in the view layer, which is how a UI ends up
   producing numbers that no run artifact contains.

## Decision

A FastAPI service that reads the immutable artifact store and nothing else, plus
a Vite + React 19 + TypeScript front end (§13.3). The API's contract is
`tearsheet.json` and its sidecar parquet series; the UI never computes a metric.

Bearer-token auth is a single optional env var. No RBAC, no SSO — this is a
single-user local tool and pretending otherwise would be unearned complexity.

## Consequences

**Cost.** Phases 11 and 12 are real work: an OpenAPI contract, a generated TS
client, route-level code splitting, a bundle budget, and a Playwright suite.
Materially more than an afternoon.

**Benefit.** The UI becomes a deliverable rather than a debug view. Because the
API only reads artifacts, any number on screen can be traced to a file on disk —
which is what makes the Provenance Stamp truthful rather than decorative.

**Constraint accepted.** Every result the UI shows must exist in an artifact
first. If a chart needs a number, the pipeline computes it and writes it. This
is a feature: it means a screenshot and a `runs/` directory always agree.
