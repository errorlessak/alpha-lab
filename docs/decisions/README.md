# Architecture decision records

One file per decision, numbered, never edited after acceptance — a superseded
decision gets a new ADR that says so.

ADRs 0001–0007 record the seven deliberate deviations from the naïve stack in
§2 of `ARCHITECTURE.md`. Each is a trade-off someone could reasonably have made
differently, which is exactly why it is written down.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-in-house-indicator-library.md) | In-house indicator primitives instead of `pandas-ta` / TA-Lib | Accepted |
| [0002](0002-own-event-driven-backtester.md) | Own event-driven backtester instead of VectorBT / Backtrader | Accepted |
| [0003](0003-pytorch-over-tensorflow.md) | PyTorch behind an optional extra instead of TensorFlow / Keras | Accepted |
| [0004](0004-ruff-only.md) | Ruff as the only formatter and linter instead of Black + Ruff | Accepted |
| [0005](0005-fastapi-react-over-streamlit.md) | FastAPI + React instead of Streamlit | Accepted |
| [0006](0006-uv-lockfile-over-pip.md) | `uv` + `uv.lock` instead of `pip` + `requirements.txt` | Accepted |
| [0007](0007-economic-leaderboard.md) | Out-of-fold economic leaderboard instead of an accuracy leaderboard | Accepted |
| 0008 | In-process job runner instead of Celery / Redis | Reserved — Phase 11 |
| [0009](0009-phase-gated-verify.md) | `just verify` gains `smoke` and `e2e` as their phases land | Accepted |

## A numbering conflict, resolved

§2 assigns the seven stack deviations to ADRs 0001–0007. §14.2 separately refers
to "ADR 0006" as the job-runner decision. Both cannot hold.

Two readings were available: renumber the stack deviations so the job runner
lands on 0006, or keep §2's explicit range and give the job runner the next free
number. The second is the more conservative reading — §2 states its range
explicitly and covers exactly seven items — so the job-runner ADR is **0008**,
reserved until Phase 11 implements it.
