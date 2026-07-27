# Alpha Lab

A research platform for deciding whether a claimed short-horizon predictive
signal on financial time series is **real** — meaning it survives purged
validation, transaction costs, and multiple-testing correction.

It deliberately refuses to trade. There is no broker connectivity, no order
routing, and no recommendation of any kind. The system reports evidence; the
judgement stays with the researcher.

> **Status: Phase 0 of 14.** The runtime foundations are in place — determinism,
> hashing, run manifests, the immutable artifact store, and the tooling gate.
> The research pipeline arrives phase by phase; see [PROGRESS.md](PROGRESS.md).
> This README gets its screenshots, its honest headline result, and its
> quickstart at Phase 14 (§20).

## Why it exists

Almost any short-horizon backtest can be made to look good. The usual failures
are well known and individually boring: same-bar execution, scaling before
splitting, global quantile labels, zero costs, reporting the best of forty models
without a haircut. Together they produce a Sharpe ratio that means nothing.

Alpha Lab treats each of those failures as an invariant to enforce and a test to
enforce it with (§4 and §12.3 of [ARCHITECTURE.md](ARCHITECTURE.md)). The
headline number is never presented alone: it comes with a bootstrap confidence
interval, a deflated Sharpe haircut carrying the actual trial count, the
probability of backtest overfitting, and the round-trip cost in basis points at
which the edge disappears.

The most valuable output this repository can produce is a run where the model
does **not** find an edge, reported honestly, with the diagnostics that prove the
pipeline would have detected one if it existed.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```bash
just setup
just verify
uv run alpha doctor
```

`alpha doctor` reports whether this machine can produce reproducible runs — the
interpreter version, whether `PYTHONHASHSEED` is pinned, and the versions of
every package that can change a result.

```bash
uv run alpha dev scratch-run
```

Writes a throwaway run directory with a valid manifest, exercising the full
artifact lifecycle without touching market data.

## Documentation

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The build specification. The contract. |
| [PROGRESS.md](PROGRESS.md) | Build log — what is done, what was deferred, what surprised us. |
| [docs/decisions/](docs/decisions/) | ADRs. Every deviation from a naïve stack, with its trade-off. |
| [DISCLAIMER.md](DISCLAIMER.md) | Research software. Not investment advice. |

## Disclaimer

Research software, not investment advice. Backtested performance is not
indicative of anything. See [DISCLAIMER.md](DISCLAIMER.md).
