# Alpha Lab

A research platform for deciding whether a claimed short-horizon predictive
signal on financial time series is **real** — meaning it survives purged
validation, transaction costs, and multiple-testing correction.

It deliberately refuses to trade. There is no broker connectivity, no order
routing, and no recommendation of any kind. The system reports evidence; the
judgement stays with the researcher.

> **Status: Phase 11 of 14.** In place: determinism and hashing, run manifests,
> the immutable artifact store, the config schema, point-in-time ingestion, the
> parquet lake, the duckdb catalog, quality reporting, 27 in-house causal
> feature primitives with a leak-proof registry, four labelling methods with
> uniqueness and time-decay sample weights, purged cross-validation
> (walk-forward, purged k-fold, CPCV), the out-of-fold leaderboard with its
> mandatory baselines, signal generation with hysteresis and three sizing
> methods, the event-driven backtester with its auditable cost model, the full
> metric set including PSR, DSR, minimum track record length and PBO, the
> canonical tearsheet with its generated research note, TreeSHAP explanations,
> hyperparameter search against an out-of-fold *economic* objective, the
> pipelines and CLI that compose all of it — **`alpha run` completes the golden
> path end to end and writes a complete run directory** — and a FastAPI service
> with an in-process job runner, SSE log streaming, RFC 9457 errors, and a
> TypeScript client generated from its own OpenAPI document. All seven leakage
> tests from §12.3 pass, the same config run twice produces a byte-identical
> tearsheet, and the API is contract-tested against its schema. The web app
> arrives at Phase 12; see [PROGRESS.md](PROGRESS.md). This README gets its
> screenshots at Phase 14 (§20).

## Why it exists

Almost any short-horizon backtest can be made to look good. The usual failures
are well known and individually boring: same-bar execution, scaling before
splitting, global quantile labels, zero costs, reporting the best of forty
models without a haircut. Together they produce a Sharpe ratio that means
nothing.

Alpha Lab treats each of those failures as an invariant to enforce and a test to
enforce it with (§4 and §12.3 of [ARCHITECTURE.md](ARCHITECTURE.md)). The
headline number is never presented alone: it comes with a bootstrap confidence
interval, a deflated Sharpe haircut carrying the actual trial count, the
probability of backtest overfitting, and the round-trip cost in basis points at
which the edge disappears.

Some of those invariants are enforced rather than documented. A feature that
names a label column as an input cannot be registered. Quantile label edges
cannot be fitted without an explicit training mask. Correlation pruning cannot
run on the full sample. Every cross-validation fold is purged and embargoed
before it is handed back, and `train_test_split` and plain `KFold` fail the
build if they appear in library code. Every leaderboard carries a
shuffled-label row, so there is always a scale on the axis. The backtester
cannot execute on the bar that produced its signal — orders are queued against
a strictly later bar, and `execution_lag_bars: 0` raises. Each of these raises
rather than warns.

Two tests guard the last one, and they only mean anything together: a perfect
close-of-bar predictor must earn nothing, and a model handed the actual forward
return must earn an absurd amount. Without the second, an engine that simply
never made money would pass the first.

The most valuable output this repository can produce is a run where the model
does **not** find an edge, reported honestly, with the diagnostics that prove
the pipeline would have detected one if it existed.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```bash
just setup
```

```bash
just verify
```

```bash
uv run alpha doctor
```

`alpha doctor` reports whether this machine can produce reproducible runs — the
interpreter version, whether `PYTHONHASHSEED` is pinned, and the versions of
every package that can change a result.

### Ingest some data

The smoke config runs entirely offline, from a committed synthetic fixture:

```bash
uv run alpha data pull -c configs/experiments/smoke.yaml
```

```bash
uv run alpha data inspect --dataset csv:SYNTH:1d
```

`inspect` prints the catalog row — including the `content_hash` a run pins
itself to — and the quality report: bars present against bars the NYSE calendar
says should exist, longest consecutive gap, duplicates, zero-volume bars, OHLC
violations, and return outliers.

To pull real data instead:

```bash
uv run alpha data pull -c configs/experiments/spy_daily_tb.yaml
```

### Run the whole thing

```bash
uv run alpha run -c configs/experiments/smoke.yaml
```

Ingest, features, labels, purged folds, every candidate and every mandatory
baseline scored out of fold, a costed backtest of each, the honesty metrics, and
a written research note — about ten seconds, entirely offline. It leaves a run
directory under `runs/`:

```
runs/{run_id}/
├── tearsheet.json      # the canonical result; the API and UI read nothing else
├── manifest.json       # config hash, data hash, code revision, seed, timings
├── config.resolved.yaml  quality.json  features.json  folds.json
├── equity.parquet  drawdown.parquet  rolling_sharpe.parquet
├── trades.parquet  predictions.parquet  ledger.parquet
├── leaderboard.csv  metrics.csv
├── research.md         # the tearsheet as prose, caveats first
├── models/{model}/model.pkl
└── logs.jsonl
```

The run directory is written once and never modified. Producing a different
result means producing a new run.

```bash
uv run alpha runs list
```

```bash
uv run alpha runs show <run-id>
```

```bash
uv run alpha runs compare <run-id> <run-id>
```

`compare` puts two runs side by side and says so explicitly when they share a
config hash — at which point any difference between them came from the data or
the code, not the configuration.

Individual stages print rather than write: `alpha features build`, `alpha train`
and `alpha backtest` each take the same `-c <config>` and stop where their name
says. Only `alpha run` produces a run directory.

### What a good result looks like

On the committed fixture — a seeded random walk with no predictable structure —
the leaderboard is topped by `baseline_buyhold` at a net Sharpe of 0.960, whose
deflated Sharpe of 0.52 barely clears the line once the six configurations tried
are accounted for. Every model that actually predicts something is flagged
`LOW_DSR` and lands below zero, including a logistic regression with an IC
t-statistic of 3.46 — statistically detectable, economically worthless. The
shuffled-label baseline sits at the bottom where it belongs, and PBO is 0.08.

**That is the pipeline working.** There is no edge in a random walk, and a run
that claimed one would mean something upstream is leaking.

### Serve it

```bash
just serve
```

The API on `127.0.0.1:8000`, with interactive documentation at
[`/api/v1/docs`](http://127.0.0.1:8000/api/v1/docs). It reads the artifact store
and never writes to it — runs are created by jobs and never modified afterwards.

```bash
curl -s localhost:8000/api/v1/runs | jq '.runs[0]'
```

```bash
curl -s -X POST localhost:8000/api/v1/jobs \
  -H 'content-type: application/json' \
  -d '{"type":"full_run","config_ref":"configs/experiments/smoke.yaml"}'
```

Submitting a job returns `202` and a `Location`. Follow the run live:

```bash
curl -N localhost:8000/api/v1/jobs/<job-id>/events
```

That is a server-sent event stream of log lines, structured progress, and one
terminal status frame. It resumes from `Last-Event-ID` if the connection drops.

Series are downsampled server-side to the pixel width you ask for, so a
hundred-thousand-point equity curve arrives as the twelve hundred points a chart
can actually draw:

```bash
curl -s "localhost:8000/api/v1/runs/<run-id>/series/equity?width=1200"
```

Errors are RFC 9457 problem documents that name the cause and the fix rather
than "something went wrong". The service has **no authentication** and can start
jobs, so it binds loopback by default.

```bash
just client
```

Regenerates `docs/openapi.json` and the TypeScript client in
`web/src/api/generated/`. Both are committed, and the test suite fails if they
drift from the routes.

## Documentation

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The build specification. The contract. |
| [PROGRESS.md](PROGRESS.md) | Build log — what is done, what was deferred, what surprised us. |
| [docs/data-dictionary.md](docs/data-dictionary.md) | Every feature: inputs, parameters, warm-up cost. Generated from the registry. |
| [docs/decisions/](docs/decisions/) | ADRs. Every deviation from a naïve stack, with its trade-off. |
| [docs/openapi.json](docs/openapi.json) | The API contract. The TypeScript client is generated from it. |
| [DISCLAIMER.md](DISCLAIMER.md) | Research software. Not investment advice. |

## Disclaimer

Research software, not investment advice. Backtested performance is not
indicative of anything. See [DISCLAIMER.md](DISCLAIMER.md).
