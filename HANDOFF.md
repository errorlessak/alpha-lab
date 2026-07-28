# Handoff — Alpha Lab

**Read this first, then `ARCHITECTURE.md`, then `PROGRESS.md`.**

This file exists because the build spans more than one session. It carries the
things that are *not* recoverable from the code: environment quirks that cost
time, conventions that look arbitrary until you know why, mistakes already made
so they are not made twice, and the exact state of the phase gate.

---

## 1. What this project is

**Alpha Lab** — a research platform for deciding whether a claimed short-horizon
predictive signal on financial time series is **real**, meaning it survives
purged validation, transaction costs, and multiple-testing correction.

`ARCHITECTURE.md` at the repo root is **the contract**. It is a ~1,300-line
build specification with numbered sections (§4 invariants, §12.3 leakage tests,
§15 the phase roadmap). When it and instinct disagree, it wins. When it is
silent, write an ADR in `docs/decisions/` and move on.

The single most important framing, from §19: *the most impressive thing this
repository can contain is a run where the model **does not** find an edge,
reported honestly, with the diagnostics that prove the pipeline would have
detected one if it existed.*

---

## 2. Where things are

- **Repo:** `C:\dev\alpha-lab` (git repo, branch `main`, working tree clean)
- **Original spec copy:** the user's copy lives at
  `C:\Users\akshy\OneDrive\Desktop\alpha ml\ARCHITECTURE_1.md`. The repo's
  `ARCHITECTURE.md` is the same file. The repo was deliberately placed **outside
  OneDrive** — OneDrive syncing `data/`, `runs/`, `.venv/` and `node_modules/`
  is slow and can corrupt locks, and the OneDrive path contains a space.

---

## 3. Environment — read before running anything

Windows 11. PowerShell is the shell. `uv` and `just` are **not on the default
PATH** in a fresh tool shell. Prefix every command:

```powershell
$env:Path = "$env:Path;C:\Users\akshy\AppData\Roaming\Python\Python314\Scripts;C:\Users\akshy\.local\bin"
Set-Location "C:\dev\alpha-lab"
$env:PYTHONHASHSEED="0"
```

Extras are **not** installed by `just setup`. The full suite needs them:

```powershell
uv sync --all-packages --extra gbm --extra tune --extra dl
```

`tune` is optuna (ADR 0010) and `dl` is torch (2.13.0, a ~120 MB download);
without either, those tests skip rather than fail. `just setup-dl` does the same
with `--all-extras`.

**Nothing may hold `.venv/Scripts/alpha.exe` while uv syncs.** A leftover
`alpha serve` from an earlier session made `uv sync` fail with
`failed to remove file ... (os error 32)` **after** it had already uninstalled
`alphacore`, leaving a venv that could not import the package. Stop the server
first.

- Python **3.12.13**, uv-managed, at
  `C:\Users\akshy\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\`.
  The system Python is 3.14 — do not use it.
- `PYTHONHASHSEED=0` is required by invariant I8 and cannot be set from inside
  Python. `just` exports it; direct `uv run pytest` does not, so set it yourself.
- Resolved stack is **much newer than the spec assumed**: numpy 2.5.1,
  pandas 3.0.5, scikit-learn 1.9.0, scipy 1.18.0, pydantic 2.13.4,
  pandera 0.32.1, duckdb 1.5.5, lightgbm 4.7.0, mypy 2.3.0, ruff 0.16.0.

### Three environment traps already hit

1. **Never edit source files with PowerShell string substitution.**
   `Get-Content -Raw` decodes UTF-8 as ANSI and `Set-Content` re-encodes it, so
   every `§` and `—` becomes mojibake plus a BOM. This happened **twice** —
   once in `features/builtins.py`, once in `README.md`, the second time *after*
   the lesson was written down. Use the Edit/Write tools.
   `tests/unit/test_source_encoding.py` now fails the build on a BOM or
   mojibake anywhere in the tree.
2. **Do not pass multi-line commit messages via PowerShell here-strings.**
   Quotes and `?` inside them break argument splitting and git receives the
   message as dozens of pathspecs. Write the message to a file and use
   `git commit -F <file>`.
3. **`pandas 3` defaults `DatetimeIndex` to microseconds.** §6.2 requires
   nanoseconds, so every index is coerced explicitly. Also: pandas refuses to
   `astype` an *empty* tz-naive index to a tz-aware one — build empty indexes
   with `dtype=` up front.

---

## 4. Commands

```bash
just verify
```

The gate, now in its final form: `fmt lint types test test-leak smoke e2e
budget`. **Must be green before a phase is called done.** ADR 0009's sequencing
is complete — `smoke` joined at Phase 10, `e2e` at Phase 12 — so `verify-full`
is now just an alias.

It covers both languages: ruff and Prettier, ruff and ESLint, mypy strict and
`tsc --noEmit`, pytest and Vitest, plus the smoke run, the Playwright golden
path, and the bundle budget.

```bash
just docs
```

Regenerates `docs/data-dictionary.md` from the feature registry. CI checks it
matches.

```bash
uv run alpha doctor
```

Reports whether the machine can produce reproducible runs.

```bash
uv run alpha data pull -c configs/experiments/smoke.yaml
```

Offline ingest from the committed fixture.

```bash
uv run alpha run -c configs/experiments/smoke.yaml
```

The golden path. Ingests, builds, trains, backtests, reports, and writes a
complete run directory under `runs/`. Takes about ten seconds.

```bash
just serve
```

The API on `127.0.0.1:8000`. Interactive docs at `/api/v1/docs`.

```bash
just client
```

Regenerates `docs/openapi.json` and `web/src/api/generated/schema.d.ts`. Both
are committed, and `tests/api/test_openapi_document.py` fails if either is stale.

Other CLI: `alpha version`, `alpha dev scratch-run`, `alpha data inspect
--dataset csv:SYNTH:1d`, `alpha data list`, `alpha config show -c <yaml>`,
`alpha features build -c <yaml>`, `alpha train -c <yaml>`,
`alpha backtest -c <yaml>`, `alpha report <run-id>`,
`alpha runs list|show|compare`, `alpha serve`, `alpha openapi`.

---

## 5. State of the build

**Phases 0–13 complete. Phase 14 is next.**

| Phase | Status |
|---|---|
| 0 Foundations | ✅ |
| 1 Data layer | ✅ |
| 2 Features | ✅ |
| 3 Labels and weights | ✅ |
| 4 Cross-validation | ✅ |
| 5 Models and leaderboard | ✅ |
| 6 Signals and risk | ✅ |
| 7 Backtest engine | ✅ |
| 8 Metrics and tearsheet | ✅ |
| 9 Explainability | ✅ |
| 10 Pipelines and CLI | ✅ |
| 11 API service | ✅ |
| 12 Web app | ✅ |
| 13 Sequence models (optional) | ✅ |
| **14 Packaging and docs** | **⬜ next** |

Last `just verify`: **1062 Python tests passed, 34 web tests passed, 75 leakage
tests passed**, coverage 89.33%, mypy strict clean on 104 source files, `tsc`
clean, the smoke run completing the golden path, 10 Playwright tests green, and
99 KB of initial JS against a 200 KB budget. Lighthouse 96 / 100 / 100.

Commits, newest first:

```
(Phase 13) feat: sequence models, fold-aware windows, the GBM ensemble
35be705 docs: record the Phase 12 commit in the handoff log
817348b feat: Phase 12 — the web application
884133c docs: record the Phase 11 commit in the handoff log
16c9b31 feat: Phase 11 — the API service, the job runner, and SSE
e6ba42d docs: record the Phase 10 commit in the handoff log
531ebd5 feat: Phase 10 — pipelines, the CLI, and the economic leaderboard
3a7aecb docs: add HANDOFF.md for continuing the build in a new session
c31f3aa feat: Phases 8 and 9 — honesty metrics, the tearsheet, and TreeSHAP
51ae04f feat: Phases 6 and 7 — signals, risk, and the event-driven backtester
0f5c3f5 feat: Phase 5 — models, baselines, and the out-of-fold leaderboard
34558c7 feat: Phase 4 — purged cross-validation
ca6a21b feat: Phases 2 and 3 — causal features and labels with sample weights
acbb74b feat: Phase 1 data layer — config schema, ingestion, lake, catalog, quality
750f522 feat: Phase 0 foundations — runtime, artifact store, tooling gate
```

### `alphacore` package map

```
runtime/    determinism, hashing, timers, logging, manifest, artifacts
config/     pydantic schema (§5 in full), loader, defaults.yaml
schemas/    bars.py (pandera), labels.py
data/       protocol, providers/{csv,yahoo}, calendars, lake, catalog, quality
features/   primitives (27), registry, builtins, transforms, pipeline, selection
labels/     horizon, triple_barrier, meta, weights, binarise
cv/         splits (+purge/embargo), walkforward, purged, cpcv
models/     protocol, base, linear, trees, baselines, oof, leaderboard, registry,
            windows (fold-aware windowing), sequence + _nets (torch), ensemble
signal/     thresholds (hysteresis), sizing
risk/       stops, limits (kill switch)
backtest/   costs, portfolio, ledger, engine, reference
eval/       classification, financial, deflated, bootstrap, pbo
report/     tearsheet, exporters, research_md
explain/    shap_runner
pipelines/  ingest, build_features, train, backtest, tune, full_run
```

```
alphaservice/
  cli.py            every verb in §15, plus `serve` and `openapi`
  errors.py         ServiceError, JobNotFoundError, JobStateError
  api/              app, settings, deps (ETag/immutable), problems (RFC 9457)
  api/routers/      system, datasets, runs, jobs
  jobs/             store.py (SQLite + rolling log), runner.py (process pool)
```

The pipeline stages compose in one direction: `build_features` produces a
`FeatureSet`; `train` turns it into a `TrainingResult` (leaderboard, per-model
OOF predictions, folds, backtests); `full_run` writes those into the §16 run
directory. `alpha run` is `full_run.run_experiment` and nothing else.

---

## 6. Conventions that are load-bearing

Break these and the project stops meaning anything.

- **Invariants are enforced, not documented.** A feature naming a label column
  raises at registration. Quantile label edges cannot be fitted without a
  training mask. Correlation pruning cannot run on the full sample. The
  backtester cannot execute on the bar that produced its signal, and
  `execution_lag_bars: 0` raises. Each of these **raises**; none warns.
- **Nothing returns `NaN`.** A metric that cannot be computed raises
  (§0.7). A silent NaN in a Sharpe ratio is a critical bug.
- **Every error is a typed exception from `alphacore.errors`.** No bare
  `ValueError`.
- **Every annualisation is explicit.** `periods_per_year` is a required
  argument, derived from the calendar, recorded in the manifest and the
  tearsheet. 252 is never hardcoded (§11.1).
- **No `print()`.** loguru only; ruff `T20` enforces it.
- **Docstrings** with `Args`/`Returns`/`Raises`, and the formula for anything
  financial. ruff `D` and `ANN` enforce it.
- **Determinism (I8):** canonical JSON everywhere, sorted iteration, seeded
  RNGs via `deterministic_rng(seed, *stream_labels)`, `n_jobs=1` on every GBM.
- **Tests in the same commit as the code.** TDD was used throughout: write the
  test, watch it fail for the right reason, then implement.
- **Every "canary" test needs a control.** A test that proves nothing is worse
  than no test. Examples already in the tree: the future-poison harness has a
  deliberately leaky centred-window control; the same-bar-execution canary has
  a perfect-foresight inverse canary; the splitter ban has positive samples;
  the scaler spy has a deliberately global fit; the window-purge test has the
  obvious positional windower, which straddles the gap and fails it.

### The phase gate

`tests/conftest.py` holds `CURRENT_PHASE` (currently **13**).
`tests/leakage/test_leakage_suite_complete.py` maps the seven §12.3 leakage
tests to the phase that unlocks each, and fails if one that should exist does
not. **Bump `CURRENT_PHASE` only when the phase's exit criteria pass.**

All seven §12.3 tests now exist and pass: `test_future_poison`,
`test_quantile_edges_per_fold`, `test_label_span_purged`,
`test_pipeline_fit_on_train_only`, `test_shuffled_labels_have_no_edge`,
`test_no_same_bar_execution`, `test_perfect_foresight_sanity`.

---

## 7. Deviations from the spec already taken

All are recorded in `PROGRESS.md` with reasoning. The short list:

1. **ADR numbering conflict.** §2 assigns the seven stack swaps to ADRs
   0001–0007; §14.2 separately calls the job runner "ADR 0006". Kept §2's
   explicit range; the job-runner ADR is **reserved as 0008** for Phase 11.
   ADR 0009 is the phase-gated `verify`.
2. **`verify` excluded `smoke` until Phase 10, and excludes `e2e` until Phase
   12** (ADR 0009). `smoke` is in the gate now.
3. **No `configs/default.yaml`.** The packaged
   `alphacore/config/defaults.yaml` is the only baseline; two files claiming to
   be the defaults is a drift bug.
4. **`binance` and `alphavantage` providers not implemented.** Requesting one
   raises a message naming the gap rather than falling back.
5. **GBMs live in an optional `gbm` extra**, torch in `[dl]`, so
   `import alphacore` stays light. All are installed locally and fully tested.
6. **TreeSHAP comes from the boosting libraries, not the `shap` package.**
   `shap` pulls `numba` → `llvmlite`, which will not build on Python 3.12 with
   NumPy 2.5. LightGBM/XGBoost/CatBoost expose exact TreeSHAP natively — same
   algorithm, no dependency, additivity to ~1e-15.
7. **The splitter ban parses the AST rather than grepping.** A text search also
   matches the docstrings explaining the ban, and a check that flags its own
   documentation gets silenced.
8. **Three-class labels must be reduced explicitly.** Triple-barrier emits
   `{-1, 0, +1}`; models are binary. `labels.to_binary(..., method="direction"
   | "touch")` is a required, visible step.
9. **⚠️ One exit criterion substituted, not met.** §15 Phase 8 asks DSR to
   "reproduce the published worked example". The paper was not available, and
   transcribing figures from it would be fabricating a citation. Instead PSR is
   checked against an independent transcription of the §11.2 formula inside the
   test, and DSR is pinned by its analytic properties. **This catches an
   implementation error but not a misreading of the formula.** If the paper's
   worked figures can be supplied, add them as a golden fixture in
   `tests/golden/`.
10. **Models persist as `model.pkl`**, not the `model.{txt,json,pt}` §16 names.
    The pickled object is the whole pipeline — winsoriser, scaler, estimator,
    calibrator — and a booster saved without its calibrator produces scores
    that are not probabilities.
11. **`alpha features build|train|backtest` write nothing.** Only `alpha run`
    produces a run directory. A directory with `features.json` and no
    `tearsheet.json` would satisfy §16's filenames but not its meaning.
12. **Optuna lives behind the `tune` extra** and is imported lazily (ADR 0010).
13. **Job cancellation is cooperative, not a kill** (ADR 0008). A cancel lands at
    the next phase boundary, so the run directory is finalised rather than
    killed mid-write.
14. **One schemathesis check is excluded** — `positive_data_acceptance`, because
    an opaque cursor's validity cannot be expressed in JSON Schema. The 400 it
    flags is asserted directly in `tests/api/test_api_runs.py`.
15. **Sequence models must be told the bar grid** before `fit`, and raise if they
    were not (ADR 0011). `fit_predict_oof` does it; hand-rolled code must call
    `model.set_bar_grid(features.index)` itself.
16. **Sequence models train on CPU unless a device is named** (ADR 0012), because
    accelerator reductions are not bit-reproducible and I8 is load-bearing.
    `device: auto` still does what §10.3 describes, and warns.

---

## 8. Known gaps entering Phase 14

- **Tuned parameters never reach the GBM builders.**
  `build_model(name, seed=..., **params)` passes them as flat keywords, but
  `make_lightgbm|xgboost|catboost` take a `params` mapping, so
  `build_model("lightgbm", seed=1, max_depth=4)` raises `TypeError` —
  and `tune_model`'s objective catches only `(ModelError, PipelineError)`, so
  tuning any GBM kills the study. Green since Phase 10 only because
  `tests/integration/test_tuning.py` tunes `logistic`, whose builder does take
  flat kwargs. Fix it with a failing test first; `sequence.py`'s builders take a
  mapping too, so whichever shape wins must cover them.
- **Jobs do not survive an API restart.** Nothing resumes them; the run
  directory will not claim to be `completed`. ADR 0008 names the design.
- **No authentication anywhere.** `alpha serve` binds loopback and warns if told
  to bind anything else.
- `GET /runs/{id}/shap/explain?ts=` is not implemented — it needs a fitted model
  reloaded per request, which wants a cache.
- `fill_price: next_vwap` is accepted by config but the engine always fills at
  the open.
- The backtest engine is **single-symbol**. `primary_symbol` raises on a config
  naming more than one rather than silently trading the first.
- Clustered MDA (§7.3) and reliability diagrams (§10.2) are not built.
- **`shap_summary.parquet` only appears when the selected model is a tree
  model.** The smoke config selects `baseline_buyhold`, so the smoke run has no
  SHAP artifact. Any API contract test that requires one needs a config whose
  winner is a GBM.
- **The trade blotter counts a partial reduction as a completed trade.** That
  is a deliberate definition (each row is one realising event, with its own
  entry price and P&L), but it means `n_trades` under `vol_target` sizing is a
  count of realisations, not of round trips. Worth stating plainly wherever the
  UI shows it.
- **Sequence models have no registered Optuna search space**, so `alpha tune`
  skips them rather than searching one nobody declared.
- **The window builder is single-symbol**, following the backtester. A
  cross-sectional tensor would need a symbol axis and a per-symbol grid.
- **`just verify` does not require the `dl` extra.** Every sequence test is
  `pytest.importorskip`-gated, so a fresh clone is green without torch — and a
  machine without it is not testing Phase 13 at all.

---

## 9. Phase 14 — what to do next

From §15:

> Deliverables: multi-stage Dockerfile + compose, GitHub Actions CI (lint, type,
> test, coverage gate 85% on `alphacore`, web typecheck + build + Playwright,
> docker build), README with screenshots and a 5-minute quickstart,
> `docs/methodology.md`, mermaid architecture diagrams rendered to SVG, three
> example experiment configs, the generated research report for one real run.
>
> **Exit:** `docker compose up` serves the app with the sample dataset on a
> clean machine; CI green; a stranger can go from `git clone` to a rendered
> tearsheet in under 10 minutes following only the README.

Things Phase 13 leaves that bear on it:

- **The image should not carry torch.** The `dl` extra is ~2 GB installed for a
  model family the sample config does not use. Build the default image from
  `uv sync --all-packages --extra gbm`, and if a deep-learning image is wanted,
  make it a second stage rather than the default.
- **CI must install the extras deliberately**, or the sequence and tuning tests
  silently skip and the coverage gate drops. Whatever the workflow runs, assert
  the skip count is zero on the job that claims to cover Phase 13.
- **Three example configs** are the deliverable, and `smoke.yaml`,
  `spy_daily_tb.yaml` and `btc_1h_meta.yaml` already exist. A fourth naming a
  sequence candidate would be the natural place to show `lstm` on a leaderboard
  outside the test suite.
- **The e2e web server path was wrong** and hid behind `reuseExistingServer`
  until a stale server was killed (fixed in `web/playwright.config.ts`). CI has
  no stale server, so any similar path assumption fails there first.

Run the API with `just serve`; it binds `127.0.0.1:8000` and the Vite dev origin
is already in the CORS allow-list.

---

## 10. Working agreement the user expects

- **One phase at a time.** Do not run ahead. Stop at the phase boundary and
  report.
- **`just verify` green before claiming done**, and paste the actual output.
- **Update `PROGRESS.md`** every phase: what was built, exit criteria with
  evidence, deviations with reasoning, and a "what surprised us" section. That
  last section has repeatedly been the most useful part.
- **Write an ADR** for any deviation from the spec.
- **Report honestly.** If an exit criterion is substituted rather than met, say
  so plainly and say what the substitute does *not* cover — see §7.9 above for
  the precedent.
- **Commit at phase boundaries**, message via `git commit -F <file>`.

The user's style preference for chat responses is terse. Code, commits and
documentation stay in full prose.
