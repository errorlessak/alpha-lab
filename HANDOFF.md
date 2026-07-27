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

The gate. Runs `fmt lint types test test-leak`. **Must be green before a phase
is called done.** Per ADR 0009, `smoke` joins it at Phase 10 and `e2e` at
Phase 12; `verify-full` already names the final seven-recipe form.

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

Other CLI: `alpha version`, `alpha dev scratch-run`, `alpha data inspect
--dataset csv:SYNTH:1d`, `alpha data list`, `alpha config show -c <yaml>`.

---

## 5. State of the build

**Phases 0–9 complete. Phase 10 is next.**

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
| **10 Pipelines and CLI** | **⬜ next** |
| 11 API service | ⬜ |
| 12 Web app | ⬜ |
| 13 Sequence models (optional) | ⬜ |
| 14 Packaging and docs | ⬜ |

Last `just verify`: **864 passed, 3 deselected, coverage 89.75%**, leakage suite
**71 passed, 0 skipped**, mypy strict clean on 81 source files.

Commits, newest first:

```
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
models/     protocol, base, linear, trees, baselines, oof, leaderboard, registry
signal/     thresholds (hysteresis), sizing
risk/       stops, limits (kill switch)
backtest/   costs, portfolio, ledger, engine, reference
eval/       classification, financial, deflated, bootstrap, pbo
report/     tearsheet, exporters, research_md
explain/    shap_runner
pipelines/  ingest  ← only ingest so far; the rest is Phase 10
```

`alphaservice` has `cli.py` only. `api/` and `jobs/` are Phase 11.

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
  the scaler spy has a deliberately global fit.

### The phase gate

`tests/conftest.py` holds `CURRENT_PHASE` (currently **9**).
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
2. **`verify` excludes `smoke` until Phase 10, `e2e` until Phase 12** (ADR
   0009).
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

---

## 8. Known gaps entering Phase 10

- `pipelines/` has **only** `ingest.py`. `build_features.py`, `train.py`,
  `backtest.py`, `full_run.py` do not exist.
- Nothing yet writes `tearsheet.json`, `folds.json`, `features.json`,
  `research.md`, or the parquet series into a run directory. The pieces all
  exist (`report/exporters.py`, `report/tearsheet.py`, `cv/splits.py`
  serialisation) but are unwired.
- The leaderboard's **economic columns are `None`** and every row carries a
  `NO_BACKTEST` flag. Phase 10 wires the backtester in and fills
  `oos_net_sharpe`, `sharpe_ci`, `dsr`, `psr`, `break_even_bps`,
  `max_drawdown`; ranking then switches from IC t-stat to net Sharpe.
- **Optuna tuning (§10.4) is not built.** The objective must be out-of-fold and
  economic — run the inner walk-forward, generate signals, run the costed
  backtest, return net Sharpe. `n_trials` must flow into the DSR haircut
  automatically.
- `fill_price: next_vwap` is accepted by config but the engine always fills at
  the open.
- The backtest engine is **single-symbol**.
- Clustered MDA (§7.3) and reliability diagrams (§10.2) are not built.

---

## 9. Phase 10 — what to do next

From §15:

> **Deliverables:** `alpha data pull|inspect`, `alpha features build`,
> `alpha train`, `alpha backtest`, `alpha run` (full golden path),
> `alpha report`, `alpha runs list|show|compare`. Rich console output.
>
> **Exit:** `alpha run -c configs/experiments/smoke.yaml` completes end to end
> and writes a complete run directory. **This is the integration gate for
> everything before it.**

The run directory contract is in §16 of `ARCHITECTURE.md` — `manifest.json`,
`config.resolved.yaml`, `tearsheet.json`, `quality.json`, `features.json`,
`folds.json`, the parquet series, `leaderboard.csv`, `research.md`,
`models/`, `logs.jsonl`.

Also due at Phase 10: add `smoke` to the `verify` recipe (ADR 0009 says so),
and add §12.4's determinism test — run `smoke.yaml` twice, in the same session
and in a fresh subprocess, and assert `tearsheet.json` is byte-identical after
stripping `created_at`, `duration_s` and `host`.

`tests/integration/test_smoke_leaderboard.py` already does bars → features →
labels → folds → leaderboard in-process. It is the closest thing to a template
for `pipelines/full_run.py`.

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
