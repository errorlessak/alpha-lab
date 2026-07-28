# Build log

One entry per phase from §15 of [ARCHITECTURE.md](ARCHITECTURE.md). A phase is
done when its exit criteria pass, not when the code looks finished.

| Phase | Name | Status |
|---|---|---|
| 0 | Foundations | ✅ Done — 2026-07-27 |
| 1 | Data layer | ✅ Done — 2026-07-27 |
| 2 | Features | ✅ Done — 2026-07-27 |
| 3 | Labels and weights | ✅ Done — 2026-07-27 |
| 4 | Cross-validation | ✅ Done — 2026-07-27 |
| 5 | Models and leaderboard | ✅ Done — 2026-07-27 |
| 6 | Signals and risk | ✅ Done — 2026-07-27 |
| 7 | Backtest engine | ✅ Done — 2026-07-27 |
| 8 | Metrics and tearsheet | ✅ Done — 2026-07-27 |
| 9 | Explainability | ✅ Done — 2026-07-27 |
| 10 | Pipelines and CLI | ✅ Done — 2026-07-28 |
| 11 | API service | ✅ Done — 2026-07-28 |
| 12 | Web app | ⬜ Not started |
| 13 | Sequence models (optional) | ⬜ Not started |
| 14 | Packaging and docs | ⬜ Not started |

---

## Phase 0 — Foundations ✅

**Completed:** 2026-07-27
**`CURRENT_PHASE`** (in `tests/conftest.py`): `0`
**Recipes in `verify`:** `fmt lint types test test-leak` (ADR 0009)

### Built

- **uv workspace** with two members, `alphacore` and `alphaservice`, and a
  committed `uv.lock`. Python pinned to 3.12.
- **`alphacore.errors`** — the typed exception hierarchy §18 requires, with
  `LeakageError` carrying the explicit instruction that it must never be
  downgraded to a warning.
- **`runtime/hashing.py`** — canonical JSON (sorted keys, ASCII, no
  insignificant whitespace, `NaN`/`Infinity` rejected) and blake2b digests over
  values, single files, and file sets. Sorting is applied to sets and to file
  lists so that iteration order can never reach a digest.
- **`runtime/determinism.py`** — `set_global_seed`, `deterministic_rng` for
  per-stream generators, `seed_sequence` for APIs that insist on int seeds, and
  a `PYTHONHASHSEED` guard that fails rather than pretending.
- **`runtime/timers.py`** — `Timer`, `timed`, and `PhaseTimings`, which records
  a phase's duration even when the phase raises.
- **`runtime/logging.py`** — loguru with a console sink and a JSON-lines sink
  into `runs/{run_id}/logs.jsonl`. `print()` banned repo-wide via ruff `T20`.
- **`runtime/manifest.py`** — `RunManifest`, run ids in the documented
  `2026-07-27T14-22-01Z_spy-daily-tb_a3f91c` shape, git revision capture with a
  `-dirty` marker, and a tracked-package version snapshot.
- **`runtime/artifacts.py`** — `ArtifactStore`, `RunDirectory`, and
  `run_context`, enforcing write-once semantics and finalising a crashed run as
  `failed` with its traceback.
- **`alphaservice.cli`** — `alpha version`, `alpha doctor`, `alpha dev
  scratch-run`.
- **Tooling** — ruff (format + lint, broad rule set), mypy strict, pytest with
  a coverage gate, import-linter contracts, pre-commit, GitHub Actions CI, and
  a cross-platform justfile.
- **ADRs 0001–0007 and 0009**, plus `docs/decisions/README.md` as the index.

### Exit criteria

| Criterion | Result |
|---|---|
| `just verify` green | ✅ ruff format, ruff check, import-linter, mypy strict, 122 passed / 7 skipped, leakage suite green |
| Coverage gate | ✅ 99.58% on `alphacore`, gate is 85% |
| `alpha --help` works | ✅ |
| Throwaway run dir with a valid manifest | ✅ `alpha dev scratch-run` writes `manifest.json`, `logs.jsonl`, `config.resolved.yaml`, `scratch.json`; manifest validates and reports `status: completed` |

### Deviations, all recorded

1. **`verify` excludes `smoke` until Phase 10 and `e2e` until Phase 12** —
   ADR 0009. §15's own Phase 0 criterion assumes this. `verify-full` names the
   final seven-recipe form from day one.
2. **`setup` does not pass `--all-extras`** — ADR 0006. `--all-extras` pulls
   PyTorch, a multi-gigabyte download nothing before Phase 13 needs, and §10.3
   requires the base install to work without it. `just setup-dl` is the opt-in.
3. **ADR numbering** — §2 assigns the seven stack swaps to 0001–0007 while
   §14.2 separately calls the job runner "ADR 0006". Kept §2's explicit range;
   the job-runner ADR is reserved as 0008 for Phase 11. Reasoning in
   `docs/decisions/README.md`.
4. **Config pydantic schema deferred to Phase 1** — §15's Phase 0 deliverables
   do not list it. `RunManifest.start` takes any canonically serialisable
   mapping, which is exactly what a pydantic `model_dump` produces, so the
   signature is stable when §5's schema lands.
5. **Two modules added under `runtime/`** beyond the four in §3's tree —
   `hashing.py` and `timers.py`. §3 lists "seeds, logging, timers, run manifest,
   hashing" as the package's responsibilities, so these are the same scope split
   along its natural seams rather than new scope.
6. **`justfile` uses `npm --prefix web` rather than `cd web &&`** — each `just`
   recipe line runs in its own shell, and `&&` is not available in Windows
   PowerShell. `set windows-shell` keeps `sh` on every other platform.

### What surprised us

- **A property-style test caught a real determinism bug on its first run.**
  `deterministic_rng(seed, *stream)` derived its entropy by truncating the
  canonical JSON of the stream labels to eight bytes. `["cpcv",0]` and
  `["cpcv",1]` agree in their first eight bytes, so every fold in a CPCV loop
  would have drawn *identical* random numbers — silently, and in exactly the
  place where independent streams matter most. Fixed by hashing before
  truncating; `test_deterministic_rng_streams_sharing_a_prefix_are_independent`
  is the regression test.
- **An empty leakage suite is a false green.** `pytest tests/leakage` on an
  empty directory exits 5 ("no tests collected"), and any wrapper tolerant of
  that would have stayed green through Phase 11 — and kept staying green if a
  leakage test were later deleted. `tests/leakage/test_leakage_suite_complete.py`
  now holds all seven tests from §12.3 against the phase that unlocks each, so
  the suite has something real to assert from Phase 0 onward.
- **`PYTHONHASHSEED` cannot be fixed from inside Python.** CPython reads it at
  startup. The only honest options were to fail loudly or to stop claiming I8,
  so `require_pinned_hash_seed()` raises, the justfile and CI export it, and
  `alpha doctor` reports it.
- **Ruff's `RUF100` is a good forcing function.** Several `# noqa` codes copied
  in from habit (`S603`, `S607`, `NPY002`) referred to rules that were not
  enabled. Ruff flagged every one, which is how a `noqa` graveyard is avoided.
- **The environment resolved much newer than expected** — pandas 3.0.5, numpy
  2.5.1, mypy 2.3.0. Nothing broke, but it is a concrete argument for the
  lockfile in ADR 0006 and for `_TRACKED_PACKAGES` in the manifest.

### Deferred to Phase 1

- `alphacore.config` — pydantic v2 schema, YAML loader, `defaults.yaml`,
  generated `configs/schema.json`
- `configs/experiments/smoke.yaml` and the other experiment configs
- The `data/` lake layout and the duckdb catalog

---

## Phase 1 — Data layer ✅

**Completed:** 2026-07-27
**`CURRENT_PHASE`** (in `tests/conftest.py`): `0` — unchanged. Phase 1 unlocks
no leakage test from §12.3; the first is `test_future_poison`, at Phase 2.
**Recipes in `verify`:** `fmt lint types test test-leak` (ADR 0009)

### Built

- **`alphacore.config`** — the full §5 schema as frozen pydantic v2 models with
  `extra="forbid"`, the packaged `defaults.yaml`, and a loader layering
  `defaults → experiment YAML → ALPHA__* env → --set overrides`. Cross-field
  validation catches the errors that would otherwise surface as nonsense
  results: `end <= start`, a short threshold above the long one, a negative
  feature lag, `execution_lag_bars: 0`, CPCV holding out every group.
- **`alphacore.schemas.bars`** — the canonical bar schema in pandera, plus
  `normalise_bars` (lowercase columns, UTC nanosecond index, default
  `adj_factor`, sort, keep-last on duplicate timestamps) and error formatting
  that names the failed check and the offending timestamps.
- **`alphacore.data.calendars`** — expected-bar grids over `exchange_calendars`,
  daily and intraday, equities and 24/7. Bars are stamped with their **left
  edge**; the reasoning is in the module docstring and it is what makes I2
  checkable.
- **`alphacore.data.providers`** — the `BarProvider` protocol, a jittered
  exponential-backoff `retry`, the `csv` provider, and the `yahoo` provider
  (unadjusted OHLC plus a separate `adj_factor`, published history limits, lazy
  `yfinance` import).
- **`alphacore.data.lake`** — parquet partitioned
  `raw/{provider}/{symbol}/{interval}/year=YYYY/`, ZSTD, with a `content_hash`
  over the sorted file set.
- **`alphacore.data.catalog`** — the duckdb catalog, one row per dataset,
  pinned to UTC.
- **`alphacore.data.quality`** — the §6.4 report: missing bars against the
  session grid, longest consecutive gap, unexpected bars, duplicates, zero
  volume, OHLC violations, robust return outliers, rows per year, and a
  PASS/WARN/FAIL verdict from the configured thresholds.
- **`alphacore.pipelines.ingest`** — fetch, validate, assess, write, catalogue.
- **CLI** — `alpha data pull|inspect|list` and `alpha config show`.
- **Fixtures** — `samples/SYNTH_1d.csv`, 2,516 synthetic daily bars on the real
  NYSE grid (198 KB), plus `samples/make_fixtures.py` that regenerates it.
- **Configs** — `smoke.yaml` (offline, CI), `spy_daily_tb.yaml`,
  `btc_1h_meta.yaml`.

### Exit criteria

| Criterion | Result |
|---|---|
| Pull produces validated parquet + catalog row + quality report | ✅ `alpha data pull -c configs/experiments/smoke.yaml` writes 10 year-partitioned parquet files, `quality.json`, and a catalog row. 2,516 bars, quality `PASS`, `content_hash 87bc163ba30e2720c2b2e84220acc666` |
| Schema violations raise with a useful message | ✅ e.g. `ohlc_ordering: low <= min(open, close) <= max(open, close) <= high violated: 1 failing row(s) first at 2024-01-05 00:00:00+00:00` |
| Offline tests pass with no network | ✅ everything runs through the `csv` provider and the committed fixture; the three live Yahoo tests are `@pytest.mark.network` and deselected by default |
| `just verify` green | ✅ 282 passed, 7 skipped, 3 deselected; coverage 96.19% |

### Deviations and judgement calls

1. **A Phase 0 bug fixed here.** The import-linter contract listed `duckdb`
   among the modules `alphacore` may not import — but §3 puts `catalog.py`
   inside `alphacore`, and duckdb is exactly what it is for. The contract was
   wrong, not the design. Removed.
2. **OHLC ordering lives in both the schema and the quality report.** §6.2
   lists it as a schema check; §6.4 exposes `fail_on_ohlc_violation` as a
   config threshold. Both cannot hold with a single strict schema, so there are
   two: `RAW_BARS_SCHEMA` (strict, the default path) and
   `STRUCTURAL_BARS_SCHEMA` (no ordering check), selected by
   `validate_bars(..., check_ohlc_ordering=False)`. Only `--allow-dirty-data`
   takes the relaxed path, and the run is stamped when it does.
3. **No `configs/default.yaml`.** §3's tree lists both it and
   `alphacore/config/defaults.yaml`. Two files claiming to be the baseline is a
   drift bug, so the packaged one is the only baseline; `configs/` holds
   experiments. Conservative reading per §0.6.
4. **`binance` and `alphavantage` are not implemented.** §6.1 names four
   providers; §15's Phase 1 exit criteria name two. Requesting an unimplemented
   one raises a `ProviderError` naming the gap rather than silently falling back
   — a run whose manifest says `binance` must not have been fed by Yahoo.
   `btc_1h_meta.yaml` is committed anyway, because it pins the crypto calendar
   and annualisation choices for review before the provider exists.
5. **Fixtures are synthetic.** Committing real vendor history raises a
   licensing question and goes stale. `SYNTH` is a seeded geometric random walk
   with volatility clustering, aligned to real NYSE sessions — redistributable,
   reproducible, and by construction containing no edge, which makes it the
   right subject for §19's honest headline result.
6. **`quality.json` is written beside the dataset**, not only returned. It sits
   at the dataset root rather than inside a year partition, so it cannot perturb
   `content_hash` — the hash describes the bars, not the verdict on them. There
   is a test for exactly that.
7. **`samples/make_fixtures.py` is a file §3's tree does not list.** Generating
   a committed fixture from an uncommitted script would make it unreproducible.

### What surprised us

- **pandas 3 changed the default datetime resolution to microseconds.** §6.2
  specifies `datetime64[ns, UTC]`. Every index is now coerced explicitly rather
  than inheriting the pandas default, because a resolution that silently
  changes with a dependency release is exactly the kind of thing that makes two
  runs disagree for no visible reason.
- **pandas refuses to `astype` an *empty* tz-naive index to a tz-aware one.**
  Four tests failed on it at once. Empty indexes are now constructed with the
  dtype up front. A small thing, but it only showed up because the suite tests
  the empty case — which is the case a real pull hits when a symbol is
  delisted.
- **duckdb renders `TIMESTAMPTZ` in the session timezone.** The catalog read
  back `2015-01-02T05:30:00+05:30` on a machine in Kolkata. The instant was
  right and the string was wrong — and that string is what the API and UI
  would have shown. Connections now `SET TimeZone = 'UTC'`, with two
  regression tests.
- **pandera reports a dataframe-wide check failure once per column.** One bad
  bar in a seven-column frame was reported as "7 failing rows". Failure cases
  are now de-duplicated by row before being counted.
- **`exchange_calendars` returns tz-naive sessions but tz-aware intraday
  indexes**, and `trading_index` returns an `IntervalIndex` unless
  `intervals=False`. Both were found by probing the library rather than by
  reading its docs, and both are pinned by tests now.
- **The Phase 0 `.gitignore` silently excluded the entire data layer.** It
  carried an unanchored `data/`, which matches *any* directory of that name at
  any depth — including `packages/alphacore/src/alphacore/data/`. Ten source
  files were quietly absent from `git status` and would have been missing from
  the repository while every local test passed. Caught by reading the staged
  file list rather than by any check; now anchored as `/data/` and `/runs/`,
  with the reasoning in the file so nobody un-anchors it later.

### Deferred to later phases

- `binance` and `alphavantage` providers (§6.1)
- `configs/schema.json` generated from the pydantic models for editor
  autocomplete (§5) — wants a CLI command, arriving with the rest of the config
  tooling
- Multi-symbol universes: ingestion loops over symbols, but nothing downstream
  consumes more than one yet

---

## Phase 2 — Features ✅

**Completed:** 2026-07-27
**`CURRENT_PHASE`:** `2` at completion — `test_future_poison` is now required
**Method:** test-driven throughout. Every module here had its tests written and
watched fail before a line of implementation existed.

### Built

- **`features/primitives.py`** — all 27 primitives from §7.2 as pure numpy
  functions: SMA, EMA, WMA, HMA, RSI, MACD, ROC, momentum, PPO, true range,
  ATR, realised vol, Bollinger, rolling std, Parkinson, Garman-Klass, OBV,
  rolling VWAP and distance, CMF, volume ratio, dollar volume, log and simple
  returns, rolling statistics, z-score, percentile rank, and fractional
  differencing.
- **`features/registry.py`** — `FeatureSpec`, the causality contract, and the
  data-dictionary generator. Registering a feature that names a label column as
  an input raises `LeakageError` (§4, I5) — the leak is unrepresentable, not
  merely discouraged.
- **`features/builtins.py`** — registration adapters with declared lookbacks.
- **`features/transforms.py`** — lags and warm-up dropping, both causal by
  construction.
- **`features/pipeline.py`** — spec expansion, computation, lagging, warm-up
  dropping, and a `FeatureManifest` recording every column's provenance.
- **`features/selection.py`** — pairwise and clustered correlation pruning,
  both requiring an explicit `fit_mask`.
- **`data/calendars.py`** — `periods_per_year` derived from the calendar,
  never hardcoded (§11.1).
- **`docs/data-dictionary.md`** — generated, with a CI check that it matches.

### Exit criteria

| Criterion | Result |
|---|---|
| Golden indicator tests to `1e-9` | ✅ 8 frozen-value tests against `tests/golden/indicators/reference.csv` |
| `test_future_poison` over the whole registry | ✅ 27 features parametrised, all pass; the leaky control is caught |
| Data dictionary regenerates identically | ✅ `test_the_committed_file_matches_the_registry` |
| `just verify` green | ✅ 458 passed, coverage 93.56% |

### Deviations

1. **The reference oracle is a naive transcription, not vendor values.** §7.2
   asks for "hand-verified expected values from the original published
   definitions". Vendor exports are not that — they bake in the vendor's own
   smoothing conventions, are not redistributable, and disagreeing with one
   tells you nothing about who is right. Instead:
   `tests/golden/reference_indicators.py` implements each formula as an
   explicit loop, a genuinely different algorithm from the vectorised one, and
   the values are additionally frozen into a committed CSV so editing both
   implementations the same wrong way still fails. Hand arithmetic covers the
   closed-form cases (SMA, WMA, EMA seeding, ROC, true range, log returns)
   where the expected value can be read off the test.
2. **Two modules added under `features/`** beyond §3's list: `builtins.py`
   (registration adapters, kept apart so the primitives stay pure array
   functions) and `__main__.py` (see below).
3. **Pruning is not a pipeline step.** §7.3 says the correlation matrix is
   built on fold 0's training rows; the pipeline has no folds. `build_features`
   therefore computes and lags, and `prune_correlated` is applied inside the
   fold with an explicit mask. Requesting pruning without one raises
   `LeakageError`.
4. **`max-args` raised to 8** in the ruff config, with the reasoning in
   `pyproject.toml`: several functions take many keyword-only provenance
   fields, and bundling them into an options object would hide exactly what a
   reviewer needs to see at the call site.

### What surprised us

- **`python -m alphacore.features.registry` built an empty registry.** Running
  a module with `-m` executes it a second time under the name `__main__`, so
  that copy got its own empty `_REGISTRY` while the built-ins registered into
  the first. The symptom was a data dictionary with a header and no features —
  which reads like a formatting bug and is not. Fixed with a dedicated
  `__main__.py`; the reasoning is in its docstring.
- **Fractional differencing is expensive in sample size.** At `d=0.4` the
  truncated weight expansion is 282 bars at `thresh=1e-4` and **1,458 bars** at
  `1e-5` — nearly six years of daily data, gone to warm-up. The spec's example
  config uses `1e-4`; the registry declares the real lookback and the data
  dictionary prints it, because discovering it after a run has silently lost a
  decade of history is much worse.
- **A PowerShell `-replace` on a UTF-8 source file corrupted it.**
  `Get-Content -Raw` decoded the bytes as ANSI and `Set-Content` re-encoded
  them as UTF-8, so every section sign and em dash became a two-character
  sequence, plus a BOM on the front. It happened twice — once in
  `features/builtins.py` and once, after the lesson was written down, in
  `README.md`. Both repaired, and the check is now automated:
  `tests/unit/test_source_encoding.py` fails on a BOM or a mojibake sequence
  anywhere in the tree. Edit source files with editing tools, not with shell
  string substitution.
- **Two contract tests caught wrong assumptions rather than wrong code.**
  `dollar_volume` and `obv` genuinely have no warm-up, and the alphabetical
  survivor of a correlated pair is deliberate — both tests were asserting
  something untrue about correct implementations.

---

## Phase 3 — Labels and weights ✅

**Completed:** 2026-07-27
**`CURRENT_PHASE`:** `3` — `test_quantile_edges_per_fold` is now required

### Built

- **`schemas/labels.py`** — the `(t_start, t_end, label, realised_return,
  weight)` contract, validated. `t_end` is the column the CV purger consumes
  (§8.2); a frame without it cannot be purged.
- **`labels/horizon.py`** — `horizon_return`, `horizon_sign` with a dead zone
  that drops rather than labels flat, and `horizon_quantile` with **per-fold**
  bin edges.
- **`labels/triple_barrier.py`** — ATR-scaled barriers evaluated on intrabar
  high and low, with the stop winning any same-bar tie (§12.1). `t_end` is the
  touch bar, not the vertical barrier, so purging blocks the right number of
  bars.
- **`labels/meta.py`** — meta-labelling: the primary rule picks the side, the
  model answers only whether to take the trade.
- **`labels/weights.py`** — uniqueness from average inverse concurrency
  (computed with a difference array, one pass) times linear time decay in
  cumulative uniqueness.
- **`labels/__init__.py`** — `build_labels`, dispatching on config and
  attaching weights, class balance, and mean uniqueness.

### Exit criteria

| Criterion | Result |
|---|---|
| Triple barrier verified against a hand-computed 20-bar fixture | ✅ `test_triple_barrier_hand_computed_twenty_bar_fixture`, plus explicit upper-touch, lower-touch, vertical-expiry, and same-bar-tie cases |
| Uniqueness weights sum correctly on a synthetic overlapping set | ✅ non-overlapping labels weigh 1.0; a uniquely-spanning label outweighs a crowded one; weights average to 1.0 |
| Per-fold quantile test passes | ✅ `tests/leakage/test_quantile_edges_per_fold.py` — 6 tests including a global-fit control |
| `just verify` green | ✅ 504 passed, 5 skipped, coverage 91.80% |

### Deviations

1. **`horizon_quantile` requires a `fit_mask` and raises without one.** §8.1
   says edges must be fitted per fold; making the parameter required means
   there is no API path that fits globally by accident. Same discipline as
   correlation pruning.
2. **The `build_labels` dispatch has no `else` branch.** `LabelsConfig.method`
   is a `Literal` over exactly four values, so mypy proves the chain
   exhaustive. Adding a fifth without adding a branch becomes a type error —
   a stronger guard than the runtime `raise` it replaced.

### What surprised us

- **The dead zone should drop, not label flat.** §8.1 says
  `|r| < min_return → drop` for `horizon_sign`, and the reason is worth
  stating: a move too small to trade carries no directional information, and
  labelling it "flat" teaches the model that noise has a sign. Triple-barrier
  vertical expiries are different — those are genuine "nothing happened"
  observations and stay as class 0.
- **Time decay is linear in *cumulative uniqueness*, not in bar count.** A
  stretch of heavily overlapping labels therefore ages the schedule less than a
  stretch of independent ones, which is right: it carries less information. The
  first observation lands at `decay + (1-decay)/n` rather than exactly `decay`,
  which cost one wrong test expectation before it was understood.
- **Sample weights get a `tiny` floor rather than zero.** A zero weight removes
  an observation from the fit without recording that it went; a floored one
  stays visible in the class balance and the manifest.

### Deferred to later phases

- Mean Decrease Accuracy and clustered feature importance (§7.3) — both need a
  fitted model, so they land with Phase 5
- Winsorisation (§7.3) — belongs inside the sklearn pipeline so it is fitted on
  train rows only (I3), which is Phase 5
- Wiring `build_features` and `build_labels` into a run directory — Phase 10

---

## Phase 4 — Cross-validation ✅

**Completed:** 2026-07-27
**`CURRENT_PHASE`:** `4` — `test_label_span_purged` is now required

### Built

- **`cv/splits.py`** — the `Split` record and the shared purge/embargo
  machinery. Every scheme routes through `build_split`, so there is exactly one
  implementation of I4 rather than three.
- **`cv/walkforward.py`** — anchored and rolling, chronological, the primary
  scheme and the one the Fold Strip will draw.
- **`cv/purged.py`** — contiguous test blocks with purging and embargo.
- **`cv/cpcv.py`** — all `C(N, k)` combinations, giving the multiple backtest
  paths PBO needs (§11.3).
- **`tests/unit/test_no_banned_splitters.py`** — the §9 ban on
  `train_test_split` and plain `KFold`.

### Exit criteria

| Criterion | Result |
|---|---|
| `test_label_span_purged` for all three schemes across randomised horizons | ✅ one named test over four configurations, plus three hypothesis-driven variants sweeping horizon 1–30, fold counts, window geometries, anchored/rolling, and embargo 0–10% |
| Splits round-trip through JSON | ✅ `test_splits_round_trip_through_json`, and the serialisation is canonical so `folds.json` is byte-comparable between runs |
| CI ban on `train_test_split` / `KFold` | ✅ `test_no_banned_splitter_appears_in_library_code` |
| `just verify` green | ✅ 558 passed, 4 skipped, coverage 91.14%; leakage suite 49 passed |

### Deviations and judgement calls

1. **`Split` carries `purged_idx` and `embargoed_idx` beyond §9's fields.**
   §9 specifies counts, but also requires that "the purge/embargo gaps must be
   visible in the UI" (§13.2). Counts alone cannot be drawn. The arrays are
   serialised so the Fold Strip renders the gaps that were actually used rather
   than gaps it infers.
2. **The splitter ban parses instead of grepping.** §9 asks for a CI grep, but
   a text search also matches the docstrings that *explain* the ban — it fired
   on `cv/purged.py` and `cv/__init__.py`, both of which only mention
   `train_test_split` to say it is forbidden. A check that flags its own
   documentation is a check that gets silenced, so it walks the AST and looks at
   identifiers only. That is also stricter: it catches
   `from sklearn.model_selection import KFold as _k`, which a naive grep for
   `KFold(` would miss.
3. **`PurgedKFold` trains on data that follows the test block**, and its module
   docstring says so. That is legitimate for measuring a predictive
   relationship and dishonest for simulating a strategy someone could have run.
   Walk-forward is the scheme for the latter and remains the default.

### What surprised us

- **The leakage registry caught a naming slip immediately.** The three
  hypothesis variants were named `test_label_span_purged_kfold` and so on;
  §12.3 requires a test called exactly `test_label_span_purged`, and
  `test_leakage_suite_complete.py` failed the moment `CURRENT_PHASE` was bumped.
  The registry built in Phase 0 did precisely the job it was built for.
- **The interval-overlap test is worth writing as a negation.** `not (ends
  before it starts or starts after it ends)` is harder to get wrong than
  enumerating the four overlap cases, and the hypothesis sweep over horizons
  1–30 is what gives any confidence that the boundaries are right.
- **The embargo needs the *maximum* `t_end` of the test block**, not its last
  `t_start`. Using the latter would leave the final horizon's worth of test
  labels unprotected — the subtlest of the off-by-ones here, and the reason
  `apply_purge_and_embargo` takes `max(index[test].max(), t_end[test].max())`.

### Deferred

- Wiring folds into a run directory as `folds.json` — Phase 10
- Per-fold metrics on the `Split` record — Phase 8, once metrics exist

---

## Phase 5 — Models and leaderboard ✅

**Completed:** 2026-07-27
**`CURRENT_PHASE`:** `5` — `test_pipeline_fit_on_train_only` and
`test_shuffled_labels_have_no_edge` are now required

### Built

- **`models/protocol.py`** — the `AlphaModel` contract from §10.1.
- **`models/base.py`** — the shared sklearn `Pipeline` (winsoriser, scaler,
  estimator) and inner-split calibration. I3 and §10.2 are enforced once here
  rather than four times across the adapters.
- **`models/linear.py`** — logistic and elastic net. Always on the leaderboard:
  a linear model beating the GBM is important information.
- **`models/trees.py`** — LightGBM, XGBoost, CatBoost behind the optional `gbm`
  extra, lazily imported, single-threaded and seeded for determinism.
- **`models/baselines.py`** — all five mandatory baselines, including the
  shuffled-label wrapper that permutes *inside* the fold.
- **`models/oof.py`** — out-of-fold assembly.
- **`models/leaderboard.py`** — the economic leaderboard and its schema.
- **`eval/classification.py`** — accuracy, AUC, Brier, log-loss, and the
  information coefficient with its t-statistic. Nothing returns `NaN`.
- **`labels/binarise.py`** — the explicit three-class-to-binary reduction (see
  below).

### Exit criteria

| Criterion | Result |
|---|---|
| `test_shuffled_labels_have_no_edge` | ✅ block-bootstrap IC interval contains zero, with a control proving the harness detects a real edge |
| Every model trains on the smoke config under 60s | ✅ `test_every_model_scores_within_the_time_budget` — all six models, timed |
| Leaderboard JSON validates against the schema | ✅ `LeaderboardRow` + `validate_leaderboard` |
| `just verify` green | ✅ 638 passed, 2 skipped, coverage 89.94%; leakage suite 61 passed |

### Deviations and judgement calls

1. **Economic columns are `None`, not placeholders.** §10.1 ranks by
   out-of-fold net-of-cost Sharpe with a DSR haircut. There is no cost model
   until Phase 7, so `oos_net_sharpe`, `dsr`, `psr`, `break_even_bps` and
   `max_drawdown` are null and every row carries a `NO_BACKTEST` flag. Ranking
   falls back to the IC t-statistic — deliberately *not* accuracy (ADR 0007).
   Filling those columns with a number would guarantee someone reads it.
2. **Three-class labels must be reduced explicitly.** Triple-barrier emits
   `{-1, 0, +1}`; every model here is binary. Rather than collapsing silently,
   `alphacore.labels.to_binary` offers `direction` (keep `{-1,+1}`, drop
   expiries) and `touch` (profit barrier hit or not), and the OOF loop raises a
   message naming both if handed multiclass. The choice changes what "accuracy"
   means, so the library does not make it for you.
3. **GBMs live in an optional `gbm` extra**, for the reason ADR 0003 gives for
   torch: `import alphacore` should not drag three compiled boosting libraries
   into a process that only wants to read a tearsheet. All three are installed
   locally and fully tested.
4. **`n_jobs=1` on every GBM.** Multi-threaded histogram building sums
   gradients in a nondeterministic order, so the same seed gives different
   trees run to run — which breaks byte-identical artifacts (I8). The cost is
   speed; the alternative is an unreproducible run.
5. **`N803`/`N806` suppressed inside `models/`.** sklearn's interface is
   `fit(self, X, y)`; renaming would break keyword calls into a `Pipeline` and
   read as a different API to anyone who has used scikit-learn.

### What surprised us

- **`test_pipeline_fit_on_train_only` was passing vacuously.** The spy recorded
  `X.index` on every fitted transform — but only the *first* pipeline step
  receives a DataFrame. The scaler gets the winsoriser's output, a bare numpy
  array with no index, so the recorded index was empty and every containment
  check passed against nothing. Rewritten: the winsoriser is checked by index,
  the scaler by row count, and both spies now assert they observed something.
  A sibling test — `test_the_spy_would_catch_a_global_fit` — performs the leak
  deliberately and requires both spies to trip.
- **The first version of that test also flagged legitimate behaviour.** It
  compared each fit against the union of *every* fold's test rows, but in a
  walk-forward, fold 1 trains on fold 0's test period by design. The check is
  now per fold.
- **sklearn 1.8 deprecated `LogisticRegression(penalty=...)`.** The mix is
  expressed by `l1_ratio` alone now. Caught because the adapter turns estimator
  failures into `ModelError` rather than letting a warning pass.
- **The baselines needed a `_CONFIDENT = 0.99` rather than 1.0.** A probability
  of exactly one produces an infinite log-loss and makes the whole column
  unreadable on the leaderboard.
- **A constant prediction breaks rank metrics, and that is fine.** Every
  deterministic baseline produces one, so the IC is undefined. The row records
  `None` and a `CONSTANT_PREDICTION` flag rather than inventing a number.

### Deferred to later phases

- Optuna tuning against the out-of-fold economic objective (§10.4) — Phase 10,
  since the objective needs the backtester
- Mean Decrease Accuracy and clustered feature importance (§7.3) — Phase 9
- Reliability diagrams and the calibration curve artifact (§10.2) — Phase 8
- Ensembles (`models/ensemble.py`) and sequence models — Phases 5+/13

---

## Phase 6 — Signals and risk ✅

**Completed:** 2026-07-27

### Built

- **`signal/thresholds.py`** — thresholds with a dead zone, and hysteresis as
  an asymmetric hold band: conviction to enter, patience to leave.
- **`signal/sizing.py`** — fixed fraction, volatility targeting, capped Kelly.
  Every path ends at the same clip, with no configuration that bypasses it.
- **`risk/stops.py`** — ATR-scaled stops and targets, ratcheting trailing
  stops, and the same-bar tie-break the labeller uses.
- **`risk/limits.py`** — drawdown kill switch, daily loss, concurrency.

### Exit criteria

| Criterion | Result |
|---|---|
| Sizing never exceeds `max_leverage` | ✅ property test over three methods × five seeds, every knob set above the cap and volatility running to near zero, plus a test that the cap actually binds |
| Hysteresis strictly reduces flips vs plain thresholding | ✅ over 20 random walks, plus a monotonicity test across four hysteresis levels |

### Judgement calls

1. **"Strictly reduces flips" is stated over paths, not per path.** A path that
   crosses a threshold once has one flip either way, so per-path strictness is
   false for a correct implementation. The suite asserts hysteresis *never
   increases* flips on any of 20 paths, and *strictly reduces* them in
   aggregate. The first version failed on seed 7 and the test was wrong, not
   the code.
2. **The kill switch latches.** Recovering does not resume trading. A drawdown
   limit that resets when equity recovers is not a limit, it is a description.
3. **Breaches accumulate rather than short-circuit.** A run that breached three
   limits at once is a different situation from one that breached one.
4. **Kelly mirrors the probability for a short.** A short wins when the
   negative class realises; sizing on `p` regardless would systematically
   oversize every short.

---

## Phase 7 — Backtest engine ✅

**Completed:** 2026-07-27
**`CURRENT_PHASE`:** `7` — **all seven** §12.3 leakage tests now exist and pass,
with nothing skipped

### Built

- **`backtest/costs.py`** — commission, half-spread, linear and square-root
  impact, minimum commission, borrow on short notional, financing on leveraged
  long notional, and the participation cap.
- **`backtest/portfolio.py`** — cash and position, with `assert_reconciles` for
  `--strict`.
- **`backtest/ledger.py`** — the append-only event log. Equity, drawdown and
  the trade blotter are all *views* over it, so they cannot disagree.
- **`backtest/engine.py`** — the event loop in §12.1's order.
- **`backtest/reference.py`** — the deliberately dumb vectorised oracle.
- **`cost_sensitivity`** — the sweep whose curve shape says whether a strategy
  survives its own execution.

### Exit criteria

| Criterion | Result |
|---|---|
| `test_no_same_bar_execution` | ✅ a perfect close-of-bar predictor earns \|Sharpe\| < 1, at lags 1, 2 and 5 |
| `test_perfect_foresight_sanity` | ✅ the forward return produces Sharpe > 10, proving the canary above can detect anything |
| Engine matches `reference.py` to `1e-8` frictionless | ✅ over a 300-bar random walk with a random signal |
| Cash + position reconciles every bar under `--strict` | ✅ and `test_strict_mode_catches_an_accounting_break` proves the assertion can fail |

### Judgement calls

1. **Orders are keyed to the bar they may execute on.** Rather than trusting
   the documented step order, the loop reads bar `t`'s signal only *after* it
   has already executed and marked that bar, and places the resulting order in
   a queue keyed to a strictly later bar. Same-bar execution is
   unrepresentable, and `execution_lag_bars=0` raises rather than being
   silently clamped.
2. **The reference engine compounds open-to-open, not close-to-close.** The
   engine rebalances at each bar's open, so the naive oracle has to match that
   convention. Getting it wrong makes the two disagree by exactly one bar's
   return — which reads as a subtle bug and is actually a specification
   difference. The reference's docstring says so.
3. **`bars_held` is measured from recorded bar indices, never a timestamp
   difference.** On a session calendar a three-day gap can be one bar or three,
   and guessing would put a wrong holding period in the blotter.
4. **A zero-quantity fill does not attract the minimum commission.** Charging
   for a trade that did not happen would make a flat strategy lose money.

### What surprised us

- **The engine silently skipped stops for positions opened before the ATR
  warm-up.** Levels were only attached when `atr[i]` was finite, so a position
  entered on bar 1 with a 14-bar ATR ran unstopped for the rest of the sample —
  no error, no warning, just no protection. Found by a stop test that failed
  for the wrong reason. Levels now attach as soon as an ATR exists, and any
  bars held before that raise an `UNSTOPPED_WARMUP` flag on the result rather
  than passing quietly.
- **The `reason` variable was doing two jobs.** mypy caught an `ExitReason`
  being assigned into a slot typed `str` — the exit reason and the order reason
  shared a name in the event loop. Harmless today, and exactly the kind of
  thing that becomes a wrong `exit_reason` in a blotter later.
- **The reference cross-check needed a companion test.** Agreement to `1e-8`
  proves little unless disagreement is also demonstrable, so
  `test_the_reference_engine_disagrees_once_costs_are_charged` shows the check
  has teeth and is only valid frictionless.

### Deferred to later phases

- `fill_price: next_vwap` — the config accepts it; the engine fills at the open
  regardless. Phase 8, alongside the intraday VWAP series it needs.
- Multi-instrument portfolios — the engine is single-symbol, which is what
  every config so far uses
- Wiring signal, sizing and backtest into a run directory — Phase 10

---

## Phase 8 — Metrics and tearsheet ✅

**Completed:** 2026-07-27

### Built

- **`eval/financial.py`** — the return and trade metric sets, plus
  `break_even_cost_bps`.
- **`eval/deflated.py`** — PSR, the expected maximum Sharpe from `N` trials,
  DSR, and minimum track record length.
- **`eval/bootstrap.py`** — stationary block bootstrap with a Politis-White
  block length.
- **`eval/pbo.py`** — probability of backtest overfitting via CSCV.
- **`report/tearsheet.py`** — the canonical `tearsheet.json` schema, with a
  `verdict` property that states a high PBO in plain language (§11.3).
- **`report/exporters.py`** — parquet and CSV writers plus LTTB downsampling.
- **`report/research_md.py`** — the generated research note.

### Exit criteria

| Criterion | Result |
|---|---|
| Metrics validated against hand-computed fixtures | ✅ geometric return, max drawdown, VaR, and every trade metric asserted against arithmetic written out in the test |
| DSR reproduces the published worked example | ⚠️ **substituted** — see deviation 1 |
| Tearsheet validates against its schema | ✅ round-trips through JSON; an unknown key is rejected |
| Determinism test passes | ✅ the research note and the fold serialisation are byte-stable |

### Deviations

1. **The DSR check is against the formula, not a published worked example.**
   §15 asks DSR to "reproduce the published worked example". Transcribing
   numbers from a paper I cannot open would be fabricating a citation, so the
   substitute is stronger in one respect and weaker in another, and both are
   stated here rather than glossed. The PSR is asserted against an
   **independent transcription of the §11.2 formula** written inside the test
   (`psr_by_hand`), which catches an implementation error but not a
   misreading of the formula itself. DSR is then pinned by its analytic
   properties: it never exceeds PSR when more than one trial was run, it falls
   as the trial count rises, it falls as the spread of trial Sharpes widens,
   and a single trial produces no haircut at all. If someone can supply the
   paper's worked figures, they should be added as a golden fixture.
2. **`eval/bootstrap.py` is a module §3 does not list.** §11.2 requires a
   stationary block bootstrap with a Politis-White block length; putting it in
   `financial.py` would have buried a hundred lines of resampling machinery
   inside the metric definitions.

### What surprised us

- **`np.full(100, 0.001).std(ddof=1)` is `2.18e-19`, not zero.** The
  zero-volatility guard was written as `if deviation == 0` and therefore never
  fired — a constant return series returned a Sharpe of about 4.6e15 instead of
  raising. Every dispersion guard now compares against a tolerance relative to
  the data's own scale, and the test that caught it says so in a comment.
- **PSR saturates fast.** A "Sharpe" of 0.5 *per observation* over 500
  observations pins the normal CDF at 1.0, so the skew and kurtosis tests
  measured nothing until the inputs were changed to a realistic per-observation
  value of 0.05. Worth knowing before reading a PSR of 1.00 as good news.
- **The ulcer index only beats max drawdown after 64 bars.** Drawdowns enter
  squared, so a 5% drawdown must last `(0.40/0.05)² = 64` bars to outweigh a
  single 40% one. The first version of that test asserted the opposite and was
  simply wrong about the arithmetic.

---

## Phase 9 — Explainability ✅

**Completed:** 2026-07-27

### Built

- **`explain/shap_runner.py`** — sampled TreeSHAP, the summary frame behind the
  beeswarm, single-prediction waterfalls, and `verify_additivity`.

### Exit criteria

| Criterion | Result |
|---|---|
| SHAP values sum to `f(x) − E[f(x)]` within tolerance | ✅ observed discrepancy ~1e-15 against a 1e-6 tolerance |
| Runtime on the smoke config under 30s | ✅ timed in the suite |

### Deviations

1. **TreeSHAP comes from the boosting libraries, not the `shap` package.**
   `shap` pulls `numba` → `llvmlite`, which has no wheel resolving on Python
   3.12 with NumPy 2.5 — `uv add shap` fails to build. LightGBM, XGBoost and
   CatBoost each implement exact TreeSHAP in C++ and expose it directly
   (`pred_contrib`, `pred_contribs`, `ShapValues`). Using those is the same
   algorithm with **no new dependency**, and the additivity identity holds to
   ~1e-15 rather than the ~1e-4 the `shap` package needs. The tolerance here is
   set at 1e-6 accordingly, which is a much sharper check than §15 requires.
2. **Only tree models are explained, and a linear model is refused.** §9 scopes
   SHAP to tree models. Forcing TreeSHAP onto a logistic regression or a
   baseline would produce numbers that look like explanations and are not — a
   linear model's coefficients already are its explanation.

### What surprised us

- **Contributions are in raw-score space, not probability space.** The
  additivity identity holds on the log-odds because the sigmoid is not
  additive. `ShapResult` therefore carries `raw_output` alongside, so
  `verify_additivity` compares like with like instead of quietly failing by a
  sigmoid's worth.
- **`explain_prediction` returns the reconstructed output.** The UI can then
  *display* `sum(contributions) + base = output` rather than asking a reviewer
  to trust that it holds — which is the entire argument for SHAP over
  permutation importance.

### Deferred to later phases

- Dependence-plot data beyond the `direction` column — Phase 12, when the chart
  that consumes it exists
- Clustered MDA (§7.3) — needs a permutation loop over the OOF set
- Wiring SHAP output into a run directory as `shap_summary.parquet` — Phase 10

---

## Phase 10 — Pipelines and CLI ✅

**Completed:** 2026-07-28
**`CURRENT_PHASE`:** `10`
**Recipes in `verify`:** `fmt lint types test test-leak smoke` — `smoke` joined
the gate here, as ADR 0009 said it would. Only `e2e` is still outstanding, and
it lands at Phase 12.

This is the integration phase. Nothing here is a new idea; everything here is
Phases 1 to 9 finally wired to each other, and the wiring is what exposes
whether they were ever really compatible.

### Built

- **`pipelines/build_features.py`** — bars out of the lake, features, labels,
  the explicit three-class reduction, and the alignment of all three onto one
  index. `FeatureSet` carries the pre-reduction class balance and the mean
  label uniqueness, because both belong in the tearsheet and neither survives a
  `DataFrame.attrs` round trip reliably.
- **`pipelines/train.py`** — folds, tuning, out-of-fold scoring, one costed
  backtest per model, then deflation across all of them. `TrainingResult` keeps
  the predictions, so the leaderboard row and the equity curve are guaranteed to
  describe the same probabilities.
- **`pipelines/backtest.py`** — signal, sizing, the event-driven engine, the
  §11.2 metric set, the cost sweep, the bootstrap interval, PSR, DSR and the
  minimum track record length. `economic_columns` is the adapter that fills the
  leaderboard's economic columns.
- **`pipelines/tune.py`** — Optuna with a TPE sampler and the configured pruner,
  over an objective that is out-of-fold and economic: fit across the same purged
  folds, generate the signal, run the costed backtest, return net Sharpe (§10.4).
- **`pipelines/full_run.py`** — the golden path, writing the §16 run directory
  contract inside `run_context` so a crashed run is finalised as `failed` with
  its traceback rather than left looking complete.
- **`models.score_models`** — `build_leaderboard` with the predictions kept;
  `build_leaderboard` is now a thin wrapper over it, so no existing caller
  changed.
- **`models.attach_economics`** — fills the economic columns, drops
  `NO_BACKTEST`, adds `LOW_DSR`, and re-ranks. This is the step where ranking
  switches from IC t-stat to net Sharpe (ADR 0007).
- **Per-fold correlation pruning.** `fit_predict_oof` now takes the
  `features.prune` section and applies it *inside* each fold, with that fold's
  training rows as the fit mask. Until this phase the `prune` config was accepted
  and never took effect anywhere.
- **`eval.net_sharpe` and `eval.rolling_sharpe`** — the flat-curve reading of a
  Sharpe, and the rolling 63/252-bar series §11.2 asks for.
- **`Ledger.fill_log`** and `BacktestResult.fills` — the fill-by-fill record
  written to `ledger.parquet`.
- **`RunDirectory.update_manifest`** — lets a run record what it learns about
  itself (the data hash) without breaking write-once, and `run_context` now
  finalises from the on-disk manifest so those amendments survive.
- **CLI** — `alpha features build`, `alpha train`, `alpha backtest`, `alpha run`,
  `alpha report`, `alpha runs list|show|compare`, all with Rich output.

### Exit criteria

| Criterion | Result |
|---|---|
| `alpha run -c configs/experiments/smoke.yaml` completes end to end | ✅ ~10s wall clock, run directory written, manifest `completed` |
| The run directory satisfies the §16 contract | ✅ asserted file by file in `test_the_run_directory_satisfies_the_section_16_contract` |
| Every §15 verb exists with Rich output | ✅ `tests/unit/test_cli_pipelines.py` exercises all of them |
| §12.4 determinism, same session | ✅ `tearsheet.json` identical after stripping `created_at`, `duration_s`, `host` |
| §12.4 determinism, fresh subprocess | ✅ same, from a new interpreter |
| `smoke` added to `verify` (ADR 0009) | ✅ |

### Deviations

1. **Models persist as `model.pkl`, not `model.{txt,json,pt}`.** §16 names
   library-native formats. `FittedModel.save` has pickled the whole pipeline
   since Phase 5, and that is the right unit: a LightGBM booster written as
   `model.txt` loses the winsoriser, the scaler and the calibrator wrapped around
   it, and a model reloaded without its calibrator produces scores that are not
   probabilities. One file per model, holding the thing that actually made the
   predictions.
2. **`alpha features build`, `alpha train` and `alpha backtest` write nothing.**
   §15 lists them as deliverables but does not say they produce run directories.
   A directory containing `features.json` and no `tearsheet.json` would satisfy
   §16's filenames without satisfying its meaning, and the API and UI read those
   directories as complete results. `alpha run` is the only verb that writes one.
3. **`features.n_after_prune` is the last fold's count.** Pruning is fitted, so
   under §4 (I3) it happens per fold and there is no global number. The last
   fold's is reported because that is the selection the model a reader would
   carry forward was trained under. `OOFPredictions.selected_features` keeps all
   of them.
4. **`rolling_sharpe.parquet` duplicates the shortest window.** §16 names one
   file; §11.2 asks for two windows. Both are written as
   `rolling_sharpe_63.parquet` and `rolling_sharpe_252.parquet`, and the shorter
   is copied to the contract's filename for the UI's default.
5. **The run id is pinned, not stripped, in the determinism test.** §12.4 says to
   strip `created_at`, `duration_s` and `host`. `run_id` contains a timestamp, so
   a literal reading of that list would fail on every pair of runs.
   `alpha run --run-id` exists so the two runs differ in nothing else, and the
   test asserts the ids *were* different before pinning them.
6. **Optuna lives behind a `tune` extra** and is imported lazily (ADR 0010).

### A bug found in Phase 7's ledger

`Ledger.trades()` treated **any** fill against an open position as closing a
round trip — including a same-direction scale-in. Phase 7's tests only ever fed
it ±1 signals, so a position was opened and closed and nothing ever scaled.
Phase 10 turned on `vol_target` sizing, which rebalances on most bars, and the
smoke run reported **744 trades for a buy-and-hold position that was opened once
and never closed**. Every trade metric downstream — win rate, payoff ratio,
average holding period, max consecutive losses — was describing something that
did not happen.

A fill in the same direction as the position now blends into the open trade at a
quantity-weighted average entry price. A reducing fill closes only the quantity
it sold, and the remainder keeps the entry price it was actually bought at rather
than inheriting the price it was partly sold at. Six unit tests in
`tests/unit/test_pipeline_units.py` pin each case.

### What surprised us

- **The cost sweep was backtesting a different strategy than the headline.**
  `cost_sensitivity` took a signal but no weights, so it ran the unsized ±1
  version while the headline Sharpe came from the vol-targeted one. The curve
  started at 0.306 against a headline of 1.049, and the break-even cost derived
  from it described a strategy nobody ran. It now takes the same weights, and the
  curve starts where the headline sits. **A sweep that varies one thing has to
  hold everything else constant, and "everything else" includes the parts you did
  not think of as inputs.**
- **A model can be statistically detectable and economically worthless, in the
  same run.** On the synthetic random walk, `logistic` scores an IC t-stat of
  3.46 and an AUC of 0.570 — well clear of the shuffled-label baseline's 0.471 —
  and lands **fifth of six** on net Sharpe at −0.50, tied with the shuffled
  baseline itself. It is predicting which triple barrier gets touched first,
  which is largely a question about volatility, and volatility is genuinely
  predictable. Knowing it earns nothing after costs. This is exactly the
  divergence ADR 0007 exists for, and it showed up unprompted on the first
  honest run.
- **Buy-and-hold wins, and only just clears the haircut.** The top row on the
  smoke leaderboard is `baseline_buyhold` at net Sharpe 0.960 with a deflated
  Sharpe of 0.524 — a whisker above the 0.5 line, on a series constructed to
  contain nothing but drift and noise. Every model that actually predicts
  something is flagged `LOW_DSR` and lands below zero. §19 asks for a run where
  the model does not find an edge, reported honestly, with the diagnostics that
  prove the pipeline would have detected one. This is that run: PBO 0.08, the
  shuffled baseline sitting exactly where it should, and the only thing that
  makes money being the thing that does no work.
- **A look-ahead can hide inside a warm-up.** Vol-target sizing needs a
  volatility estimate, and the first 20 bars of any window do not have one. The
  first implementation back-filled the earliest computed value into them — which
  sizes a position at bar 1 using the volatility of bars 1 to 20. Small, and
  still a leak. Those bars are held flat now, which also removed the
  `UNSTOPPED_WARMUP` warning as a side effect: no position is opened before the
  ATR exists either. It moved the headline Sharpe from 1.049 to 0.960 and the
  deflated Sharpe from 0.47 to 0.52 — **a leak this small was worth a tenth of a
  Sharpe point.**
- **Deflation cannot be computed per model.** The first design ran the engine
  twice — once to collect the spread of Sharpe ratios, once to deflate against
  it. Splitting `backtest_predictions` from `deflate_outcomes` halved the work
  and made the dependency obvious: `Var[SR_n]` is a property of the *search*, not
  of any one configuration in it.
- **`sharpe_ratio` needed a companion, not a fix.** A flat equity curve makes it
  raise, which is correct — but a cost sweep needs the level where trading stops
  being worth it to be a *point on the curve*, not an exception. `net_sharpe`
  reads an all-zero series as zero and still raises on a constant *non-zero* one,
  which is the case that actually indicates a bug.

### Deferred to later phases

- `fill_price: next_vwap` — still accepted by config and still ignored by the
  engine, which always fills at the open
- Multi-symbol portfolios — `primary_symbol` raises rather than silently trading
  the first of several
- Clustered MDA (§7.3) and reliability diagrams (§10.2)
- `shap_summary.parquet` is written only when the selected model is a tree model;
  the smoke config selects a baseline, so the smoke run has no SHAP artifact
- Only the **selected** model is persisted under `models/`. §16 shows a
  directory per model; writing six pickles per run to keep five that nobody
  loads is storage spent on nothing. If the API needs to re-score with a
  runner-up, this is the line to change.
- **The winning hyperparameters are not written to the run directory.** §16 does
  not name an artifact for them and the tearsheet schema has no field. They are
  recoverable — the study is deterministic given the config and the seed, and
  the fitted estimator inside `model.pkl` carries them — but a reviewer has to
  go and get them. Worth a `tuning.json` when the API needs to display them.

---

## Phase 11 — API service ✅

**Completed:** 2026-07-28
**`CURRENT_PHASE`:** `11`
**Recipes in `verify`:** `fmt lint types test test-leak smoke` — unchanged.
`e2e` joins at Phase 12 (ADR 0009).

### Built

- **`alphaservice/api/`** — the FastAPI application. Every endpoint §14.1 names,
  under `/api/v1`, with OpenAPI 3.1 generated from the routes.
- **`api/problems.py`** — RFC 9457 problem documents for every error, including
  a `ProblemDocument` model so the generated TypeScript client has a type for
  the failure branch instead of `unknown`.
- **`api/deps.py`** — `ETag`, `Cache-Control: immutable` and `If-None-Match`
  handling on run resources. Invariant I9 restated in HTTP: a run directory is
  never rewritten, so a client that holds one representation holds it forever.
- **`api/routers/runs.py`** — the tearsheet, config with a diff against the
  packaged defaults, folds, leaderboard, LTTB-downsampled series, paginated
  trades and predictions, SHAP summary, artifact export, the research note, and
  `compare`.
- **`api/routers/jobs.py`** — submission, status, cancellation, and a hand-rolled
  SSE stream carrying `log`, `progress` and one terminal `status` event,
  resumable from `Last-Event-ID`.
- **`alphaservice/jobs/`** — `JobStore` (SQLite, WAL, a rolling per-job log
  table) and `JobRunner` (`ProcessPoolExecutor`, cooperative cancellation).
  ADR 0008.
- **`alphacore.errors.RunCancelledError`** and
  `run_experiment(should_cancel=...)` — the hook that lets a job stop at a phase
  boundary and leave a directory marked `cancelled` rather than half-written.
- **CLI** — `alpha serve` (loopback by default, with a warning if not) and
  `alpha openapi`, which writes the document from the application object so
  `just client` needs no running server.
- **`docs/openapi.json`** and **`web/src/api/generated/schema.d.ts`**, both
  committed, both regenerated by `just client`.

### Exit criteria

| Criterion | Result |
|---|---|
| Schemathesis contract tests pass | ✅ every GET operation, `tests/api/test_openapi_contract.py` |
| A run created via `POST /jobs` is fully readable through the API | ✅ `test_a_run_created_through_the_api_is_fully_readable_through_it` submits a full run over HTTP and reads back tearsheet, config, folds, leaderboard, series, trades, predictions, research note and export |
| SSE streams logs in real time | ✅ `test_the_event_stream_carries_logs_progress_and_a_terminal_status`, plus a resume-from-`Last-Event-ID` test |
| OpenAPI 3.1 auto-generated | ✅ asserted in `test_api_core.py` |
| TS client generated into `web/src/api/generated/` | ✅ `just client`, with a test that fails if the committed copy is stale |

### Deviations

1. **Cancellation is cooperative, not a kill.** §14.2 says jobs are cancellable.
   A `ProcessPoolExecutor` cannot kill one of its workers, and the alternative —
   a process per job — reintroduces exactly the half-written run directory §14.2
   forbids in the next sentence. The flag is set in SQLite and observed at each
   phase boundary. The cost is latency, bounded by the longest phase. ADR 0008
   has the reasoning and what would change at scale.
2. **One schemathesis check is excluded, narrowly.**
   `positive_data_acceptance` asserts that any schema-valid request is accepted.
   Cursors break it: `cursor` is a string because it is *opaque*, and JSON Schema
   cannot express "a string this service issued". A random string is schema-valid
   and correctly refused with a 400. That behaviour is asserted directly in
   `test_api_runs.py`; every other check, including `not_a_server_error` and
   schema conformance, still runs on every operation.
3. **`GET /runs/{id}/research` returns JSON, not `text/markdown`.** §14.1 does not
   name this route; it exists because the UI needs the note. Keeping it JSON
   keeps the generated client typed.
4. **Two warnings are silenced repo-wide.** Schemathesis' ASGI transport leaves
   anyio memory streams for the garbage collector, which surfaces during teardown
   rather than inside the test, so a per-module mark cannot reach it. The filter
   is narrow enough that only an in-process ASGI client can trigger it.

### What surprised us

- **The contract tests found two real defects in the first run.** Not
  hypothetical ones: the routes returned 404, 400 and 409 problem documents that
  the OpenAPI document did not mention, so the generated client had no type for
  any of them and would have surfaced every error as an unhandled rejection.
  Documenting them was the fix. **This is the value of generating the client from
  the schema and then testing the schema against the service** — either alone
  would have missed it.
- **Starlette 1.3 deprecated `httpx` in its test client**, mid-phase, in favour
  of `httpx2`. With `filterwarnings = ["error"]` that is a hard stop rather than
  a notice. Worth knowing that this repo's warning policy makes dependency
  deprecations arrive as build failures — which is the intended trade, but it
  does mean an upgrade can block a phase.
- **`X | Response` return annotations break FastAPI's response-model inference.**
  Any route that can return a 304 has to declare `response_model=` explicitly and
  annotate its return as `Any`. Easy to get wrong quietly: without the explicit
  model the OpenAPI schema for that route silently becomes `{}`, and the
  generated client returns `any`.
- **The job database is the whole cross-process contract.** Once the worker
  writes state and log lines to SQLite and the API only reads, the two halves
  need no other channel — no queue, no pipe, no shared memory. The design got
  simpler the more it was pushed into the database.

### On the discipline

The TDD cycle was followed properly for the job store, the cancellation hook,
and the job runner: test written, watched fail for the right reason, then
implemented. **It was not followed for `api/routers/runs.py` and
`api/routers/jobs.py`** — those were written from §14.1 and tested afterwards.
The tests for them were written without consulting the implementation and all
passed on the first run, which is exactly the weak signal the discipline exists
to avoid: passing immediately proves the test matches the code, not that the code
matches the requirement. The contract tests are the backstop that caught what
those tests did not.

### Deferred to later phases

- **Jobs do not survive a restart.** A job running when the API stops is lost and
  nothing resumes it. Its run directory will not claim to be `completed`, so it
  cannot be mistaken for a result. ADR 0008 names the lease-and-heartbeat design
  this needs.
- `GET /runs/{id}/shap/explain?ts=` — single-prediction contributions. The
  explainer supports it; the route needs a fitted model reloaded per request,
  which wants a cache. Phase 12, when the waterfall chart that consumes it exists.
- Per-fold metrics on `GET /runs/{id}/folds`. The route serves the split
  definitions; the metrics are on the tearsheet's `cv` section, and merging them
  is a Phase 12 shaping decision.
- Authentication. There is none. `alpha serve` binds loopback and warns if told
  to bind anything else.
