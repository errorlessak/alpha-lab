# Build log

One entry per phase from §15 of [ARCHITECTURE.md](ARCHITECTURE.md). A phase is
done when its exit criteria pass, not when the code looks finished.

| Phase | Name | Status |
|---|---|---|
| 0 | Foundations | ✅ Done — 2026-07-27 |
| 1 | Data layer | ✅ Done — 2026-07-27 |
| 2 | Features | ⬜ Not started |
| 3 | Labels and weights | ⬜ Not started |
| 4 | Cross-validation | ⬜ Not started |
| 5 | Models and leaderboard | ⬜ Not started |
| 6 | Signals and risk | ⬜ Not started |
| 7 | Backtest engine | ⬜ Not started |
| 8 | Metrics and tearsheet | ⬜ Not started |
| 9 | Explainability | ⬜ Not started |
| 10 | Pipelines and CLI | ⬜ Not started |
| 11 | API service | ⬜ Not started |
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
