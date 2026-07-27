# Build log

One entry per phase from §15 of [ARCHITECTURE.md](ARCHITECTURE.md). A phase is
done when its exit criteria pass, not when the code looks finished.

| Phase | Name | Status |
|---|---|---|
| 0 | Foundations | ✅ Done — 2026-07-27 |
| 1 | Data layer | ⬜ Not started |
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
