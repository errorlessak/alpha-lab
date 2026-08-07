# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — while it
is pre-1.0, the public surface (the config schema, the `/api/v1` contract, and
the §16 run directory layout) may change in a minor release.

## [Unreleased]

### Added

- **Packaging (§15 Phase 14).** A multi-stage `Dockerfile`, a
  `docker-compose.yml` that serves the whole application on
  `127.0.0.1:8000`, and a `.dockerignore`. The first container start produces a
  run from the committed fixture, so the index is not empty.
- **The API can serve the built web bundle from its own origin.** Set
  `ALPHA__WEB_DIST`, or run `alpha serve --web-dist web/dist` / `just
  serve-bundled`. Hashed assets are immutable, `index.html` revalidates, and
  deep links fall back to the client router instead of 404ing.
- **CI covering the whole gate**, split into jobs: the Python suite with the
  85% coverage floor, the Phase 13 sequence models with an assertion that
  nothing skipped, the web checks, the Playwright golden path, a check that the
  generated files (data dictionary, OpenAPI document, TypeScript client) are
  current, and `docker compose up` against the exit criterion in §15.
- `docs/methodology.md` — the statistical argument for a reader who has not
  read the architecture specification — and `docs/diagrams.md`.
- **A contrast check over every text node**, in both colour schemes, in the
  end-to-end suite. See below for what it found on its first run.
- ADR [0013](docs/decisions/0013-one-container-one-origin.md) (one container,
  one origin) and ADR
  [0014](docs/decisions/0014-text-only-colour-variants.md) (text-only colour
  variants).
- Repository furniture: `LICENSE` (MIT), `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, this changelog, issue and pull request templates.
- An end-to-end assertion that a run page does not overflow at 768px, alongside
  the existing one for the index.

### Fixed

- **A typo in `models.candidates` produced a leaderboard instead of an error.**
  An unregistered name was reported exactly like a model whose optional
  dependency is missing: a `FAILED` row, a warning in the log, and a full
  leaderboard for everything else — which the researcher reads as though the
  model they meant to run had been on it, with the trial count feeding the
  deflated Sharpe haircut to match. The two are now separated. A missing extra
  still earns a `FAILED` row, because doing without torch is a decision a
  researcher is allowed to make. A name that is not in the registry stops the
  run and lists what is available.
- **A NUL byte in a path segment answered `500 Internal Server Error` in
  `text/plain`** — the one response shape §14.1 rules out. `Path.resolve()`
  raises `ValueError` on one rather than reporting the file missing, and
  `Path.is_dir()` had been swallowing it. `GET /api/v1/runs/%00` and
  `.../export/a%00b.json` are now a problem document.
- **A failed job's `error` field was a raw traceback.** The UI puts that field
  in front of a reader, and forty frames of virtualenv paths in place of
  "Could not parse YAML in pyproject.toml: expected '<document start>'" is the
  opposite of §13.5's rule — it also published the server's filesystem layout
  into the browser. Library errors now arrive as their own message; anything
  else is a bug rather than a bad request and keeps its traceback, which is
  written to the job log either way.
- **Jobs left `queued` by a process that died stayed queued forever**,
  indistinguishable in the jobs list from one about to run. A starting runner
  now closes them out, saying so. `running` is deliberately untouched: a worker
  may be mid-write on a run directory.
- **A run id could escape the artifact store.** `GET
  /api/v1/runs/%2E%2E/export/README.md` served a file from the store's parent
  directory: `RunDirectory.artifact_path` refused to escape a run directory,
  but nothing checked that the run directory was inside the store.
  `ArtifactStore.open_run` and `create_run` now require a run id to name a
  direct child. The service has no authentication and binds loopback, so this
  was not remotely exploitable in the default posture — it was still a
  traversal.
- **`ALPHA__JOBS_DB` broke every `alpha run` in the same shell.** The variable
  the API service reads was missing from the config loader's set of
  process-level names, and `AlphaConfig` forbids extra keys, so setting it
  failed the load with `jobs_db: Extra inputs are not permitted`. A test now
  derives the service's variables from `alphaservice.api.settings` so the two
  packages cannot drift apart again — and it immediately caught
  `ALPHA__WEB_DIST`, added later in the same release, before it shipped.
- **An unmatched route returned `{"detail": "Not Found"}`** rather than an RFC
  9457 problem document. The handler was registered against FastAPI's
  `HTTPException`; Starlette's router raises the base class for an unmatched
  path and an unsupported method.
- **Three colours failed WCAG AA as text.** `--color-sulphur` measured 2.31:1 in
  light mode — it carries the `HIGH_PBO` and `LOW_DSR` flags on the run index —
  `--color-oxblood` 3.92:1 on the striped row fill in dark, on the label for a
  run that did not complete, and `--color-ink-muted` 4.02:1, which is most of
  the secondary text in the application. Text-only variants now exist and the
  structural colours are unchanged; Lighthouse accessibility is 100 in both
  schemes.
- **The run index scrolled the whole page sideways below about 1000px.** The
  ten-column ledger is now in a scroll container that takes focus only while it
  actually overflows.
- Two end-to-end tests raced the first paint and the webfont swap, and failed
  intermittently: one pressed Tab before the application had rendered anything
  focusable, the other measured layout mid-swap.

### Changed

- `.env.example` names the override flag correctly — it is `--set`, not `-o`.
- `.env.example` documents only variables the code reads. Removed
  `ALPHA__API_TOKEN` — the service has no authentication and the variable
  implied otherwise — along with `ALPHA__ALPHAVANTAGE_API_KEY` (the provider is
  not implemented) and `ALPHA__LOG_LEVEL` (it is a flag). Added
  `ALPHA__JOBS_DB` and the `ALPHA__SECTION__KEY` override precedence.
- Prettier now covers `e2e/`, `scripts/` and the root configuration files, not
  only `src/`.

## [0.1.0]

Phases 0–13 of the build specification: determinism and hashing, run manifests,
the immutable artifact store, the config schema, point-in-time ingestion, the
parquet lake and duckdb catalog, quality reporting, 27 causal feature
primitives with a leak-proof registry, four labelling methods with uniqueness
and time-decay weights, purged cross-validation (walk-forward, purged k-fold,
CPCV), the out-of-fold leaderboard with mandatory baselines, signal generation
and sizing, the event-driven backtester with its cost model, the honesty
metrics (PSR, DSR, minimum track record length, PBO), the canonical tearsheet
and generated research note, TreeSHAP explanations, hyperparameter search
against an out-of-fold economic objective, the CLI, the FastAPI service with
SSE job streaming, the web application, and five sequence models with
fold-aware windowing.

See [PROGRESS.md](PROGRESS.md) for the phase-by-phase build log, including what
was deferred and why.
