# Architecture diagrams

Four views of the system, rendered by GitHub from their source. They are
Mermaid rather than exported images on purpose: a diagram that has to be
re-exported to stay true is a diagram that stops being true.

For what the pipeline *means* rather than how it is wired, read
[methodology.md](methodology.md). For the full specification, read
[ARCHITECTURE.md](../ARCHITECTURE.md).

---

## 1. Layering

Dependencies run one way and only one way. The contract is enforced by
import-linter in `pyproject.toml` and checked by `just lint`, so a violation
turns the build red rather than becoming a convention nobody can undo later.

```mermaid
flowchart TD
    W["<b>web/</b><br/>React 19 · TanStack Router<br/>types generated from the OpenAPI document"]
    S["<b>alphaservice</b><br/>FastAPI · Typer CLI · job runner<br/>reads the artifact store, never writes to it"]
    C["<b>alphacore</b><br/>data · features · labels · CV · models<br/>backtest · eval · report · pipelines"]
    F["<b>runs/</b> — the immutable artifact store<br/><b>data/</b> — the point-in-time lake"]

    W -->|HTTP /api/v1| S
    S -->|imports| C
    C -->|writes once| F
    S -->|reads| F

    style C fill:#1e3a5f,color:#ffffff
    style F fill:#e4e6de,color:#191b18
```

`alphacore` never imports FastAPI, Starlette, uvicorn, Typer or Click — a second
import-linter contract says so. It is a research library that happens to have a
service in front of it, not a web application with some statistics inside.

---

## 2. What a run produces

`alpha run` is the only verb that writes a run directory. The stage verbs
(`features build`, `train`, `backtest`) run the pipeline up to that point and
print what they found, because a directory holding `features.json` and no
`tearsheet.json` would satisfy §16's filenames without satisfying its meaning.

```mermaid
flowchart LR
    CFG["config.yaml"] --> RUN["alpha run"]
    RUN --> DIR["runs/{run_id}/"]

    DIR --> T["<b>tearsheet.json</b><br/>the canonical result —<br/>the API and UI read nothing else"]
    DIR --> M["manifest.json<br/>config hash · data hash<br/>code revision · seed · timings"]
    DIR --> S["equity · drawdown · rolling_sharpe<br/>trades · predictions · ledger<br/><i>parquet</i>"]
    DIR --> R["research.md<br/>the tearsheet as prose, caveats first"]
    DIR --> O["config.resolved.yaml · quality.json<br/>features.json · folds.json<br/>leaderboard.csv · metrics.csv"]
    DIR --> L["logs.jsonl · models/{model}/model.pkl"]

    style T fill:#1e3a5f,color:#ffffff
```

The directory is written once and never modified (§4, I9). Producing a
different result means producing a new run — which is why `alpha runs compare`
can say, when two runs share a config hash, that any difference between them
came from the data or the code rather than the configuration.

---

## 3. Submitting a job

The API creates runs through an in-process job runner
([ADR 0008](decisions/0008-in-process-job-runner.md)) rather than a queue and a
broker. One user, one machine, and a dependency on Redis that would exist only
to move a JSON blob three inches.

```mermaid
sequenceDiagram
    participant UI as web app
    participant API as FastAPI
    participant JS as job store<br/>(SQLite)
    participant W as worker process
    participant FS as runs/

    UI->>API: POST /api/v1/jobs
    API->>JS: record queued
    API-->>UI: 202 Accepted + Location
    API->>W: submit to pool

    UI->>API: GET /jobs/{id}/events
    activate API
    W->>FS: write run directory
    W->>JS: append log lines, phase progress
    JS-->>API: poll
    API-->>UI: SSE: log · progress · terminal status
    deactivate API

    UI->>API: GET /runs/{run_id}
    API->>FS: read tearsheet.json
    API-->>UI: the result
```

The stream resumes from `Last-Event-ID` if the connection drops, and ends with
exactly one terminal status frame. Cancellation is cooperative: a cancel lands
at the next phase boundary, so the run directory is finalised rather than killed
mid-write.

**Jobs do not survive an API restart.** Nothing resumes them, and the run
directory will not claim to be `completed`. That is a documented limitation, not
a bug to be discovered.

---

## 4. The container

One image, one process, one origin. The web client requests `/api/v1/…` with no
host, so the bundle and the API have to answer on the same origin — which is
also why the CORS allow-list names only the Vite dev server rather than `*`.

```mermaid
flowchart TD
    subgraph build["build stages"]
        N["node:22-slim<br/>npm ci → tsc → vite build"]
        P["python:3.12-slim + uv<br/>uv sync --locked --extra gbm"]
    end

    subgraph run["runtime image"]
        U["uvicorn<br/>alphaservice.api.app:app"]
        ST["ALPHA__WEB_DIST → /app/web/dist"]
    end

    N -->|dist/| ST
    P -->|.venv| U

    U --- V1[("/data<br/>volume")]
    U --- V2[("/runs<br/>volume")]
    B["browser"] -->|127.0.0.1:8000| U

    style U fill:#1e3a5f,color:#ffffff
```

In development none of this applies: Vite serves the app on 5173 and proxies
`/api` to `just serve` on 8000, and the static mount is inert unless
`ALPHA__WEB_DIST` is set.

The image carries the `gbm` extra and not `dl` — LightGBM, XGBoost and CatBoost
are what the baselines and TreeSHAP need, while torch adds about 2 GB for a
model family no packaged config uses. `--build-arg EXTRAS="--extra gbm --extra
dl"` puts it in.
