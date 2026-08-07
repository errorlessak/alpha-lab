# ADR 0013 — One container serving the API and the bundle from one origin

- **Status:** Accepted
- **Date:** 2026-08-07
- **Phase:** 14

## Context

§15 Phase 14 asks for a multi-stage Dockerfile and a compose file whose exit
criterion is that `docker compose up` serves the app with the sample dataset on
a clean machine.

The web client requests `/api/v1/…` with no host (`web/src/lib/api.ts`), and
nothing in `src/` may declare an API shape or an API location by hand. In
development Vite proxies `/api` to the service on 8000, so the browser only ever
sees one origin, and the CORS allow-list in `api/app.py` names the two Vite dev
origins rather than `*` precisely because production was expected to need no
entry at all.

That leaves a question the specification does not answer: in the container, what
serves the bundle?

Three arrangements were available.

1. **Two containers** — nginx serving `dist/` and proxying `/api` to a uvicorn
   container. The conventional shape.
2. **One container, two processes** — nginx and uvicorn under a supervisor.
3. **One container, one process** — uvicorn serving both.

## Decision

**One container, one process.** The service mounts a built bundle when
`ALPHA__WEB_DIST` names a directory containing `index.html`, in
`alphaservice/api/static.py`:

- Assets under `assets/` are served `immutable` — Vite fingerprints those
  filenames with a content hash, so each URL names one build. Nothing else is:
  `index.html` is the document that points at the current hashes, and
  `favicon.svg` keeps its name across builds.
- Any other path falls back to `index.html`, so `/runs/{id}/backtest` is
  resolved by the client router rather than 404ing on a path no server route
  knows about.
- An unmatched path under `/api/v1` raises rather than falling back. Returning
  the HTML shell to a client expecting JSON turns a wrong URL into a parse error
  a long way from its cause.
- The candidate path is resolved and required to sit under the bundle root, so a
  traversal is rejected by where it lands rather than by pattern-matching the
  path that produced it.
- The routes are `include_in_schema=False`. The OpenAPI document is the contract
  the TypeScript client is generated from, and a catch-all returning `text/html`
  describes nothing a client should call.

Mounting is **opt-in**. With `ALPHA__WEB_DIST` unset — which is every
development shell and every test that does not ask for it — none of this is
registered, and `just serve` behaves exactly as it did.

## Consequences

**Benefit.** The exit criterion is met by one image, one port and one thing to
configure. There is no second server whose proxy rules can disagree with the
first about buffering, which matters here: the job console is a server-sent
event stream, and a buffering proxy breaks it in a way that is invisible until
you need it.

**Benefit.** Same-origin in production means the CORS allow-list stays a
development-only concession rather than growing a deployment origin, which is
the usual route to `allow_origins=["*"]` on a service with no authentication.

**Benefit.** The mount being opt-in keeps one code path for tests and
development. `tests/api/test_static_web.py` builds an application with a bundle
and one without, and asserts the second registers no fallback at all.

**Cost.** uvicorn serves static files less efficiently than nginx. At the scale
this runs — one user, one machine, a 99 KB initial bundle — that is not a cost
anyone can measure, and putting a CDN or a reverse proxy in front later requires
no change here.

**Cost.** A misconfigured `ALPHA__WEB_DIST` logs a warning and leaves the API
running rather than failing to start. A bundle that is not there is a
misconfiguration, and taking the working half of the service down with the
broken half helps nobody.

**Cost.** The fallback route matches every path, so it must be registered after
every router. That ordering is now load-bearing, and `create_app` says so.
