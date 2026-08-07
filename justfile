# Alpha Lab task runner (§17 of ARCHITECTURE.md).
#
# `just verify` is the gate. A phase is not done until it is green.
#
# Every recipe line runs in its own shell, so no line may depend on a `cd` from
# a previous one. `npm --prefix web ...` is used instead of `cd web && ...` so
# the same recipe works under sh and PowerShell.

set windows-shell := ["powershell.exe", "-NoProfile", "-Command", "$ErrorActionPreference='Stop';"]

# Determinism (§4, I8). CPython reads this at startup, so it has to be exported
# by the task runner rather than set from inside Python.
export PYTHONHASHSEED := "0"

# Default target: show the recipe list.
default:
    @just --list

# Install everything needed to develop. Excludes the [dl] extra — `import
# alphacore` must work without torch (§10.3), and torch is a multi-GB download
# that Phase 13 is the first to need. Use `just setup-dl` when you get there.
setup:
    uv sync --all-packages
    npm --prefix web ci
    npm --prefix web exec playwright install --with-deps chromium
    uv run pre-commit install

# Add the optional deep-learning extra (Phase 13).
setup-dl:
    uv sync --all-packages --all-extras

# Format. Ruff for Python (ADR 0004), Prettier for the web app.
fmt:
    uv run ruff format .
    npm --prefix web run format

# Lint, including the one-way dependency contract from §3.
lint:
    uv run ruff check .
    uv run lint-imports
    npm --prefix web run lint

# Type-check both source trees under mypy strict, and the web app under
# TypeScript strict.
types:
    uv run mypy packages/alphacore/src packages/alphaservice/src
    npm --prefix web run typecheck

# Full test suite with the coverage gate, plus the web app's unit tests.
test:
    uv run pytest --cov=alphacore --cov-fail-under=85
    npm --prefix web run test

# The leakage suite (§12.3) — the tests that decide whether any number this
# system produces is real.
test-leak:
    uv run pytest tests/leakage -x

# End-to-end smoke run — the §15 Phase 10 exit criterion and the integration
# gate for every phase before it (ADR 0009).
smoke:
    uv run alpha run -c configs/experiments/smoke.yaml

# The Playwright golden path (§13.5). Builds first, because the suite runs
# against the built app rather than the dev server — a bundle that only works
# unbundled is a bundle nobody can ship.
e2e:
    npm --prefix web run build
    npm --prefix web run e2e

# Lighthouse against the built app, median of three (§13.5 sets the bar at 95).
# Needs `npm --prefix web run preview` already serving on 4173.
lighthouse:
    npm --prefix web run lighthouse

# Regenerate the committed screenshots in docs/screenshots (§15 Phase 12 exit).
screenshots:
    npm --prefix web run build
    npm --prefix web run screenshots

# The bundle budget: initial JS <= 200 KB gzipped (§13.5). Fails, never warns.
budget:
    npm --prefix web run build
    npm --prefix web run budget

# The gate, in its final form. `smoke` joined at Phase 10 and `e2e` at Phase 12,
# which is what ADR 0009 said would happen — so `verify` and `verify-full` are
# now the same recipe and that ADR is history.
verify: fmt lint types test test-leak smoke e2e budget

# Kept as an alias so muscle memory and any existing CI invocation still work.
verify-full: verify

# Regenerate the feature data dictionary from the registry (§7.1).
# The command writes the file itself rather than relying on shell redirection,
# which defaults to UTF-16 in Windows PowerShell.
docs:
    uv run python -m alphacore.features --emit-docs --output docs/data-dictionary.md

# Run the API on loopback. The service has no authentication and can start
# jobs, so it does not bind a routable interface by default.
serve:
    uv run uvicorn alphaservice.api.app:app --host 127.0.0.1 --port 8000 --reload

# Regenerate the OpenAPI document and the TypeScript client from it (§14.1).
# The document is written from the application object, so no server is needed.
# `tests/api/test_openapi_document.py` fails if the committed copy is stale.
client:
    uv run alpha openapi -o docs/openapi.json
    npx --yes openapi-typescript@7 docs/openapi.json -o web/src/api/generated/schema.d.ts

# Serve the API *and* the built web app from one origin, the way the container
# does. Useful for checking the production bundle without Docker; `just serve`
# plus the Vite dev server is the normal development loop.
serve-bundled:
    npm --prefix web run build
    uv run alpha serve --web-dist web/dist

# Build the container image (§15 Phase 14). The default image carries the `gbm`
# extra and not `dl` — see the header of the Dockerfile.
docker-build:
    docker build -t alpha-lab:latest .

# The whole application on http://127.0.0.1:8000, seeded with one run from the
# committed fixture.
docker-up:
    docker compose up --build

# Emit a requirements.txt for graders who want one (ADR 0006).
export-requirements:
    uv export --no-hashes --format requirements-txt -o requirements.txt
