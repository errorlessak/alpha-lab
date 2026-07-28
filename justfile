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
    uv run pre-commit install

# Add the optional deep-learning extra (Phase 13).
setup-dl:
    uv sync --all-packages --all-extras

# Format. Ruff is the only formatter (ADR 0004).
fmt:
    uv run ruff format .

# Lint, including the one-way dependency contract from §3.
lint:
    uv run ruff check .
    uv run lint-imports

# Type-check both source trees under mypy strict.
types:
    uv run mypy packages/alphacore/src packages/alphaservice/src

# Full test suite with the coverage gate.
test:
    uv run pytest --cov=alphacore --cov-fail-under=85

# The leakage suite (§12.3) — the tests that decide whether any number this
# system produces is real.
test-leak:
    uv run pytest tests/leakage -x

# End-to-end smoke run — the §15 Phase 10 exit criterion and the integration
# gate for every phase before it (ADR 0009).
smoke:
    uv run alpha run -c configs/experiments/smoke.yaml

# Playwright end-to-end suite. Wired up in Phase 12.
e2e:
    npm --prefix web exec playwright test

# The gate. `smoke` joined it at Phase 10; `e2e` joins at Phase 12 (ADR 0009).
verify: fmt lint types test test-leak smoke

# The gate in its final form, for use once every phase has landed.
verify-full: verify e2e

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

# Emit a requirements.txt for graders who want one (ADR 0006).
export-requirements:
    uv export --no-hashes --format requirements-txt -o requirements.txt
