"""The committed OpenAPI document must match the application (§14.1).

`web/src/api/generated/schema.d.ts` is generated from `docs/openapi.json`, and
that file is committed. If it can drift from the routes, the TypeScript client
describes an API that no longer exists — and the drift is invisible until
someone calls the endpoint that changed.

This is the same guard `just docs` has for the feature data dictionary.
"""

from __future__ import annotations

import json

import pytest

from alphaservice.api.app import create_app

REGENERATE = "just client"


@pytest.fixture(scope="module")
def committed(repo_root):
    path = repo_root / "docs" / "openapi.json"
    assert path.is_file(), f"docs/openapi.json is missing; run `{REGENERATE}`"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_committed_document_matches_the_application(committed):
    current = json.loads(json.dumps(create_app().openapi(), sort_keys=True))
    assert current == committed, (
        f"docs/openapi.json is stale — the routes have changed since it was written. "
        f"Regenerate it with `{REGENERATE}`."
    )


def test_the_generated_typescript_client_is_committed(repo_root):
    client = repo_root / "web" / "src" / "api" / "generated" / "schema.d.ts"
    assert client.is_file(), f"the generated client is missing; run `{REGENERATE}`"
    text = client.read_text(encoding="utf-8")
    # A client that lost its operations is a client that generates `any`.
    for operation in ("getRun", "listRuns", "createJob", "streamJobEvents", "getRunSeries"):
        assert operation in text, f"{operation} is absent from the generated client"
