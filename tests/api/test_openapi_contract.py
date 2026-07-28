"""Schemathesis contract tests — the Phase 11 exit criterion (§15).

Every operation in the OpenAPI document is exercised with generated inputs and
its response checked against what the document promises. This is the test that
makes the generated TypeScript client trustworthy: the client is derived from
the same document, so a route whose real behaviour disagrees with its schema
produces a client that lies.

Fuzzed inputs mean most of these requests are nonsense, and nonsense must be
refused with a documented status rather than a traceback. The `not_a_server_error`
check is the one that matters most: a 500 from a malformed query string is a bug
whatever the schema says.
"""

from __future__ import annotations

import pytest
import schemathesis
from hypothesis import HealthCheck, given, settings
from schemathesis import checks

from alphaservice.api.app import create_app
from alphaservice.api.settings import Settings

pytestmark = pytest.mark.slow

checks.load_all_checks()

_EXCLUDED_CHECKS = [
    check for check in checks.CHECKS.get_all() if check.__name__ == "positive_data_acceptance"
]
"""One check is excluded, deliberately and narrowly.

`positive_data_acceptance` asserts that any request matching the schema is
accepted. Cursors break that: `cursor` is typed as a string because it is
**opaque**, and JSON Schema cannot express "a string this service previously
issued". A random string is schema-valid and is correctly refused with a 400.

Rejecting it is the behaviour we want, and it is asserted directly in
`test_api_runs.py::test_a_cursor_this_service_did_not_issue_is_rejected`.
Everything else — undocumented statuses, schema conformance, content types, and
`not_a_server_error` — still runs."""


@pytest.fixture(scope="module")
def contract_app(api_workspace, tmp_path_factory):
    """The real application over the session's run fixture."""
    root, _ = api_workspace
    return create_app(
        Settings(
            data_dir=root / "lake",
            runs_dir=root / "runs",
            jobs_db=tmp_path_factory.mktemp("contract-jobs") / "jobs.sqlite",
            max_workers=1,
        )
    )


def _load_schema(app) -> schemathesis.BaseSchema:
    """Load the OpenAPI document straight out of the application.

    Args:
        app: The FastAPI application.

    Returns:
        The schema, bound to the ASGI app so no server has to be started.
    """
    return schemathesis.openapi.from_asgi("/api/v1/openapi.json", app)


def test_the_document_describes_every_endpoint_section_14_names(contract_app):
    document = contract_app.openapi()
    required = {
        "/api/v1/health",
        "/api/v1/system/config/schema",
        "/api/v1/datasets",
        "/api/v1/datasets/{dataset_id}",
        "/api/v1/runs",
        "/api/v1/runs/{run_id}",
        "/api/v1/runs/{run_id}/config",
        "/api/v1/runs/{run_id}/folds",
        "/api/v1/runs/{run_id}/series/{name}",
        "/api/v1/runs/{run_id}/trades",
        "/api/v1/runs/{run_id}/predictions",
        "/api/v1/runs/{run_id}/leaderboard",
        "/api/v1/runs/{run_id}/shap/summary",
        "/api/v1/runs/{run_id}/export/{artifact}",
        "/api/v1/runs/compare",
        "/api/v1/jobs",
        "/api/v1/jobs/{job_id}",
        "/api/v1/jobs/{job_id}/events",
        "/api/v1/jobs/{job_id}/cancel",
    }
    missing = sorted(required - set(document["paths"]))
    assert not missing, f"§14.1 names {missing}, which the document does not describe"


def _contract_operations(schema: schemathesis.BaseSchema) -> list[schemathesis.APIOperation]:
    """Every operation the contract check should exercise.

    Args:
        schema: The loaded schema.

    Returns:
        The read-only operations. `POST` is excluded because a fuzzed job
        submission would queue real pipeline work, and the SSE route because a
        stream has no terminating response for the checker to validate. Both
        are covered directly in `test_api_jobs.py`.
    """
    operations = []
    for result in schema.get_all_operations():
        operation = result.ok()
        if operation is None:  # pragma: no cover - a malformed document
            continue
        if operation.method.upper() != "GET" or operation.path.endswith("/events"):
            continue
        operations.append(operation)
    return operations


def test_the_contract_holds_under_generated_input(contract_app):
    """Every GET operation answers within its documented schema.

    Fuzzed inputs mean most of these requests are nonsense; nonsense must be
    refused with a documented status rather than a traceback.
    """
    schema = _load_schema(contract_app)
    operations = _contract_operations(schema)
    assert operations, "the contract check found no operations to exercise"

    for operation in operations:

        @settings(
            max_examples=6,
            deadline=None,
            suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        )
        @given(case=operation.as_strategy())
        def check(case: schemathesis.Case) -> None:
            case.call_and_validate(excluded_checks=_EXCLUDED_CHECKS)

        check()
