"""The application shell: health, the config schema, errors, and OpenAPI (§14.1).

The conventions §14.1 names are tested here rather than per-route, because they
are properties of the application: RFC 9457 problem documents, OpenAPI 3.1, and
immutable caching on run resources.
"""

from __future__ import annotations

PROBLEM_JSON = "application/problem+json"


def test_health_reports_the_version_and_the_store(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["alphacore_version"]
    assert body["alphaservice_version"]
    assert body["artifact_store"]
    assert "git_sha" in body


def test_the_config_schema_is_served_for_the_ui_to_render(client):
    response = client.get("/api/v1/system/config/schema")
    assert response.status_code == 200
    schema = response.json()
    # The real pydantic schema, not a hand-maintained copy that can drift.
    assert schema["title"] == "AlphaConfig"
    assert "meta" in schema["properties"]
    assert "backtest" in schema["properties"]


def test_an_unknown_run_is_a_problem_document_not_a_bare_404(client):
    response = client.get("/api/v1/runs/does-not-exist")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith(PROBLEM_JSON)

    problem = response.json()
    # RFC 9457's required shape.
    assert problem["type"]
    assert problem["title"]
    assert problem["status"] == 404
    assert "does-not-exist" in problem["detail"]


def test_a_validation_error_is_also_a_problem_document(client):
    response = client.get("/api/v1/runs", params={"limit": -5})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith(PROBLEM_JSON)
    assert response.json()["status"] == 422


def test_the_openapi_document_is_3_1(client):
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    document = response.json()
    assert document["openapi"].startswith("3.1")
    assert "/api/v1/runs/{run_id}" in document["paths"]


def test_every_route_carries_an_operation_id_the_client_generator_can_use(client):
    # openapi-typescript names its functions from these. Duplicates or
    # auto-generated noise produce a client nobody wants to import.
    document = client.get("/api/v1/openapi.json").json()
    operation_ids = [
        operation["operationId"]
        for path in document["paths"].values()
        for operation in path.values()
        if isinstance(operation, dict) and "operationId" in operation
    ]
    assert operation_ids
    assert len(operation_ids) == len(set(operation_ids))
    assert all("__" not in name for name in operation_ids)


def test_the_disclaimer_travels_with_the_api(client):
    document = client.get("/api/v1/openapi.json").json()
    assert "not investment advice" in document["info"]["description"].lower()
