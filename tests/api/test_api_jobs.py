"""Job submission, status, cancellation and the SSE stream (§14.1, §14.2).

These run against the application's real process pool, because the property
being tested is that work done in another process is visible through the API.
"""

from __future__ import annotations

import json
import time

import pytest

TERMINAL = {"completed", "failed", "cancelled"}


def _await_terminal(client, job_id: str, *, timeout: float = 300.0) -> dict:
    """Poll `GET /jobs/{id}` until the job finishes.

    Args:
        client: The test client.
        job_id: The job id.
        timeout: Seconds to wait.

    Returns:
        The finished job document.

    Raises:
        AssertionError: If it is still running when the timeout expires.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] in TERMINAL:
            return body
        time.sleep(0.1)
    msg = f"job {job_id} did not finish within {timeout}s"
    raise AssertionError(msg)


@pytest.fixture(scope="module")
def smoke_config_ref(repo_root):
    return str(repo_root / "configs" / "experiments" / "smoke.yaml")


def test_submitting_a_job_is_accepted_with_a_location(client, smoke_config_ref):
    response = client.post("/api/v1/jobs", json={"type": "ingest", "config_ref": smoke_config_ref})
    assert response.status_code == 202
    body = response.json()
    assert response.headers["location"] == f"/api/v1/jobs/{body['job_id']}"
    assert body["type"] == "ingest"
    assert body["status"] in {"queued", "running"}
    assert response.headers["cache-control"] == "no-store"


def test_a_job_is_readable_through_the_api_until_it_finishes(client, smoke_config_ref):
    job_id = client.post(
        "/api/v1/jobs", json={"type": "ingest", "config_ref": smoke_config_ref}
    ).json()["job_id"]
    finished = _await_terminal(client, job_id)
    assert finished["status"] == "completed"
    assert finished["duration_s"] is not None
    assert finished["progress"]["fraction"] == 1.0


def test_submitting_an_unknown_job_type_is_a_validation_problem(client, smoke_config_ref):
    response = client.post(
        "/api/v1/jobs", json={"type": "teleport", "config_ref": smoke_config_ref}
    )
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")


def test_submitting_a_config_that_does_not_exist_fails_fast(client):
    response = client.post("/api/v1/jobs", json={"type": "ingest", "config_ref": "nope.yaml"})
    assert response.status_code == 422
    # The caller learns about the typo now, not from a job that dies later.
    assert "nope.yaml" in response.json()["detail"]


def test_jobs_are_listed_and_filterable(client, smoke_config_ref):
    job_id = client.post(
        "/api/v1/jobs", json={"type": "ingest", "config_ref": smoke_config_ref}
    ).json()["job_id"]
    _await_terminal(client, job_id)

    listing = client.get("/api/v1/jobs").json()
    assert any(job["job_id"] == job_id for job in listing["jobs"])

    completed = client.get("/api/v1/jobs", params={"status": "completed"}).json()
    assert all(job["status"] == "completed" for job in completed["jobs"])


def test_an_unknown_job_is_a_problem_document(client):
    response = client.get("/api/v1/jobs/ghost")
    assert response.status_code == 404
    assert response.json()["title"] == "Job not found"


def test_cancelling_a_finished_job_is_a_conflict(client, smoke_config_ref):
    job_id = client.post(
        "/api/v1/jobs", json={"type": "ingest", "config_ref": smoke_config_ref}
    ).json()["job_id"]
    _await_terminal(client, job_id)

    response = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert response.status_code == 409
    assert "already finished" in response.json()["detail"]


def test_cancelling_an_unknown_job_is_a_404(client):
    assert client.post("/api/v1/jobs/ghost/cancel").status_code == 404


def test_the_event_stream_carries_logs_progress_and_a_terminal_status(client, smoke_config_ref):
    """§15's exit criterion: SSE streams logs in real time.

    The stream is opened while the job is still running and must deliver its
    log lines, its progress, and exactly one terminal `status` frame.
    """
    job_id = client.post(
        "/api/v1/jobs", json={"type": "ingest", "config_ref": smoke_config_ref}
    ).json()["job_id"]

    events = []
    with client.stream("GET", f"/api/v1/jobs/{job_id}/events") as stream:
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")
        assert stream.headers["cache-control"] == "no-store"

        name = None
        for line in stream.iter_lines():
            if line.startswith("event:"):
                name = line.partition(":")[2].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.partition(":")[2].strip())))
                if name == "status":
                    break

    kinds = [name for name, _ in events]
    assert "log" in kinds, "a job that streamed no log line is not streaming in real time"
    assert "progress" in kinds
    assert kinds[-1] == "status"
    assert kinds.count("status") == 1
    assert events[-1][1]["status"] in TERMINAL


def test_the_stream_can_resume_from_a_last_event_id(client, smoke_config_ref):
    job_id = client.post(
        "/api/v1/jobs", json={"type": "ingest", "config_ref": smoke_config_ref}
    ).json()["job_id"]
    _await_terminal(client, job_id)

    everything = _collect_log_seqs(client, job_id)
    assert len(everything) > 1

    resumed = _collect_log_seqs(client, job_id, last_event_id=everything[0])
    assert everything[0] not in resumed
    assert resumed == everything[1:]


def _collect_log_seqs(client, job_id: str, *, last_event_id: int | None = None) -> list[int]:
    """Read a finished job's stream and return the log sequence numbers.

    Args:
        client: The test client.
        job_id: The job id.
        last_event_id: Resume point, as a browser would send it.

    Returns:
        The sequence numbers delivered.
    """
    headers = {"Last-Event-ID": str(last_event_id)} if last_event_id is not None else {}
    seqs: list[int] = []
    with client.stream("GET", f"/api/v1/jobs/{job_id}/events", headers=headers) as stream:
        name = None
        for line in stream.iter_lines():
            if line.startswith("event:"):
                name = line.partition(":")[2].strip()
            elif line.startswith("data:"):
                payload = json.loads(line.partition(":")[2].strip())
                if name == "log":
                    seqs.append(payload["seq"])
                if name == "status":
                    break
    return seqs


def test_streaming_an_unknown_job_is_a_404(client):
    assert client.get("/api/v1/jobs/ghost/events").status_code == 404


@pytest.mark.slow
def test_a_run_created_through_the_api_is_fully_readable_through_it(client, smoke_config_ref):
    """§15's Phase 11 exit criterion, end to end.

    Submit a full run over HTTP, wait for it, then read every artifact the UI
    needs back out of the API. Nothing here touches the filesystem directly.
    """
    job_id = client.post(
        "/api/v1/jobs", json={"type": "full_run", "config_ref": smoke_config_ref}
    ).json()["job_id"]

    finished = _await_terminal(client, job_id, timeout=600.0)
    assert finished["status"] == "completed", finished.get("error")
    run_id = finished["run_id"]
    assert run_id

    tearsheet = client.get(f"/api/v1/runs/{run_id}")
    assert tearsheet.status_code == 200
    assert tearsheet.json()["selected_model"]

    for path in (
        f"/api/v1/runs/{run_id}/config",
        f"/api/v1/runs/{run_id}/folds",
        f"/api/v1/runs/{run_id}/leaderboard",
        f"/api/v1/runs/{run_id}/series/equity",
        f"/api/v1/runs/{run_id}/trades",
        f"/api/v1/runs/{run_id}/predictions",
        f"/api/v1/runs/{run_id}/research",
        f"/api/v1/runs/{run_id}/export/tearsheet.json",
    ):
        assert client.get(path).status_code == 200, path

    index = client.get("/api/v1/runs").json()
    assert any(row["run_id"] == run_id for row in index["runs"])
