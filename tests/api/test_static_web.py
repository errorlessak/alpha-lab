"""Serving the built web bundle from the API origin (§14, Phase 14).

The container runs one process on one port. The client requests `/api/v1/...`
with no host, so if the bundle were served from anywhere else every request
would be cross-origin against a service whose allow-list names only the Vite dev
server. These tests pin the three behaviours that makes a single origin work:
assets are served, unknown paths fall back to the shell so deep links survive a
refresh, and an unknown `/api/v1` path still answers as an API rather than as
HTML.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from alphaservice.api.app import create_app

INDEX_HTML = "<!doctype html><title>Alpha Lab</title><div id='root'></div>"


@pytest.fixture
def dist(tmp_path):
    """A minimal Vite-shaped bundle."""
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (root / "assets" / "index-abc123.js").write_text("export const x = 1\n", encoding="utf-8")
    (root / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    return root


@pytest.fixture
def web_client(settings, dist):
    """The application with a bundle mounted."""
    with TestClient(
        create_app(replace(settings, web_dist=dist)), raise_server_exceptions=False
    ) as client:
        yield client


def test_the_shell_is_served_at_the_root(web_client):
    response = web_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "id='root'" in response.text


def test_a_deep_link_falls_back_to_the_shell(web_client):
    # The client router owns this path; the server has never heard of it. A 404
    # here means every refresh on a run page breaks.
    response = web_client.get("/runs/2026-01-01T00-00-00Z_smoke_abc/backtest")
    assert response.status_code == 200
    assert "id='root'" in response.text


def test_a_hashed_asset_is_served_and_cached_forever(web_client):
    response = web_client.get("/assets/index-abc123.js")
    assert response.status_code == 200
    assert "immutable" in response.headers["cache-control"]


def test_the_shell_and_unhashed_files_are_revalidated(web_client):
    # index.html names the current asset hashes. Caching it is caching the
    # pointer to a build that no longer exists.
    assert web_client.get("/").headers["cache-control"] == "no-cache"
    assert web_client.get("/favicon.svg").headers["cache-control"] == "no-cache"


def test_an_unknown_api_route_stays_an_api_error(web_client):
    response = web_client.get("/api/v1/nope")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert "/api/v1/docs" in response.json()["detail"]


def test_the_api_still_answers_with_a_bundle_mounted(web_client):
    assert web_client.get("/api/v1/health").json()["status"] == "ok"


def test_a_traversal_cannot_escape_the_bundle(web_client, tmp_path):
    (tmp_path / "secret.txt").write_text("not yours", encoding="utf-8")
    response = web_client.get("/../secret.txt")
    # Either the client normalises the path away or the fallback serves the
    # shell. Neither may return the file.
    assert "not yours" not in response.text


def test_a_missing_bundle_leaves_the_api_working(settings, tmp_path):
    empty = tmp_path / "not-built"
    empty.mkdir()
    with TestClient(
        create_app(replace(settings, web_dist=empty)), raise_server_exceptions=False
    ) as client:
        assert client.get("/api/v1/health").status_code == 200
        # No fallback was registered, so the root is a plain 404 rather than an
        # HTML page the service cannot produce.
        assert client.get("/").status_code == 404


def test_no_bundle_configured_registers_no_fallback(client):
    # The default fixture has `web_dist=None`, which is what `just serve` uses.
    assert client.get("/").status_code == 404
