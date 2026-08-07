"""The service-facing CLI verbs added in Phase 11."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

from typer.testing import CliRunner

from alphaservice.api.settings import WEB_DIST_ENV_VAR
from alphaservice.cli import app

runner = CliRunner()


def test_help_lists_the_service_verbs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "openapi" in result.stdout
    assert "serve" in result.stdout


def test_openapi_writes_a_3_1_document(tmp_path):
    destination = tmp_path / "openapi.json"
    result = runner.invoke(app, ["openapi", "--output", str(destination)])
    assert result.exit_code == 0, result.output

    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["openapi"].startswith("3.1")
    assert "/api/v1/runs" in document["paths"]


def test_openapi_writes_lf_line_endings(tmp_path):
    # The document is committed and diffed. CRLF on Windows would make every
    # regeneration look like a whole-file change.
    destination = tmp_path / "openapi.json"
    runner.invoke(app, ["openapi", "--output", str(destination)])
    assert b"\r\n" not in destination.read_bytes()


def test_openapi_prints_when_no_output_is_given():
    result = runner.invoke(app, ["openapi"])
    assert result.exit_code == 0
    assert "paths" in result.stdout


def test_serve_rejects_a_web_dist_that_is_not_there():
    result = runner.invoke(app, ["serve", "--web-dist", "no/such/bundle"])
    assert result.exit_code != 0
    assert "no/such/bundle" in result.output.replace("\\", "/")


def test_serve_passes_the_bundle_through_the_environment(tmp_path, monkeypatch):
    # Through the environment rather than an argument, because `--reload`
    # re-imports the application in a process the CLI cannot reach.
    # `setenv` rather than `delenv`: monkeypatch then owns the variable and
    # restores it on teardown, so the value the command sets cannot leak into
    # another test's application.
    monkeypatch.setenv(WEB_DIST_ENV_VAR, "")
    dist = tmp_path / "dist"
    dist.mkdir()

    with patch("uvicorn.run") as run:
        result = runner.invoke(app, ["serve", "--web-dist", str(dist)])

    assert result.exit_code == 0, result.output
    assert run.called
    assert os.environ[WEB_DIST_ENV_VAR] == str(dist)
