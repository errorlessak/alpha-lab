"""The service-facing CLI verbs added in Phase 11."""

from __future__ import annotations

import json

from typer.testing import CliRunner

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
