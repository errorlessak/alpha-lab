"""Fixtures for the API suite.

One real run is produced for the whole session. It takes about ten seconds and
every read endpoint needs one: an API tested only against fabricated artifacts
would pass while disagreeing with what the pipeline actually writes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from alphacore.config import load_config
from alphacore.pipelines import run_experiment
from alphacore.runtime import ArtifactStore
from alphaservice.api.app import create_app
from alphaservice.api.settings import Settings


@pytest.fixture(scope="session")
def api_config(repo_root):
    """The smoke config with an absolute fixture path."""
    config = load_config(repo_root / "configs" / "experiments" / "smoke.yaml", use_env=False)
    data = config.data.model_copy(update={"csv_dir": str(repo_root / "samples")})
    return config.model_copy(update={"data": data})


@pytest.fixture(scope="session")
def api_workspace(api_config, tmp_path_factory):
    """A lake and an artifact store holding one completed run."""
    root = tmp_path_factory.mktemp("api")
    summary = run_experiment(
        api_config,
        data_dir=root / "lake",
        store=ArtifactStore(root / "runs"),
        run_id="2026-07-28T00-00-00Z_smoke_apitest",
    )
    return root, summary


@pytest.fixture(scope="session")
def run_id(api_workspace):
    _, summary = api_workspace
    return summary.run_id


@pytest.fixture(scope="session")
def settings(api_workspace, tmp_path_factory):
    root, _ = api_workspace
    return Settings(
        data_dir=root / "lake",
        runs_dir=root / "runs",
        jobs_db=tmp_path_factory.mktemp("api-jobs") / "jobs.sqlite",
        max_workers=1,
    )


@pytest.fixture(scope="session")
def client(settings):
    """A test client over the real application."""
    with TestClient(create_app(settings), raise_server_exceptions=False) as test_client:
        yield test_client
