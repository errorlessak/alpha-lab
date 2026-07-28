"""The Phase 10 exit criterion: `alpha run` writes a complete run directory.

§15 calls this the integration gate for everything before it. If ingestion,
features, labels, folds, models, signals, risk, the backtester, the metrics and
the explainer do not compose, this is where it shows.

The §12.4 determinism check lives here too: the same config run twice — once in
this session, once in a fresh interpreter — must produce a byte-identical
`tearsheet.json` once `created_at`, `duration_s` and `host` are stripped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from alphacore.config import load_config
from alphacore.pipelines import run_experiment
from alphacore.report import TEARSHEET_FILENAME, validate_tearsheet
from alphacore.runtime import ArtifactStore, RunStatus

TIME_BUDGET_S = 120.0
"""A smoke run is meant to be something a reviewer runs while reading. If it
stops being that, the config has grown past what `just smoke` should gate on."""

VOLATILE_MANIFEST_FIELDS = ("duration_s", "host")
"""Fields §12.4 strips before comparing two runs. They describe the machine and
the wall clock, not the result."""


@pytest.fixture(scope="module")
def smoke_config(repo_root):
    """The committed smoke config, with an absolute fixture path."""
    config = load_config(repo_root / "configs" / "experiments" / "smoke.yaml", use_env=False)
    data = config.data.model_copy(update={"csv_dir": str(repo_root / "samples")})
    return config.model_copy(update={"data": data})


@pytest.fixture(scope="module")
def completed(smoke_config, tmp_path_factory):
    """One full run of the smoke config, plus how long it took."""
    root = tmp_path_factory.mktemp("golden")
    started = time.perf_counter()
    summary = run_experiment(
        smoke_config,
        data_dir=root / "lake",
        store=ArtifactStore(root / "runs"),
    )
    return summary, time.perf_counter() - started, root


def test_the_run_completes_within_the_smoke_budget(completed):
    _, elapsed, _ = completed
    assert elapsed < TIME_BUDGET_S, (
        f"the smoke run took {elapsed:.1f}s, over the {TIME_BUDGET_S}s budget"
    )


def test_the_manifest_is_completed_and_carries_every_phase(completed):
    summary, _, root = completed
    manifest = ArtifactStore(root / "runs").open_run(summary.run_id).read_manifest()
    assert manifest.status is RunStatus.COMPLETED
    assert manifest.config_hash
    assert manifest.data_hash, "the content hash of the ingested bars must reach the manifest"
    assert set(manifest.phase_durations_s) == {"ingest", "features", "train", "report"}


def test_the_run_directory_satisfies_the_section_16_contract(completed):
    summary, _, _ = completed
    required = [
        "manifest.json",
        "config.resolved.yaml",
        "tearsheet.json",
        "quality.json",
        "features.json",
        "folds.json",
        "equity.parquet",
        "drawdown.parquet",
        "rolling_sharpe.parquet",
        "trades.parquet",
        "predictions.parquet",
        "ledger.parquet",
        "leaderboard.csv",
        "metrics.csv",
        "research.md",
        "logs.jsonl",
    ]
    missing = [name for name in required if not (summary.path / name).is_file()]
    assert not missing, f"§16 requires {missing}, which this run did not write"


def test_a_model_is_persisted_for_the_selected_row(completed):
    summary, _, _ = completed
    selected = summary.tearsheet.selected_model
    assert (summary.path / "models" / str(selected) / "model.pkl").is_file()


def test_the_tearsheet_validates_and_is_internally_consistent(completed):
    summary, _, _ = completed
    tearsheet = validate_tearsheet(json.loads((summary.path / TEARSHEET_FILENAME).read_bytes()))

    assert tearsheet.run_id == summary.run_id
    assert tearsheet.dataset.periods_per_year > 250
    assert tearsheet.cv is not None and len(tearsheet.cv.folds) >= 1
    assert tearsheet.features is not None and tearsheet.features.n_built > 5
    assert tearsheet.labels is not None and tearsheet.labels.binarised == "direction"
    assert tearsheet.backtest is not None
    assert tearsheet.selected_model == tearsheet.leaderboard[0].model


def test_every_leaderboard_row_carries_an_economic_column(completed):
    summary, _, _ = completed
    # The Phase 5 leaderboard left these empty and flagged NO_BACKTEST on every
    # row. Phase 10 is the phase that fills them.
    for row in summary.tearsheet.leaderboard:
        assert "NO_BACKTEST" not in row.flags
        assert row.oos_net_sharpe is not None
        assert row.dsr is not None


def test_the_break_even_cost_travels_with_the_headline_sharpe(completed):
    summary, _, _ = completed
    headline = summary.headline
    # §11.2 requires these two to appear together, always.
    assert headline["oos_net_sharpe"] is not None
    assert headline["break_even_bps"] is not None
    assert headline["assumed_round_trip_bps"] is not None


def test_the_cost_sensitivity_curve_covers_every_configured_level(completed, smoke_config):
    summary, _, _ = completed
    curve = summary.tearsheet.backtest.cost_sensitivity
    assert [point["bps"] for point in curve] == list(smoke_config.backtest.cost_sensitivity_bps)
    assert all("sharpe" in point for point in curve)


def test_the_shuffled_label_baseline_is_on_the_leaderboard(completed):
    summary, _, _ = completed
    names = {row.model for row in summary.tearsheet.leaderboard}
    assert "baseline_shuffled" in names, (
        "a leaderboard without the shuffled-label row has no scale on its axis"
    )


def test_no_model_beats_the_shuffled_baseline_by_a_convincing_margin(completed):
    """The honest headline §19 asks for.

    `samples/SYNTH_1d.csv` is a seeded random walk. Any model whose deflated
    Sharpe survives on it has found something that is not there.
    """
    summary, _, _ = completed
    for row in summary.tearsheet.leaderboard:
        assert row.dsr is not None
        assert row.dsr < 0.95, (
            f"{row.model} reports a deflated Sharpe of {row.dsr:.2f} on a synthetic "
            "random walk, which would mean the pipeline found an edge that cannot exist"
        )


def test_the_research_note_leads_with_what_the_run_does_not_establish(completed):
    summary, _, _ = completed
    note = (summary.path / "research.md").read_text(encoding="utf-8")
    assert "Not investment advice" in note
    assert "## Caveats" in note
    assert "baseline_shuffled" in note


def test_the_run_directory_is_immutable(completed):
    summary, _, root = completed
    reopened = ArtifactStore(root / "runs").open_run(summary.run_id)
    with pytest.raises(Exception, match="read-only"):
        reopened.write_text("tearsheet.json", "{}")


# --------------------------------------------------------------------------
# §12.4 — determinism
# --------------------------------------------------------------------------


def _comparable(payload: dict) -> str:
    """Strip the fields §12.4 excludes and render the rest canonically.

    Args:
        payload: A parsed `tearsheet.json`.

    Returns:
        Canonical JSON of everything that is supposed to be reproducible.
    """
    stripped = dict(payload)
    stripped.pop("created_at", None)
    manifest = dict(stripped.get("manifest", {}))
    for field in VOLATILE_MANIFEST_FIELDS:
        manifest.pop(field, None)
    stripped["manifest"] = manifest
    return json.dumps(stripped, sort_keys=True, separators=(",", ":"))


def test_two_runs_in_the_same_session_produce_an_identical_tearsheet(
    smoke_config, tmp_path_factory
):
    root = tmp_path_factory.mktemp("determinism")
    payloads = []
    for attempt in ("first", "second"):
        # A separate lake each time, so ingestion runs twice too. Reusing one
        # would leave the second run's quality section reading from the first
        # run's work, which is the half of the pipeline this check exists for.
        summary = run_experiment(
            smoke_config,
            data_dir=root / f"lake_{attempt}",
            store=ArtifactStore(root / attempt),
            run_id=f"fixed_{attempt}",
        )
        payloads.append(json.loads((summary.path / TEARSHEET_FILENAME).read_bytes()))

    # The run id is part of the document and is deliberately different per
    # directory, so it is compared separately rather than smuggled past.
    assert payloads[0]["run_id"] != payloads[1]["run_id"]
    for payload in payloads:
        payload["run_id"] = "pinned"
    assert _comparable(payloads[0]) == _comparable(payloads[1])


@pytest.mark.slow
def test_a_fresh_interpreter_produces_the_same_tearsheet(repo_root, smoke_config, tmp_path):
    """§12.4's second half: a new process, not just a new call.

    A result that depends on module import order, dictionary iteration, or a
    global RNG that happened to be warm would survive the same-session check
    and fail this one.
    """
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from alphacore.config import load_config
        from alphacore.pipelines import run_experiment
        from alphacore.runtime import ArtifactStore, configure_logging

        configure_logging("ERROR", colorize=False)
        root = Path(r"{repo_root}")
        config = load_config(root / "configs" / "experiments" / "smoke.yaml", use_env=False)
        data = config.data.model_copy(update={{"csv_dir": str(root / "samples")}})
        config = config.model_copy(update={{"data": data}})
        run_experiment(
            config,
            data_dir=Path(r"{tmp_path}") / "lake",
            store=ArtifactStore(Path(r"{tmp_path}") / "subprocess"),
            run_id="fixed_subprocess",
        )
        """
    )
    finished = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
        cwd=repo_root,
    )
    assert finished.returncode == 0, finished.stderr[-2000:]

    in_session = run_experiment(
        smoke_config,
        data_dir=tmp_path / "lake_session",
        store=ArtifactStore(tmp_path / "in_session"),
        run_id="fixed_subprocess",
    )

    subprocess_payload = json.loads(
        (tmp_path / "subprocess" / "fixed_subprocess" / TEARSHEET_FILENAME).read_bytes()
    )
    session_payload = json.loads((in_session.path / TEARSHEET_FILENAME).read_bytes())
    assert _comparable(subprocess_payload) == _comparable(session_payload)
