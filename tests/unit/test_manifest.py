"""Run manifests and run ids."""

from __future__ import annotations

import datetime as dt
import math

import pytest

from alphacore.errors import DeterminismError, SerialisationError
from alphacore.runtime.manifest import (
    RUN_ID_PATTERN,
    EnvironmentInfo,
    RunManifest,
    RunStatus,
    capture_environment,
    git_revision,
    new_run_id,
    slugify,
)

CONFIG = {"meta": {"name": "spy-daily-tb", "seed": 20260727}, "backtest": {"commission_bps": 0.5}}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SPY daily / triple barrier", "spy-daily-triple-barrier"),
        ("  BTC_1h  meta  ", "btc-1h-meta"),
        ("already-a-slug", "already-a-slug"),
        ("!!!", "run"),
        ("", "run"),
    ],
)
def test_slugify(raw, expected):
    assert slugify(raw) == expected


def test_new_run_id_matches_the_documented_shape(frozen_now):
    run_id = new_run_id("spy daily tb", "a3f91c7e2b4d5f60", now=frozen_now)
    assert run_id == "2026-07-27T14-22-01Z_spy-daily-tb_a3f91c"
    assert RUN_ID_PATTERN.match(run_id)


def test_new_run_id_normalises_to_utc():
    naive_offset = dt.datetime(2026, 7, 27, 16, 22, 1, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    assert new_run_id("x", "abcdef01", now=naive_offset).startswith("2026-07-27T14-22-01Z")


def test_run_ids_sort_reverse_chronologically():
    early = new_run_id("x", "abcdef01", now=dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    late = new_run_id("x", "abcdef01", now=dt.datetime(2026, 6, 1, tzinfo=dt.UTC))
    assert sorted([early, late], reverse=True) == [late, early]


def test_capture_environment_reports_this_interpreter():
    env = capture_environment()
    assert env.implementation
    assert env.python.count(".") == 2
    assert env.host
    assert "alphacore" in env.packages


def test_environment_info_is_frozen():
    env = capture_environment()
    with pytest.raises(ValueError, match="frozen"):
        env.python = "3.11.0"


def test_environment_info_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Extra inputs"):
        EnvironmentInfo(
            python="3.12.0",
            implementation="CPython",
            platform="test",
            host="test",
            unexpected=1,
        )


def test_manifest_start_records_hashes_and_running_status(frozen_now):
    manifest = RunManifest.start(name="spy-daily-tb", seed=20260727, config=CONFIG, now=frozen_now)
    assert manifest.status is RunStatus.RUNNING
    assert manifest.created_at == frozen_now
    assert len(manifest.config_hash) == 32
    assert manifest.run_id.endswith(manifest.config_hash[:6])
    assert manifest.finished_at is None
    assert manifest.duration_s is None


def test_manifest_config_hash_is_order_independent(frozen_now):
    reordered = {"backtest": CONFIG["backtest"], "meta": CONFIG["meta"]}
    a = RunManifest.start(name="x", seed=1, config=CONFIG, now=frozen_now)
    b = RunManifest.start(name="x", seed=1, config=reordered, now=frozen_now)
    assert a.config_hash == b.config_hash


def test_manifest_config_hash_moves_with_the_config(frozen_now):
    changed = {**CONFIG, "backtest": {"commission_bps": 2.0}}
    a = RunManifest.start(name="x", seed=1, config=CONFIG, now=frozen_now)
    b = RunManifest.start(name="x", seed=1, config=changed, now=frozen_now)
    assert a.config_hash != b.config_hash


def test_manifest_start_rejects_a_non_finite_config(frozen_now):
    with pytest.raises(SerialisationError):
        RunManifest.start(name="x", seed=1, config={"sharpe": math.nan}, now=frozen_now)


def test_manifest_start_rejects_an_invalid_seed(frozen_now):
    with pytest.raises(DeterminismError):
        RunManifest.start(name="x", seed=-1, config=CONFIG, now=frozen_now)


def test_manifest_start_accepts_an_explicit_run_id(frozen_now):
    manifest = RunManifest.start(
        name="x", seed=1, config=CONFIG, run_id="fixed-run", now=frozen_now
    )
    assert manifest.run_id == "fixed-run"


def test_manifest_finish_produces_a_terminal_copy(frozen_now):
    manifest = RunManifest.start(name="x", seed=1, config=CONFIG, now=frozen_now)
    done = manifest.finish(
        RunStatus.COMPLETED,
        now=frozen_now + dt.timedelta(seconds=412.7),
        phase_durations_s={"train": 1.0},
    )
    assert done.status is RunStatus.COMPLETED
    assert done.duration_s == pytest.approx(412.7)
    assert done.phase_durations_s == {"train": 1.0}
    # The original is untouched: an in-flight manifest can never be mistaken
    # for a finished one.
    assert manifest.status is RunStatus.RUNNING


def test_manifest_finish_carries_the_traceback_on_failure(frozen_now):
    manifest = RunManifest.start(name="x", seed=1, config=CONFIG, now=frozen_now)
    failed = manifest.finish(RunStatus.FAILED, error="Traceback...", now=frozen_now)
    assert failed.status is RunStatus.FAILED
    assert failed.error == "Traceback..."


def test_manifest_finish_refuses_a_non_terminal_status(frozen_now):
    manifest = RunManifest.start(name="x", seed=1, config=CONFIG, now=frozen_now)
    with pytest.raises(ValueError, match="terminal status"):
        manifest.finish(RunStatus.RUNNING)


def test_manifest_round_trips_through_json(frozen_now):
    manifest = RunManifest.start(name="x", seed=1, config=CONFIG, now=frozen_now)
    assert RunManifest.model_validate(manifest.model_dump(mode="json")) == manifest


def test_manifest_rejects_unknown_keys(frozen_now):
    manifest = RunManifest.start(name="x", seed=1, config=CONFIG, now=frozen_now)
    payload = {**manifest.model_dump(mode="json"), "sharpe": 3.0}
    with pytest.raises(ValueError, match="Extra inputs"):
        RunManifest.model_validate(payload)


def test_git_revision_returns_none_outside_a_repository(tmp_path):
    assert git_revision(tmp_path) is None


def test_git_revision_inside_this_repository(repo_root):
    revision = git_revision(repo_root)
    assert revision is None or revision
