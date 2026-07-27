"""Logging sinks."""

from __future__ import annotations

import json

from alphacore.runtime.logging import (
    LOG_FILENAME,
    add_run_sink,
    configure_logging,
    logger,
    remove_sink,
)


def test_run_sink_writes_one_json_object_per_line(tmp_path):
    sink_id = add_run_sink(tmp_path)
    try:
        logger.bind(phase="features").info("built {} features", 214)
    finally:
        remove_sink(sink_id)

    lines = (tmp_path / LOG_FILENAME).read_text("utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])["record"]
    assert record["message"] == "built 214 features"
    assert record["level"]["name"] == "INFO"
    assert record["extra"]["phase"] == "features"


def test_run_sink_appends_across_attachments(tmp_path):
    for message in ("first", "second"):
        sink_id = add_run_sink(tmp_path)
        logger.info(message)
        remove_sink(sink_id)
    assert len((tmp_path / LOG_FILENAME).read_text("utf-8").splitlines()) == 2


def test_records_without_a_phase_still_format(capsys):
    configure_logging("INFO", colorize=False)
    logger.info("no phase bound")
    assert "no phase bound" in capsys.readouterr().err


def test_console_level_filters_lower_records(capsys):
    configure_logging("WARNING", colorize=False)
    logger.debug("invisible")
    logger.warning("visible")
    captured = capsys.readouterr().err
    assert "invisible" not in captured
    assert "visible" in captured


def test_remove_sink_is_idempotent(tmp_path):
    sink_id = add_run_sink(tmp_path)
    remove_sink(sink_id)
    remove_sink(sink_id)
