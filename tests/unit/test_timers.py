"""Phase timing."""

from __future__ import annotations

import pytest

from alphacore.runtime.timers import PhaseTimings, Timer, timed


def test_timer_records_a_non_negative_interval():
    timer = Timer()
    timer.start()
    elapsed = timer.stop()
    assert elapsed >= 0.0
    assert timer.elapsed_s == elapsed


def test_timer_refuses_to_stop_before_it_starts():
    with pytest.raises(RuntimeError, match=r"before Timer\.start"):
        Timer().stop()


def test_timer_restart_discards_the_previous_measurement():
    timer = Timer()
    timer.start()
    timer.stop()
    timer.start()
    assert timer.elapsed_s == 0.0


def test_timed_context_populates_elapsed():
    with timed() as timer:
        pass
    assert timer.elapsed_s >= 0.0


def test_timed_context_records_duration_even_when_the_block_raises():
    with pytest.raises(ValueError, match="boom"), timed() as timer:
        msg = "boom"
        raise ValueError(msg)
    assert timer.elapsed_s >= 0.0


def test_phase_timings_preserve_insertion_order():
    timings = PhaseTimings()
    for name in ("ingest", "features", "train"):
        with timings.phase(name):
            pass
    assert list(timings.durations_s) == ["ingest", "features", "train"]


def test_phase_timings_total_sums_phases():
    timings = PhaseTimings()
    with timings.phase("a"):
        pass
    with timings.phase("b"):
        pass
    assert timings.total_s == pytest.approx(sum(timings.durations_s.values()))


def test_phase_timings_reject_a_duplicate_phase_name():
    timings = PhaseTimings()
    with timings.phase("ingest"):
        pass
    with pytest.raises(RuntimeError, match="already been timed"), timings.phase("ingest"):
        pass


def test_phase_timings_record_a_failed_phase():
    timings = PhaseTimings()
    with pytest.raises(ValueError, match="boom"), timings.phase("train"):
        msg = "boom"
        raise ValueError(msg)
    assert "train" in timings.durations_s
