"""Wall-clock timing for pipeline phases.

Durations are reported in the run manifest and the tearsheet. They are the one
class of value that is deliberately *excluded* from the determinism comparison
(§12.4) — the same run must produce identical metrics, not identical runtimes.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["PhaseTimings", "Timer", "timed"]


@dataclass
class Timer:
    """A monotonic stopwatch.

    Monotonic rather than wall-clock, so a clock adjustment mid-run cannot
    produce a negative duration.

    Attributes:
        elapsed_s: Seconds measured. Zero until the timer has been stopped.
    """

    elapsed_s: float = 0.0
    _start: float | None = field(default=None, repr=False)

    def start(self) -> None:
        """Begin timing, discarding any previous measurement."""
        self._start = time.perf_counter()
        self.elapsed_s = 0.0

    def stop(self) -> float:
        """Stop timing and record the interval.

        Returns:
            Seconds elapsed since `start`.

        Raises:
            RuntimeError: If the timer was never started.
        """
        if self._start is None:
            msg = "Timer.stop() called before Timer.start()."
            raise RuntimeError(msg)
        self.elapsed_s = time.perf_counter() - self._start
        self._start = None
        return self.elapsed_s


@contextmanager
def timed() -> Iterator[Timer]:
    """Time a block of work.

    Yields:
        A `Timer` whose `elapsed_s` is populated when the block exits. The
        duration is recorded even if the block raises, so a failed run still
        reports how long it ran before it failed.

    Example:
        >>> with timed() as t:
        ...     total = sum(range(1000))
        >>> t.elapsed_s >= 0.0
        True
    """
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


@dataclass
class PhaseTimings:
    """Ordered record of how long each pipeline phase took.

    Insertion order is preserved so the manifest reports phases in the order
    they actually ran, which is also the order the UI's progress display uses.
    """

    durations_s: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str) -> Iterator[Timer]:
        """Time one named phase and record it.

        Args:
            name: Phase label, e.g. `"features"`.

        Yields:
            The `Timer` measuring this phase.

        Raises:
            RuntimeError: If a phase with this name was already recorded.
        """
        if name in self.durations_s:
            msg = f"Phase {name!r} has already been timed; phase names must be unique."
            raise RuntimeError(msg)
        timer = Timer()
        timer.start()
        try:
            yield timer
        finally:
            timer.stop()
            self.durations_s[name] = timer.elapsed_s

    @property
    def total_s(self) -> float:
        """Sum of all recorded phase durations, in seconds."""
        return sum(self.durations_s.values())
