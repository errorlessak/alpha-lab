"""Logging configuration.

Two sinks, deliberately different in kind:

* **stderr** — human-readable, for the person watching a run
* **`runs/{run_id}/logs.jsonl`** — one JSON object per line, for the API's SSE
  stream and for post-hoc forensics on a failed run

`print()` is banned repo-wide and enforced by ruff's T20 rules. A `print` cannot
carry a level, a timestamp, or a run id, so it cannot appear in the run's log
artifact — which makes it invisible to the exact person who needs it most.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Final

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "LOG_FILENAME",
    "add_run_sink",
    "configure_logging",
    "logger",
    "remove_sink",
]

LOG_FILENAME: Final[str] = "logs.jsonl"

CONSOLE_FORMAT: Final[str] = (
    "<dim>{time:HH:mm:ss}</dim> <level>{level: <8}</level> <cyan>{extra[phase]}</cyan> {message}"
)

DEFAULT_LEVEL: Final[str] = "INFO"


def configure_logging(level: str = DEFAULT_LEVEL, *, colorize: bool | None = None) -> int:
    """Reset logging and install the console sink.

    Removes loguru's default handler so the format is ours, and binds a `phase`
    field so the console format never fails on a record that lacks one.

    Args:
        level: Minimum level to emit, e.g. `"INFO"` or `"DEBUG"`.
        colorize: Force ANSI colour on or off. Defaults to loguru's TTY
            detection, which is the right behaviour when output is piped.

    Returns:
        The sink id of the console handler.
    """
    logger.remove()
    logger.configure(extra={"phase": "-"})
    return logger.add(
        sys.stderr,
        level=level,
        format=CONSOLE_FORMAT,
        colorize=colorize,
        backtrace=False,
        diagnose=False,
    )


def add_run_sink(run_dir: Path, *, level: str = "DEBUG") -> int:
    """Attach a JSON-lines sink writing into a run directory.

    The file is opened in append mode: `logs.jsonl` is the one artifact that
    grows during a run, and is exempt from the write-once rule in
    `alphacore.runtime.artifacts` for that reason.

    Args:
        run_dir: The run directory, which must already exist.
        level: Minimum level to record. Lower than the console default, because
            the file is where you look after something has already gone wrong.

    Returns:
        The sink id, for later removal with `remove_sink`.
    """
    return logger.add(
        run_dir / LOG_FILENAME,
        level=level,
        serialize=True,
        enqueue=False,
        backtrace=True,
        diagnose=False,
        mode="a",
        encoding="utf-8",
    )


def remove_sink(sink_id: int) -> None:
    """Detach a sink, tolerating one that has already been removed.

    Args:
        sink_id: A value previously returned by `configure_logging` or
            `add_run_sink`.
    """
    try:
        logger.remove(sink_id)
    except ValueError:
        return
