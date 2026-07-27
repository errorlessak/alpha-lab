"""Shared provider machinery: retries and raw-response caching (§6.1).

Every provider retries with exponential backoff plus jitter, up to five
attempts. Jitter matters more than it looks: without it, a paginated pull that
trips a rate limit re-sends every page on the same schedule and trips it again.

Retry delays use their own RNG stream and deliberately do *not* consume the run
seed. Wall-clock backoff cannot change a computed result, and drawing from the
seeded stream here would make the sequence of random numbers a model sees
depend on how flaky the network was.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Final, TypeVar

from alphacore.errors import ProviderError
from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DEFAULT_BACKOFF_BASE_S",
    "DEFAULT_MAX_ATTEMPTS",
    "RETRYABLE_STATUS_CODES",
    "retry",
]

T = TypeVar("T")

DEFAULT_MAX_ATTEMPTS: Final[int] = 5
DEFAULT_BACKOFF_BASE_S: Final[float] = 0.5
DEFAULT_BACKOFF_CAP_S: Final[float] = 30.0

RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})
"""HTTP statuses worth retrying: rate limits and transient server faults. A 404
or a 401 is retried by nobody sensible — the answer will not change."""

_JITTER_RNG: Final[random.Random] = random.Random(0)


def _backoff_delay(attempt: int, *, base_s: float, cap_s: float) -> float:
    """Compute a jittered exponential backoff delay.

    Args:
        attempt: 1-based attempt number that just failed.
        base_s: Delay after the first failure.
        cap_s: Upper bound on the delay.

    Returns:
        Seconds to wait, drawn uniformly from `[0, min(cap, base * 2^(n-1))]`
        — "full jitter", which spreads retries instead of synchronising them.
    """
    ceiling = min(cap_s, base_s * (2 ** (attempt - 1)))
    return _JITTER_RNG.uniform(0.0, ceiling)


def retry[T](
    operation: Callable[[], T],
    *,
    description: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_s: float = DEFAULT_BACKOFF_BASE_S,
    cap_s: float = DEFAULT_BACKOFF_CAP_S,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run an operation, retrying transient failures with jittered backoff.

    Args:
        operation: The zero-argument call to attempt.
        description: Human-readable label used in log lines and the final error.
        max_attempts: Total attempts, including the first.
        base_s: Base backoff delay.
        cap_s: Maximum backoff delay.
        retry_on: Exception types considered transient.
        sleep: Sleep function. Injectable so tests do not actually wait.

    Returns:
        Whatever `operation` returns.

    Raises:
        ProviderError: If every attempt fails. The final failure is chained so
            the original traceback survives.
    """
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except retry_on as exc:
            last = exc
            if attempt == max_attempts:
                break
            delay = _backoff_delay(attempt, base_s=base_s, cap_s=cap_s)
            logger.bind(phase="ingest").warning(
                "{} failed (attempt {}/{}): {} — retrying in {:.2f}s",
                description,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            sleep(delay)

    msg = f"{description} failed after {max_attempts} attempts: {last}"
    raise ProviderError(msg) from last
