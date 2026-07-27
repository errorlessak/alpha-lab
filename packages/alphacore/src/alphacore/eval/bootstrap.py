"""Stationary block bootstrap confidence intervals (§11.2).

§11.2 asks for "the interval, not just the point estimate", and for the block
length to come from Politis-White rather than being picked by hand.

Blocks, not individual observations. Resampling returns one at a time destroys
the serial dependence that financial returns actually have — volatility
clusters, and overlapping labels make consecutive observations share a price
path. An i.i.d. bootstrap on that data produces an interval far too narrow,
which is the opposite of what a confidence interval is for.

The block length is chosen from the data's own autocorrelation, so a strongly
autocorrelated series gets longer blocks automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from alphacore.errors import EvaluationError
from alphacore.runtime.determinism import deterministic_rng

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

__all__ = ["block_bootstrap_ci", "politis_white_block_length", "stationary_bootstrap_sample"]

DEFAULT_SAMPLES: Final[int] = 1_000
_MIN_OBSERVATIONS: Final[int] = 20
_MAX_LAG_FRACTION: Final[float] = 0.25


def politis_white_block_length(returns: NDArray[np.float64]) -> int:
    """Choose a block length from the series' own autocorrelation.

    Uses the significant-autocorrelation rule: find the largest lag whose
    autocorrelation exceeds the white-noise band, and scale it. A series with
    no autocorrelation gets a block of 1, which reduces to the i.i.d. case.

    Args:
        returns: The series to bootstrap.

    Returns:
        A block length of at least 1, never more than a quarter of the sample.

    Raises:
        EvaluationError: If the series is too short to estimate from.
    """
    values = np.asarray(returns, dtype=float)
    n = values.size
    if n < _MIN_OBSERVATIONS:
        msg = f"Need at least {_MIN_OBSERVATIONS} observations to choose a block length, got {n}."
        raise EvaluationError(msg)

    centred = values - values.mean()
    variance = float(np.dot(centred, centred))
    if variance == 0:
        return 1

    max_lag = max(1, int(min(n * _MAX_LAG_FRACTION, np.ceil(2.0 * np.sqrt(n)))))
    threshold = 2.0 / np.sqrt(n)

    significant = 0
    for lag in range(1, max_lag + 1):
        rho = float(np.dot(centred[:-lag], centred[lag:]) / variance)
        if abs(rho) > threshold:
            significant = lag

    # Politis-White scales the significant lag by roughly n^(1/3).
    block = int(np.ceil((significant + 1) * n ** (1.0 / 3.0) / 4.0)) if significant else 1
    return int(np.clip(block, 1, max(1, n // 4)))


def stationary_bootstrap_sample(
    values: NDArray[np.float64], *, block: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Draw one stationary-bootstrap resample.

    Blocks start at uniformly random positions and wrap around the end of the
    series, which is what keeps the resampled series stationary rather than
    under-weighting the tail.

    Args:
        values: The series to resample.
        block: Block length.
        rng: Seeded generator.

    Returns:
        A resample of the same length as `values`.
    """
    n = values.size
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block)
    picks = (starts[:, None] + offsets[None, :]).ravel()[:n] % n
    return np.asarray(values[picks], dtype=np.float64)


def block_bootstrap_ci(
    returns: NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64]], float],
    *,
    seed: int,
    n_samples: int = DEFAULT_SAMPLES,
    block: int | None = None,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for a statistic of a return series.

    Args:
        returns: The series.
        statistic: A function of the series, e.g. a Sharpe ratio.
        seed: Run seed. The bootstrap draws from its own stream, so adding a
            bootstrap cannot perturb any other random draw in the run (§4, I8).
        n_samples: Bootstrap replicates.
        block: Block length. None chooses one via Politis-White.
        confidence: Interval width, e.g. 0.95 for a 2.5/97.5 interval.

    Returns:
        `(lower, upper)`.

    Raises:
        EvaluationError: If the series is too short, or the statistic could not
            be computed on any resample — which means the interval would be
            invented rather than estimated.
    """
    values = np.asarray(returns, dtype=float)
    if values.size < _MIN_OBSERVATIONS:
        msg = f"Need at least {_MIN_OBSERVATIONS} observations to bootstrap, got {values.size}."
        raise EvaluationError(msg)

    length = block if block is not None else politis_white_block_length(values)
    rng = deterministic_rng(seed, "block_bootstrap", n_samples, length)

    estimates: list[float] = []
    for _ in range(n_samples):
        resample = stationary_bootstrap_sample(values, block=length, rng=rng)
        try:
            estimate = statistic(resample)
        except EvaluationError:
            # A degenerate resample (zero volatility, say) contributes nothing
            # rather than a fabricated value.
            continue
        if np.isfinite(estimate):
            estimates.append(float(estimate))

    if not estimates:
        msg = (
            "The statistic could not be computed on any bootstrap resample, so no interval "
            "can be reported. Reporting one anyway would be inventing it."
        )
        raise EvaluationError(msg)

    tail = (1.0 - confidence) / 2.0 * 100.0
    return (
        float(np.percentile(estimates, tail)),
        float(np.percentile(estimates, 100.0 - tail)),
    )
