"""Turning calibrated probabilities into discrete positions (§signal).

Two devices, and the second is the one that matters economically.

**Thresholds** map a probability to `-1`, `0`, `+1`. The band between
`short_threshold` and `long_threshold` is a deliberate dead zone: a model that
is 51% confident has no opinion worth paying spread for.

**Hysteresis** widens that band asymmetrically depending on what is already
held. Entering a position requires clearing `threshold + hysteresis`; keeping
one only requires staying above `threshold - hysteresis`. Without it, a
probability oscillating around a threshold produces a flip on every bar, and
each flip pays commission, spread and impact. Turnover is the single largest
controllable drag on a short-horizon strategy, and this is the cheapest lever
on it.

Thresholds only mean anything on **calibrated** probabilities (§10.2). A raw
gradient-boosted score of 0.56 is not a 56% chance of anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alphacore.errors import BacktestError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alphacore.config.schema import SignalConfig

__all__ = ["apply_hysteresis", "count_flips", "generate_signal", "threshold_signal"]

FLAT = 0
LONG = 1
SHORT = -1


def _check_proba(proba: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate a probability series.

    Args:
        proba: Predicted probability of the positive class.

    Returns:
        The array as float64.

    Raises:
        BacktestError: If any value is non-finite or outside `[0, 1]`. A
            probability outside the unit interval means an uncalibrated score
            reached the signal layer, where every threshold would be meaningless.
    """
    values = np.asarray(proba, dtype=float)
    if values.size and (not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0):
        msg = (
            "Signal thresholds need calibrated probabilities in [0, 1]; got values outside "
            "that range. A raw model score is not a probability (§10.2)."
        )
        raise BacktestError(msg)
    return values


def threshold_signal(
    proba: NDArray[np.float64],
    *,
    long_threshold: float,
    short_threshold: float,
    allow_short: bool,
) -> NDArray[np.int_]:
    """Map probabilities to `{-1, 0, +1}` with no memory.

    Args:
        proba: Calibrated probability of the positive class.
        long_threshold: At or above this, go long.
        short_threshold: At or below this, go short.
        allow_short: False collapses short signals to flat.

    Returns:
        A signal array.

    Raises:
        BacktestError: If the probabilities are not in `[0, 1]`.
    """
    values = _check_proba(proba)
    signal = np.where(values >= long_threshold, LONG, FLAT)
    if allow_short:
        signal = np.where(values <= short_threshold, SHORT, signal)
    return signal.astype(int)


def apply_hysteresis(  # noqa: PLR0912 - the state machine is clearer written out
    proba: NDArray[np.float64],
    *,
    long_threshold: float,
    short_threshold: float,
    hysteresis: float,
    allow_short: bool,
) -> NDArray[np.int_]:
    """Apply thresholds with a hold band, so marginal readings do not flip.

    Entering needs `proba >= long_threshold + hysteresis` (or
    `<= short_threshold - hysteresis` for a short). Holding only needs
    `proba >= long_threshold - hysteresis`. The asymmetry is the point:
    conviction to enter, patience to leave.

    Args:
        proba: Calibrated probability of the positive class.
        long_threshold: Base long threshold.
        short_threshold: Base short threshold.
        hysteresis: Half-width of the band. Zero reproduces plain thresholding
            exactly.
        allow_short: False collapses short signals to flat.

    Returns:
        A signal array.

    Raises:
        BacktestError: If the probabilities are not in `[0, 1]`, or the
            hysteresis is negative.
    """
    if hysteresis < 0:
        msg = f"hysteresis must be non-negative, got {hysteresis}."
        raise BacktestError(msg)
    values = _check_proba(proba)
    if hysteresis == 0.0:
        return threshold_signal(
            values,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            allow_short=allow_short,
        )

    enter_long = long_threshold + hysteresis
    hold_long = long_threshold - hysteresis
    enter_short = short_threshold - hysteresis
    hold_short = short_threshold + hysteresis

    # Sequential by necessity: each decision depends on the position carried in
    # from the previous bar, which is exactly what "hysteresis" means.
    out = np.zeros(values.size, dtype=int)
    position = FLAT
    for i, value in enumerate(values):
        if position == LONG:
            if value >= hold_long:
                pass
            elif allow_short and value <= enter_short:
                position = SHORT
            else:
                position = FLAT
        elif position == SHORT:
            if value <= hold_short:
                pass
            elif value >= enter_long:
                position = LONG
            else:
                position = FLAT
        elif value >= enter_long:
            position = LONG
        elif allow_short and value <= enter_short:
            position = SHORT
        out[i] = position
    return out


def count_flips(signal: NDArray[np.int_]) -> int:
    """Count how many times the position changes.

    Args:
        signal: A signal array.

    Returns:
        The number of bar-to-bar changes. This is the quantity the cost model
        charges for, so it is the quantity hysteresis is judged on.
    """
    values = np.asarray(signal)
    if values.size < 2:  # noqa: PLR2004
        return 0
    return int((np.diff(values) != 0).sum())


def generate_signal(proba: NDArray[np.float64], config: SignalConfig) -> NDArray[np.int_]:
    """Produce the configured signal from calibrated probabilities.

    Args:
        proba: Calibrated probability of the positive class.
        config: The `signal` section.

    Returns:
        A signal array.

    Raises:
        BacktestError: If the probabilities are invalid.
    """
    return apply_hysteresis(
        proba,
        long_threshold=config.long_threshold,
        short_threshold=config.short_threshold,
        hysteresis=config.hysteresis,
        allow_short=config.allow_short,
    )
