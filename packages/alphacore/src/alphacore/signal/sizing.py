"""Position sizing (§signal).

Three methods. Every one of them ends at the same clip: `|weight| <=
max_leverage`, always, with no configuration that can bypass it.

* **fixed_fraction** — constant exposure. Simple, and wrong in the way that
  matters: it takes the same risk in a calm market as in a crisis.
* **vol_target** — scale exposure inversely with realised volatility so
  risk, not notional, is held constant. This is what most of the improvement in
  a real strategy comes from, and it is why `realised_vol` is annualised from
  the calendar rather than a hardcoded 252 (§11.1).
* **kelly_capped** — size by edge, hard-capped. Uncapped Kelly is correct only
  if the estimated probability is correct, which it never is; a 1% error in a
  probability near 1 produces an enormous position. The cap is the whole
  device, not a safety rail bolted on afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alphacore.errors import BacktestError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from alphacore.config.schema import SizingConfig

__all__ = [
    "fixed_fraction_weight",
    "kelly_capped",
    "size_positions",
    "vol_target_weight",
]


def fixed_fraction_weight(fraction: float, *, max_leverage: float) -> float:
    """Constant exposure, clipped to the leverage cap.

    Args:
        fraction: Target fraction of equity.
        max_leverage: Absolute cap.

    Returns:
        The weight.
    """
    return float(min(abs(fraction), max_leverage))


def vol_target_weight(volatility: float, *, target_annual_vol: float, max_leverage: float) -> float:
    """Scale exposure so the position's volatility hits a target.

    `weight = target_annual_vol / realised_annual_vol`, clipped.

    Args:
        volatility: Realised annualised volatility of the instrument.
        target_annual_vol: Desired annualised portfolio volatility.
        max_leverage: Absolute cap.

    Returns:
        The weight. Zero when volatility is zero or undefined — a division by
        zero here would size infinitely on the quietest bar in the sample.
    """
    if not np.isfinite(volatility) or volatility <= 0.0:
        return 0.0
    return float(min(target_annual_vol / volatility, max_leverage))


def kelly_capped(proba: float, *, cap: float, payoff: float = 1.0) -> float:
    """Kelly fraction for a binary bet, hard-capped.

    `f = (p * (b + 1) - 1) / b` for win probability `p` and payoff `b`.

    Args:
        proba: Probability the bet wins.
        cap: Maximum fraction, regardless of what Kelly asks for.
        payoff: Win/loss payoff ratio.

    Returns:
        The fraction, in `[0, cap]`. A negative Kelly means the bet is
        unfavourable, which sizes to zero rather than flipping the position —
        the direction is the signal's decision, not the sizer's.
    """
    if not np.isfinite(proba) or payoff <= 0:
        return 0.0
    fraction = (proba * (payoff + 1.0) - 1.0) / payoff
    return float(np.clip(fraction, 0.0, cap))


def size_positions(
    signal: NDArray[np.int_],
    config: SizingConfig,
    *,
    volatility: NDArray[np.float64] | None = None,
    proba: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Convert a signal into portfolio weights.

    Args:
        signal: Positions in `{-1, 0, +1}`.
        config: The `signal.sizing` section.
        volatility: Realised annualised volatility per bar. Required for
            `vol_target`.
        proba: Calibrated probability per bar. Required for `kelly_capped`.

    Returns:
        Signed weights, every one satisfying `|w| <= max_leverage`.

    Raises:
        BacktestError: If a required input is missing or misaligned.
    """
    sides = np.asarray(signal, dtype=float)
    n = sides.size
    cap = config.max_leverage

    if config.method == "fixed_fraction":
        magnitude = np.full(n, fixed_fraction_weight(config.fixed_fraction, max_leverage=cap))
    elif config.method == "vol_target":
        if volatility is None:
            msg = (
                "vol_target sizing needs a realised volatility series. Build one with the "
                "`realised_vol` feature, whose annualisation comes from the calendar (§11.1)."
            )
            raise BacktestError(msg)
        vols = _aligned(volatility, n, "volatility")
        magnitude = np.array(
            [
                vol_target_weight(v, target_annual_vol=config.target_annual_vol, max_leverage=cap)
                for v in vols
            ]
        )
    else:
        if proba is None:
            msg = "kelly_capped sizing needs the calibrated probability series."
            raise BacktestError(msg)
        probs = _aligned(proba, n, "proba")
        # A short signal wins when the *negative* class realises, so the Kelly
        # probability is mirrored. Sizing on `p` regardless would systematically
        # oversize shorts.
        magnitude = np.array(
            [
                kelly_capped(p if side >= 0 else 1.0 - p, cap=config.kelly_cap)
                for p, side in zip(probs, sides, strict=True)
            ]
        )
        magnitude = np.minimum(magnitude, cap)

    weights = np.sign(sides) * np.abs(magnitude)
    return np.asarray(np.clip(weights, -cap, cap), dtype=np.float64)


def _aligned(values: NDArray[np.float64], n: int, name: str) -> NDArray[np.float64]:
    """Check a supporting series matches the signal length.

    Args:
        values: The series.
        n: Expected length.
        name: Parameter name for the error message.

    Returns:
        The series as float64.

    Raises:
        BacktestError: If the lengths disagree.
    """
    array = np.asarray(values, dtype=float)
    if array.size != n:
        msg = f"{name} has {array.size} entries for a signal of {n}."
        raise BacktestError(msg)
    return array
