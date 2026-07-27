"""Stop-loss, take-profit, and trailing stops (§risk).

Levels are set in ATR multiples rather than fixed percentages, so a stop is
equally far away in risk terms whether the instrument is moving 0.5% or 5% a
day. A fixed percentage stop is really a volatility-varying stop wearing a
disguise.

**The tie-break.** When a bar's range touches both the stop and the target,
the stop is taken. Nothing in an OHLC bar says which came first, and assuming
the favourable one is exactly how a backtest manufactures an edge (§12.1). The
same rule governs triple-barrier labelling, and it has to be the same rule, or
the labels and the backtest disagree about what happened.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphacore.config.schema import RiskConfig

__all__ = ["ExitReason", "StopLevels", "check_stop_exit", "stop_levels", "update_trailing_stop"]


class ExitReason(StrEnum):
    """Why a position was closed."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    SIGNAL = "signal"
    KILL_SWITCH = "kill_switch"
    END_OF_DATA = "end_of_data"


@dataclass(frozen=True)
class StopLevels:
    """The price levels guarding one open position.

    Attributes:
        stop: Adverse exit price, or None if no stop is configured.
        target: Favourable exit price, or None.
        side: `+1` long, `-1` short.
    """

    stop: float | None
    target: float | None
    side: int


def stop_levels(*, entry_price: float, side: int, atr: float, config: RiskConfig) -> StopLevels:
    """Compute the initial stop and target for a new position.

    Args:
        entry_price: Fill price.
        side: `+1` long, `-1` short.
        atr: Average true range at entry, in price units.
        config: The `risk` section.

    Returns:
        The levels. Either may be None when unconfigured.
    """
    stop = None
    target = None
    if config.stop_loss_atr is not None:
        stop = entry_price - side * config.stop_loss_atr * atr
    if config.take_profit_atr is not None:
        target = entry_price + side * config.take_profit_atr * atr
    return StopLevels(stop=stop, target=target, side=side)


def check_stop_exit(levels: StopLevels, *, high: float, low: float) -> ExitReason | None:
    """Decide whether a bar's range triggers an exit.

    Args:
        levels: The position's current levels.
        high: Bar high.
        low: Bar low.

    Returns:
        The reason, or None if neither level was touched. When both are touched
        in the same bar the **stop** wins (§12.1).
    """
    hit_stop = _stop_touched(levels, high=high, low=low)
    hit_target = _target_touched(levels, high=high, low=low)

    if hit_stop:
        return ExitReason.STOP_LOSS
    if hit_target:
        return ExitReason.TAKE_PROFIT
    return None


def _stop_touched(levels: StopLevels, *, high: float, low: float) -> bool:
    """Whether the bar traded through the stop.

    Args:
        levels: The position's levels.
        high: Bar high.
        low: Bar low.

    Returns:
        True if the stop was touched.
    """
    if levels.stop is None:
        return False
    return low <= levels.stop if levels.side > 0 else high >= levels.stop


def _target_touched(levels: StopLevels, *, high: float, low: float) -> bool:
    """Whether the bar traded through the target.

    Args:
        levels: The position's levels.
        high: Bar high.
        low: Bar low.

    Returns:
        True if the target was touched.
    """
    if levels.target is None:
        return False
    return high >= levels.target if levels.side > 0 else low <= levels.target


def update_trailing_stop(
    levels: StopLevels, *, price: float, atr: float, config: RiskConfig
) -> StopLevels:
    """Ratchet a trailing stop in the favourable direction only.

    A trailing stop that could move against the position would not be a stop;
    it would be a way to keep losing trades open.

    Args:
        levels: Current levels.
        price: Latest price to trail from.
        atr: Current average true range.
        config: The `risk` section.

    Returns:
        The levels, with the stop tightened if the price has moved favourably.
        Returned unchanged when no trailing stop is configured.
    """
    if config.trailing_stop_atr is None:
        return levels

    candidate = price - levels.side * config.trailing_stop_atr * atr
    if levels.stop is None:
        return replace(levels, stop=candidate)

    tightened = max(levels.stop, candidate) if levels.side > 0 else min(levels.stop, candidate)
    return replace(levels, stop=tightened)
