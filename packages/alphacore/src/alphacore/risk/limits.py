"""Portfolio-level risk limits and the kill switch (§risk).

Four limits, checked every bar:

* **max drawdown** — trips the kill switch and **latches**. A drawdown limit
  that resets when equity recovers is not a limit; it is a description of what
  happened. Once tripped, the run is over and the tearsheet says why.
* **max daily loss** — resets each session, so it constrains a bad day rather
  than a bad decade.
* **max concurrent positions** — caps how many bets are open at once.
* **leverage** — enforced in the sizer (§signal), listed here because a
  reviewer looks for all four in one place.

Breaches accumulate rather than short-circuit. Reporting only the first one
would hide that a run breached three limits simultaneously, which is a
different situation from breaching one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from alphacore.runtime.logging import logger

if TYPE_CHECKING:
    from alphacore.config.schema import RiskConfig

__all__ = ["KillSwitch", "LimitBreach", "RiskLimiter"]


class LimitBreach(StrEnum):
    """Which limit was breached."""

    MAX_DRAWDOWN = "max_drawdown"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_CONCURRENT = "max_concurrent_positions"


@dataclass(frozen=True)
class KillSwitch:
    """The record of a run being stopped by a risk limit.

    Attributes:
        reason: Which limit tripped it.
        equity: Equity at the moment it tripped.
        peak_equity: High-water mark up to that point.
        drawdown: Fractional drawdown from the peak, negative.
    """

    reason: LimitBreach
    equity: float
    peak_equity: float
    drawdown: float

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-compatible mapping for the tearsheet.

        Returns:
            The mapping.
        """
        return {
            "reason": self.reason.value,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "drawdown": self.drawdown,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KillSwitch:
        """Rebuild from its serialised form.

        Args:
            payload: A mapping produced by `to_dict`.

        Returns:
            The kill switch record.
        """
        return cls(
            reason=LimitBreach(payload["reason"]),
            equity=float(payload["equity"]),
            peak_equity=float(payload["peak_equity"]),
            drawdown=float(payload["drawdown"]),
        )


class RiskLimiter:
    """Tracks equity and reports which limits a bar breaches."""

    def __init__(self, config: RiskConfig, *, initial_equity: float) -> None:
        """Start tracking from an initial equity.

        Args:
            config: The `risk` section.
            initial_equity: Starting capital.
        """
        self.config = config
        self.peak_equity = float(initial_equity)
        self.session_open_equity = float(initial_equity)
        self.kill_switch: KillSwitch | None = None

    def __repr__(self) -> str:
        """Return a debug representation naming the state."""
        return f"RiskLimiter(peak={self.peak_equity:.2f}, killed={self.killed})"

    @property
    def killed(self) -> bool:
        """Whether the kill switch has tripped."""
        return self.kill_switch is not None

    def start_session(self, *, equity: float) -> None:
        """Mark the start of a new trading session.

        Args:
            equity: Equity at the session open. The daily loss limit is
                measured against this.
        """
        self.session_open_equity = float(equity)

    def check(self, *, equity: float, open_positions: int) -> list[LimitBreach]:
        """Evaluate every limit against the current state.

        Args:
            equity: Current mark-to-market equity.
            open_positions: Positions currently open.

        Returns:
            Every breach, in a stable order. Empty when trading may continue.
        """
        self.peak_equity = max(self.peak_equity, float(equity))
        breaches: list[LimitBreach] = []

        drawdown = (equity - self.peak_equity) / self.peak_equity if self.peak_equity else 0.0
        if drawdown <= -self.config.max_drawdown_kill:
            breaches.append(LimitBreach.MAX_DRAWDOWN)
            if self.kill_switch is None:
                self.kill_switch = KillSwitch(
                    reason=LimitBreach.MAX_DRAWDOWN,
                    equity=float(equity),
                    peak_equity=self.peak_equity,
                    drawdown=float(drawdown),
                )
                logger.bind(phase="backtest").warning(
                    "kill switch: drawdown {:.1%} breached the {:.1%} limit",
                    drawdown,
                    -self.config.max_drawdown_kill,
                )
        elif self.killed:
            # Latched: recovering does not resume trading (§risk).
            breaches.append(LimitBreach.MAX_DRAWDOWN)

        if self.session_open_equity:
            session_return = (equity - self.session_open_equity) / self.session_open_equity
            if session_return <= -self.config.max_daily_loss:
                breaches.append(LimitBreach.MAX_DAILY_LOSS)

        if open_positions > self.config.max_concurrent_positions:
            breaches.append(LimitBreach.MAX_CONCURRENT)

        return breaches
