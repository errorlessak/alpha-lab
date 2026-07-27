"""Cash, position, and mark-to-market accounting (§12).

Deliberately small. Equity is always `cash + quantity * price` — there is no
separate running total that could drift away from the components, because a
running total is exactly what drifts.

`assert_reconciles` exists so the engine can check that identity every bar
under `--strict` (§15, Phase 7). An accounting bug in a backtester does not
announce itself; it shows up as a Sharpe ratio that is slightly too good.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from alphacore.errors import BacktestError

__all__ = ["Portfolio"]

_RECONCILE_TOLERANCE: Final[float] = 1e-6


@dataclass
class Portfolio:
    """A single-instrument cash-and-position account.

    Attributes:
        cash: Uninvested currency.
        quantity: Signed position size in shares.
        total_costs: Cumulative costs charged, for the tearsheet.
        n_fills: How many fills have been executed.
    """

    initial_capital: float
    cash: float = field(init=False)
    quantity: float = 0.0
    total_costs: float = 0.0
    n_fills: int = 0

    def __post_init__(self) -> None:
        """Start fully in cash."""
        self.cash = float(self.initial_capital)

    def equity(self, *, price: float) -> float:
        """Mark the account to a price.

        Args:
            price: Current instrument price.

        Returns:
            `cash + quantity * price`.
        """
        return float(self.cash + self.quantity * price)

    def position_value(self, *, price: float) -> float:
        """Signed mark-to-market value of the position.

        Args:
            price: Current instrument price.

        Returns:
            `quantity * price`.
        """
        return float(self.quantity * price)

    def execute(self, *, quantity: float, price: float, cost: float) -> None:
        """Apply a fill.

        Args:
            quantity: Signed filled quantity.
            price: Fill price.
            cost: Transaction cost, taken straight out of cash.
        """
        self.cash -= quantity * price + cost
        self.quantity += quantity
        self.total_costs += cost
        if quantity != 0.0:
            self.n_fills += 1

    def charge(self, amount: float) -> None:
        """Deduct a carry cost from cash.

        Args:
            amount: Cost to charge.
        """
        self.cash -= amount
        self.total_costs += amount

    def target_quantity(self, *, weight: float, price: float, equity: float) -> float:
        """Convert a portfolio weight into a target share count.

        Args:
            weight: Signed target weight of equity.
            price: Current price.
            equity: Current equity.

        Returns:
            The target position in shares, or the current position when the
            price is unusable.
        """
        if price <= 0 or not equity:
            return self.quantity
        return float(weight * equity / price)

    def assert_reconciles(self, *, price: float, expected_equity: float) -> None:
        """Check equity equals cash plus position value.

        Args:
            price: Price to mark at.
            expected_equity: Independently computed equity.

        Raises:
            BacktestError: If the two disagree beyond tolerance. The message
                names both figures, because the difference is the size of the
                accounting bug.
        """
        actual = self.equity(price=price)
        if abs(actual - expected_equity) > _RECONCILE_TOLERANCE * max(1.0, abs(actual)):
            msg = (
                f"Portfolio failed to reconcile: cash {self.cash:.6f} + position "
                f"{self.quantity:.6f} x {price:.6f} = {actual:.6f}, but equity was "
                f"{expected_equity:.6f}. The difference is an accounting bug, not rounding."
            )
            raise BacktestError(msg)
