"""The cost model (§12.2).

Every interesting question about a backtest is a question about this file. A
vendored engine hides the answer somewhere in its source under flags whose
defaults assume the friendliest case; here it is one page you can read.

```
cost_per_share = commission_bps/1e4 * price
               + half_spread_bps/1e4 * price
               + impact(price, qty, adv)

impact_sqrt   = coefficient * price * sigma * sqrt(qty / adv)
impact_linear = coefficient * price * (qty / adv)
```

plus `max(min_commission, ...)` per trade, borrow accrued daily on short
notional, and financing on leveraged long notional.

**Participation.** An order larger than `max_participation * bar_volume` is
partially filled. Silently filling 100% of a bar's volume at the open is a
fantasy, and the engine does not permit it — it is the difference between a
strategy that works on paper and one that cannot be traded at size.

Costs are never zero by default. A frictionless configuration is legitimate as
a diagnostic — it is what the reference cross-check in §12.5 runs under — but
`is_frictionless` makes it detectable so the run is flagged `UNCOSTED` (I10).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from alphacore.config.schema import CostsConfig

__all__ = ["BPS", "CostModel", "impact_cost", "round_trip_bps"]

BPS: Final[float] = 1e-4
_DEFAULT_VOLATILITY: Final[float] = 0.02
"""Fallback daily volatility for the impact term when none is supplied. Chosen
to be unremarkable rather than flattering; a caller with a real estimate should
pass it."""


def impact_cost(
    *,
    price: float,
    quantity: float,
    bar_volume: float,
    volatility: float,
    model: str,
    coefficient: float,
) -> float:
    """Per-share market impact.

    `sqrt` is the empirically supported shape: pushing four times the size
    through a bar costs roughly twice as much per share, not four times. The
    linear model is offered for comparison, and `none` for the frictionless
    diagnostic.

    Args:
        price: Execution price.
        quantity: Signed order size; only its magnitude matters here.
        bar_volume: Volume available in the bar.
        volatility: Per-bar volatility, as a fraction of price.
        model: `none`, `linear`, or `sqrt`.
        coefficient: Impact coefficient.

    Returns:
        Impact in currency per share, never negative.
    """
    size = abs(float(quantity))
    if model == "none" or coefficient <= 0.0 or size == 0.0 or bar_volume <= 0.0:
        return 0.0

    participation = size / bar_volume
    sigma = volatility if np.isfinite(volatility) and volatility > 0 else _DEFAULT_VOLATILITY
    if model == "linear":
        return float(coefficient * price * participation)
    return float(coefficient * price * sigma * np.sqrt(participation))


def round_trip_bps(config: CostsConfig) -> float:
    """Explicit round-trip cost in basis points.

    Commission and half-spread, charged on both sides. Impact is excluded
    because it depends on order size, which a headline number cannot know.

    Args:
        config: The `backtest.costs` section.

    Returns:
        Round-trip cost in bps. This is the figure the break-even cost in
        §11.2 is compared against.
    """
    return 2.0 * (config.commission_bps + config.half_spread_bps)


class CostModel:
    """Charges commission, spread, impact, borrow and financing."""

    def __init__(self, config: CostsConfig) -> None:
        """Bind the model to a cost configuration.

        Args:
            config: The `backtest.costs` section.
        """
        self.config = config

    def __repr__(self) -> str:
        """Return a debug representation naming the headline costs."""
        return (
            f"CostModel(commission={self.config.commission_bps}bps, "
            f"spread={self.config.half_spread_bps}bps, impact={self.config.impact_model})"
        )

    def cap_quantity(self, *, quantity: float, bar_volume: float) -> float:
        """Limit an order to the configured share of the bar's volume.

        Args:
            quantity: Signed order size.
            bar_volume: Volume available in the bar.

        Returns:
            The fillable quantity, same sign as the request. The remainder is
            the caller's problem — the engine records it as a partial fill
            rather than pretending it traded.
        """
        if bar_volume <= 0.0:
            return 0.0
        cap = self.config.max_participation * bar_volume
        return float(np.sign(quantity) * min(abs(quantity), cap))

    def trade_cost(
        self,
        *,
        price: float,
        quantity: float,
        bar_volume: float,
        volatility: float,
    ) -> float:
        """Total cost of one fill, in currency.

        Args:
            price: Execution price.
            quantity: Signed filled quantity.
            bar_volume: Volume available in the bar.
            volatility: Per-bar volatility as a fraction of price.

        Returns:
            The cost, always non-negative. Zero for a zero-size fill, which
            must not attract the minimum commission — charging for a trade that
            did not happen would make a flat strategy lose money.
        """
        size = abs(float(quantity))
        if size == 0.0:
            return 0.0

        per_share = (self.config.commission_bps + self.config.half_spread_bps) * BPS * price
        per_share += impact_cost(
            price=price,
            quantity=quantity,
            bar_volume=bar_volume,
            volatility=volatility,
            model=self.config.impact_model,
            coefficient=self.config.impact_coefficient,
        )
        return float(max(per_share * size, self.config.min_commission))

    def carry_cost(
        self, *, position_value: float, periods_per_year: float, equity: float | None = None
    ) -> float:
        """Borrow and financing accrued over one bar.

        Args:
            position_value: Signed mark-to-market value of the position.
            periods_per_year: Bars per year, for pro-rating the annual rate.
            equity: Account equity, used to detect leverage. None charges
                financing on the whole long notional.

        Returns:
            The carry cost, non-negative.
        """
        if periods_per_year <= 0:
            return 0.0

        if position_value < 0:
            rate = self.config.borrow_bps_annual * BPS
            return float(abs(position_value) * rate / periods_per_year)

        borrowed = position_value if equity is None else max(0.0, position_value - equity)
        rate = self.config.financing_bps_annual * BPS
        return float(borrowed * rate / periods_per_year)
