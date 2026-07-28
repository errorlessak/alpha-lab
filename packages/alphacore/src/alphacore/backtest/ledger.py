"""The immutable event log, and the series derived from it (§12).

Everything the backtest reports is *derived* from this log rather than
accumulated alongside it. Equity, drawdown, and the trade blotter are all
views over the same recorded events, so they cannot disagree with each other —
which is a failure mode that otherwise surfaces as a tearsheet whose equity
curve and trade list imply different P&L.

Fills are paired into round-trip trades by walking the log and closing a trade
whenever the position crosses or returns to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["EquityPoint", "Fill", "Ledger"]


@dataclass(frozen=True)
class Fill:
    """One execution.

    Attributes:
        ts: Bar the fill occurred on.
        quantity: Signed filled quantity.
        price: Fill price.
        cost: Transaction cost charged.
        reason: Why the order existed, e.g. `signal` or `stop_loss`.
        requested: Quantity originally requested, before any participation cap.
        bar: Positional index of the bar. Carried explicitly because holding
            period is measured in *bars*, and a timestamp difference cannot
            recover that on a session calendar with weekends and holidays.
    """

    ts: pd.Timestamp
    quantity: float
    price: float
    cost: float
    reason: str
    requested: float = 0.0
    bar: int | None = None

    @property
    def is_partial(self) -> bool:
        """Whether the participation cap prevented a full fill."""
        return abs(self.requested) > abs(self.quantity) + 1e-12


@dataclass(frozen=True)
class EquityPoint:
    """The account state at the close of one bar."""

    ts: pd.Timestamp
    equity: float
    position: float
    cash: float


@dataclass
class Ledger:
    """An append-only log of fills and per-bar equity."""

    fills: list[Fill] = field(default_factory=list)
    points: list[EquityPoint] = field(default_factory=list)

    def record_fill(
        self,
        *,
        ts: pd.Timestamp,
        quantity: float,
        price: float,
        cost: float,
        reason: str,
        requested: float | None = None,
        bar: int | None = None,
    ) -> None:
        """Append a fill.

        Args:
            ts: Bar the fill occurred on.
            quantity: Signed filled quantity.
            price: Fill price.
            cost: Transaction cost.
            reason: Why the order existed.
            requested: Quantity requested before the participation cap.
            bar: Positional index of the bar, used for holding periods.
        """
        self.fills.append(
            Fill(
                ts=ts,
                quantity=float(quantity),
                price=float(price),
                cost=float(cost),
                reason=reason,
                requested=float(quantity if requested is None else requested),
                bar=bar,
            )
        )

    def record_equity(
        self, *, ts: pd.Timestamp, equity: float, position: float, cash: float
    ) -> None:
        """Append the account state at a bar close.

        Args:
            ts: The bar.
            equity: Mark-to-market equity.
            position: Signed position in shares.
            cash: Uninvested currency.
        """
        self.points.append(
            EquityPoint(ts=ts, equity=float(equity), position=float(position), cash=float(cash))
        )

    @property
    def n_partial_fills(self) -> int:
        """How many fills the participation cap truncated."""
        return sum(1 for fill in self.fills if fill.is_partial)

    def fill_log(self) -> pd.DataFrame:
        """Build the fill-by-fill event log written to `ledger.parquet` (§16).

        `trades()` pairs these into round trips, which is what a reviewer reads;
        this is the record the pairing was derived from, kept so the two can be
        reconciled against each other rather than taken on trust.

        Returns:
            A frame indexed by `ts` with `quantity`, `requested`, `price`,
            `cost`, `reason`, `bar`, and `partial`.
        """
        columns = ["quantity", "requested", "price", "cost", "reason", "bar", "partial"]
        if not self.fills:
            return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="ts"))
        return pd.DataFrame(
            {
                "quantity": [f.quantity for f in self.fills],
                "requested": [f.requested for f in self.fills],
                "price": [f.price for f in self.fills],
                "cost": [f.cost for f in self.fills],
                "reason": [f.reason for f in self.fills],
                "bar": [-1 if f.bar is None else f.bar for f in self.fills],
                "partial": [f.is_partial for f in self.fills],
            },
            index=pd.DatetimeIndex([f.ts for f in self.fills], name="ts"),
        )

    def equity_curve(self) -> pd.DataFrame:
        """Build the per-bar equity series.

        Returns:
            A frame indexed by `ts` with `equity`, `position`, and `cash`.
        """
        if not self.points:
            return pd.DataFrame(
                columns=["equity", "position", "cash"],
                index=pd.DatetimeIndex([], name="ts"),
            )
        return pd.DataFrame(
            {
                "equity": [p.equity for p in self.points],
                "position": [p.position for p in self.points],
                "cash": [p.cash for p in self.points],
            },
            index=pd.DatetimeIndex([p.ts for p in self.points], name="ts"),
        )

    def drawdown(self) -> pd.DataFrame:
        """Build the underwater series.

        Returns:
            A frame indexed by `ts` with `equity`, `peak`, and `drawdown`.
            `drawdown` is fractional and never positive.
        """
        curve = self.equity_curve()
        if curve.empty:
            return pd.DataFrame(
                columns=["equity", "peak", "drawdown"], index=pd.DatetimeIndex([], name="ts")
            )
        peak = curve["equity"].cummax()
        return pd.DataFrame(
            {
                "equity": curve["equity"],
                "peak": peak,
                "drawdown": np.where(peak > 0, curve["equity"] / peak - 1.0, 0.0),
            },
            index=curve.index,
        )

    def trades(self) -> pd.DataFrame:
        """Pair fills into completed round-trip trades.

        A trade opens when the position leaves zero and closes when a fill
        *reduces* it — to zero, partially, or through zero into a flip. A fill
        in the same direction as the position is a scale-in, not a round trip:
        it blends into the open trade's average entry price and accumulates its
        cost. Vol-target sizing rebalances on most bars, so counting those as
        round trips would report several hundred trades for a position that was
        never closed, and every trade-based metric downstream — win rate,
        payoff, average holding period — would describe something that did not
        happen.

        A position still open at the end of the sample is not reported: an
        unrealised gain is not a trade, and counting it as one is how a blotter
        comes to disagree with the equity curve.

        Returns:
            A frame with one row per completed trade.
        """
        columns = [
            "entry_ts",
            "exit_ts",
            "side",
            "quantity",
            "entry_price",
            "exit_price",
            "pnl",
            "return_pct",
            "cost",
            "bars_held",
            "exit_reason",
        ]
        if not self.fills:
            return pd.DataFrame(columns=columns)

        rows: list[dict[str, Any]] = []
        position = 0.0
        entry: dict[str, Any] | None = None

        for fill in self.fills:
            before = position
            position += fill.quantity

            if before == 0.0 and position != 0.0:
                entry = {
                    "entry_ts": fill.ts,
                    "side": int(np.sign(position)),
                    "quantity": abs(position),
                    "entry_price": fill.price,
                    "cost": fill.cost,
                    "_bar": fill.bar,
                }
            elif before != 0.0 and entry is not None:
                if np.sign(fill.quantity) == np.sign(before):
                    entry = _scale_in(entry, before=before, fill=fill)
                    continue

                closed = min(abs(before), abs(fill.quantity))
                side = entry["side"]
                gross = side * closed * (fill.price - entry["entry_price"])
                total_cost = entry["cost"] + fill.cost
                notional = closed * entry["entry_price"]
                opened = {k: v for k, v in entry.items() if not k.startswith("_")}
                rows.append(
                    {
                        **opened,
                        "quantity": closed,
                        "exit_ts": fill.ts,
                        "exit_price": fill.price,
                        "pnl": gross - total_cost,
                        "return_pct": (gross - total_cost) / notional if notional else 0.0,
                        "cost": total_cost,
                        "bars_held": _bars_held(
                            entry.get("_bar"), fill.bar, entry["entry_ts"], fill.ts, self.points
                        ),
                        "exit_reason": fill.reason,
                    }
                )
                entry = _carry(entry, position=position, fill=fill)

        return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def _scale_in(entry: dict[str, Any], *, before: float, fill: Fill) -> dict[str, Any]:
    """Blend a same-direction fill into the open trade.

    Args:
        entry: The open trade.
        before: Position before this fill.
        fill: The scale-in fill.

    Returns:
        The updated entry, carrying a quantity-weighted average entry price.
        The average is what makes the eventual `return_pct` describe the whole
        position rather than only the first slice of it.
    """
    added = abs(fill.quantity)
    held = abs(before)
    total = held + added
    return {
        **entry,
        "entry_price": (held * float(entry["entry_price"]) + added * fill.price) / total,
        "quantity": total,
        "cost": float(entry["cost"]) + fill.cost,
    }


def _carry(entry: dict[str, Any], *, position: float, fill: Fill) -> dict[str, Any] | None:
    """Decide what remains open after a reducing fill.

    Args:
        entry: The trade that was just closed against.
        position: Position after the fill.
        fill: The reducing fill.

    Returns:
        None when the position is flat. When the fill only trimmed the
        position, the original entry price and timestamp are kept — the
        remainder was bought when the trade opened, not at the price it was
        partly sold at. When the fill flipped the position, a fresh trade opens
        at the fill price.
    """
    if position == 0.0:
        return None
    flipped = np.sign(position) != int(entry["side"])
    if flipped:
        return {
            "entry_ts": fill.ts,
            "side": int(np.sign(position)),
            "quantity": abs(position),
            "entry_price": fill.price,
            "cost": 0.0,
            "_bar": fill.bar,
        }
    return {**entry, "quantity": abs(position), "cost": 0.0}


def _bars_held(
    entry_bar: int | None,
    exit_bar: int | None,
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    points: Sequence[EquityPoint],
) -> int:
    """Measure a holding period in bars.

    Prefers the recorded bar indices, falls back to the equity grid, and
    returns 0 only when neither is available. It does **not** fall back to a
    timestamp difference: on a session calendar a three-day gap can be one bar
    or three, and guessing would put a wrong holding period in the blotter.

    Args:
        entry_bar: Bar index of the entry fill, if recorded.
        exit_bar: Bar index of the exit fill, if recorded.
        entry_ts: Entry timestamp.
        exit_ts: Exit timestamp.
        points: Recorded per-bar equity points.

    Returns:
        The holding period in bars.
    """
    if entry_bar is not None and exit_bar is not None:
        return int(max(0, exit_bar - entry_bar))
    if points:
        index = pd.DatetimeIndex([p.ts for p in points])
        return int(max(0, index.searchsorted(exit_ts) - index.searchsorted(entry_ts)))
    return 0
