"""Portfolio tracking and trade management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Record of a single trade."""
    entry_date: datetime
    entry_price: float
    direction: str  # "long" or "short"
    size: float
    exit_date: datetime | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    regime_at_entry: str = ""
    composite_at_entry: float = 0.0


@dataclass
class Portfolio:
    """Tracks positions, equity curve, and trade history."""

    initial_capital: float = 100_000
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005

    # State
    cash: float = field(init=False)
    position: float = field(init=False, default=0.0)
    position_direction: str = field(init=False, default="flat")
    entry_price: float = field(init=False, default=0.0)
    entry_date: datetime | None = field(init=False, default=None)

    # History
    trades: list[Trade] = field(init=False, default_factory=list)
    equity_history: list[dict] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.cash = self.initial_capital

    @property
    def current_equity(self) -> float:
        return self.cash + self.position

    def open_position(
        self,
        date: datetime,
        price: float,
        direction: str = "long",
        size_pct: float = 1.0,
        regime: str = "",
        composite: float = 0.0,
    ) -> None:
        """Open a new position."""
        if self.position_direction != "flat":
            return  # Already in a position

        # Apply slippage
        if direction == "long":
            fill_price = price * (1 + self.slippage_pct)
        else:
            fill_price = price * (1 - self.slippage_pct)

        # Calculate position size
        trade_value = self.cash * size_pct
        commission = trade_value * self.commission_pct
        net_value = trade_value - commission

        self.position = net_value
        self.cash -= trade_value
        self.position_direction = direction
        self.entry_price = fill_price
        self.entry_date = date

        self.trades.append(Trade(
            entry_date=date,
            entry_price=fill_price,
            direction=direction,
            size=net_value,
            regime_at_entry=regime,
            composite_at_entry=composite,
        ))

    def close_position(self, date: datetime, price: float) -> float:
        """Close current position and return PnL."""
        if self.position_direction == "flat":
            return 0.0

        # Apply slippage
        if self.position_direction == "long":
            fill_price = price * (1 - self.slippage_pct)
            pnl_pct = (fill_price - self.entry_price) / self.entry_price
        else:
            fill_price = price * (1 + self.slippage_pct)
            pnl_pct = (self.entry_price - fill_price) / self.entry_price

        # Calculate PnL
        gross_pnl = self.position * pnl_pct
        commission = abs(self.position * (1 + pnl_pct)) * self.commission_pct
        net_pnl = gross_pnl - commission

        self.cash += self.position + net_pnl
        self.position = 0.0
        self.position_direction = "flat"

        # Update trade record
        if self.trades:
            trade = self.trades[-1]
            trade.exit_date = date
            trade.exit_price = fill_price
            trade.pnl = net_pnl
            trade.pnl_pct = pnl_pct

        self.entry_price = 0.0
        self.entry_date = None

        return net_pnl

    def update_equity(self, date: datetime, current_price: float) -> None:
        """Record current equity value."""
        if self.position_direction == "long":
            unrealized = self.position * (
                (current_price - self.entry_price) / self.entry_price
            )
        elif self.position_direction == "short":
            unrealized = self.position * (
                (self.entry_price - current_price) / self.entry_price
            )
        else:
            unrealized = 0.0

        equity = self.cash + self.position + unrealized
        self.equity_history.append({
            "date": date,
            "equity": equity,
            "cash": self.cash,
            "position_value": self.position + unrealized,
            "direction": self.position_direction,
        })

    def get_equity_curve(self) -> pd.DataFrame:
        """Return equity curve as DataFrame."""
        if not self.equity_history:
            return pd.DataFrame()
        df = pd.DataFrame(self.equity_history)
        df = df.set_index("date")
        return df

    def get_trade_log(self) -> pd.DataFrame:
        """Return trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        records = []
        for t in self.trades:
            if t.exit_date is not None:
                records.append({
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "regime": t.regime_at_entry,
                    "composite": t.composite_at_entry,
                    "duration_days": (t.exit_date - t.entry_date).days,
                })
        return pd.DataFrame(records)
