"""Performance metrics for backtest evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """Calculate comprehensive backtest performance metrics."""

    def __init__(self, equity_curve: pd.DataFrame, trade_log: pd.DataFrame):
        self.equity = equity_curve
        self.trades = trade_log
        self.returns = self.equity["equity"].pct_change().dropna()

    def calculate_all(self) -> dict:
        """Calculate all performance metrics."""
        return {
            "total_return_pct": self.total_return(),
            "cagr_pct": self.cagr(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "max_drawdown_pct": self.max_drawdown(),
            "max_drawdown_duration_days": self.max_drawdown_duration(),
            "win_rate_pct": self.win_rate(),
            "profit_factor": self.profit_factor(),
            "avg_win_pct": self.avg_win(),
            "avg_loss_pct": self.avg_loss(),
            "avg_win_loss_ratio": self.win_loss_ratio(),
            "total_trades": len(self.trades),
            "winning_trades": self.winning_trades(),
            "losing_trades": self.losing_trades(),
            "avg_trade_duration_days": self.avg_trade_duration(),
            "volatility_annual_pct": self.annual_volatility(),
            "best_trade_pct": self.best_trade(),
            "worst_trade_pct": self.worst_trade(),
            "buy_and_hold_return_pct": self.buy_and_hold_return(),
        }

    def total_return(self) -> float:
        if self.equity.empty:
            return 0.0
        initial = self.equity["equity"].iloc[0]
        final = self.equity["equity"].iloc[-1]
        return round((final / initial - 1) * 100, 2)

    def cagr(self) -> float:
        if self.equity.empty or len(self.equity) < 2:
            return 0.0
        initial = self.equity["equity"].iloc[0]
        final = self.equity["equity"].iloc[-1]
        days = (self.equity.index[-1] - self.equity.index[0]).days
        if days <= 0 or initial <= 0:
            return 0.0
        years = days / 365.25
        return round(((final / initial) ** (1 / years) - 1) * 100, 2)

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        if self.returns.empty or self.returns.std() == 0:
            return 0.0
        excess = self.returns - risk_free_rate / 252
        return round(excess.mean() / excess.std() * np.sqrt(252), 2)

    def sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        if self.returns.empty:
            return 0.0
        excess = self.returns - risk_free_rate / 252
        downside = excess[excess < 0]
        if downside.empty or downside.std() == 0:
            return 0.0
        return round(excess.mean() / downside.std() * np.sqrt(252), 2)

    def max_drawdown(self) -> float:
        if self.equity.empty:
            return 0.0
        equity = self.equity["equity"]
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return round(drawdown.min() * 100, 2)

    def max_drawdown_duration(self) -> int:
        if self.equity.empty:
            return 0
        equity = self.equity["equity"]
        running_max = equity.cummax()
        underwater = equity < running_max

        if not underwater.any():
            return 0

        # Find longest underwater period
        groups = (~underwater).cumsum()
        underwater_periods = underwater.groupby(groups).sum()
        return int(underwater_periods.max())

    def win_rate(self) -> float:
        if self.trades.empty:
            return 0.0
        wins = (self.trades["pnl"] > 0).sum()
        return round(wins / len(self.trades) * 100, 1)

    def profit_factor(self) -> float:
        if self.trades.empty:
            return 0.0
        gross_profit = self.trades["pnl"][self.trades["pnl"] > 0].sum()
        gross_loss = abs(self.trades["pnl"][self.trades["pnl"] < 0].sum())
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return round(gross_profit / gross_loss, 2)

    def winning_trades(self) -> int:
        if self.trades.empty:
            return 0
        return int((self.trades["pnl"] > 0).sum())

    def losing_trades(self) -> int:
        if self.trades.empty:
            return 0
        return int((self.trades["pnl"] < 0).sum())

    def avg_win(self) -> float:
        if self.trades.empty:
            return 0.0
        wins = self.trades["pnl_pct"][self.trades["pnl"] > 0]
        return round(wins.mean() * 100, 2) if not wins.empty else 0.0

    def avg_loss(self) -> float:
        if self.trades.empty:
            return 0.0
        losses = self.trades["pnl_pct"][self.trades["pnl"] < 0]
        return round(losses.mean() * 100, 2) if not losses.empty else 0.0

    def win_loss_ratio(self) -> float:
        avg_w = self.avg_win()
        avg_l = abs(self.avg_loss())
        if avg_l == 0:
            return float("inf") if avg_w > 0 else 0.0
        return round(avg_w / avg_l, 2)

    def avg_trade_duration(self) -> float:
        if self.trades.empty or "duration_days" not in self.trades.columns:
            return 0.0
        return round(self.trades["duration_days"].mean(), 1)

    def annual_volatility(self) -> float:
        if self.returns.empty:
            return 0.0
        return round(self.returns.std() * np.sqrt(252) * 100, 2)

    def best_trade(self) -> float:
        if self.trades.empty:
            return 0.0
        return round(self.trades["pnl_pct"].max() * 100, 2)

    def worst_trade(self) -> float:
        if self.trades.empty:
            return 0.0
        return round(self.trades["pnl_pct"].min() * 100, 2)

    def buy_and_hold_return(self) -> float:
        """Calculate buy-and-hold return for comparison."""
        if self.equity.empty:
            return 0.0
        # This will be set externally by the engine
        return 0.0

    def print_report(self) -> str:
        """Generate a formatted performance report."""
        m = self.calculate_all()
        lines = [
            "=" * 60,
            "       BACKTEST PERFORMANCE REPORT",
            "=" * 60,
            "",
            "--- Returns ---",
            f"  Total Return:           {m['total_return_pct']:>10.2f}%",
            f"  CAGR:                   {m['cagr_pct']:>10.2f}%",
            f"  Buy & Hold Return:      {m['buy_and_hold_return_pct']:>10.2f}%",
            "",
            "--- Risk Metrics ---",
            f"  Sharpe Ratio:           {m['sharpe_ratio']:>10.2f}",
            f"  Sortino Ratio:          {m['sortino_ratio']:>10.2f}",
            f"  Calmar Ratio:           {m['calmar_ratio']:>10.2f}",
            f"  Max Drawdown:           {m['max_drawdown_pct']:>10.2f}%",
            f"  Max DD Duration:        {m['max_drawdown_duration_days']:>10d} days",
            f"  Annual Volatility:      {m['volatility_annual_pct']:>10.2f}%",
            "",
            "--- Trade Statistics ---",
            f"  Total Trades:           {m['total_trades']:>10d}",
            f"  Win Rate:               {m['win_rate_pct']:>10.1f}%",
            f"  Profit Factor:          {m['profit_factor']:>10.2f}",
            f"  Avg Win:                {m['avg_win_pct']:>10.2f}%",
            f"  Avg Loss:               {m['avg_loss_pct']:>10.2f}%",
            f"  Win/Loss Ratio:         {m['avg_win_loss_ratio']:>10.2f}",
            f"  Best Trade:             {m['best_trade_pct']:>10.2f}%",
            f"  Worst Trade:            {m['worst_trade_pct']:>10.2f}%",
            f"  Avg Duration:           {m['avg_trade_duration_days']:>10.1f} days",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def calmar_ratio(self) -> float:
        cagr = self.cagr()
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return 0.0
        return round(cagr / mdd, 2)
