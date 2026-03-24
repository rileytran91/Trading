"""Visualization charts for backtest results.

Generates a comprehensive multi-panel chart:
1. Price chart with EMA ribbon and regime shading
2. Composite score with confidence bands
3. Layer breakdown
4. Equity curve with drawdown
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# Regime colors
REGIME_COLORS = {
    "Accumulation": "#2ecc71",   # Green
    "Markup": "#3498db",          # Blue
    "Distribution": "#e74c3c",    # Red
    "Markdown": "#e67e22",        # Orange
}

CONFIDENCE_COLORS = {
    "Strong Bullish": "#00c853",
    "Medium Bullish": "#69f0ae",
    "Weak Bullish": "#b9f6ca",
    "Neutral": "#e0e0e0",
    "Weak Bearish": "#ffcdd2",
    "Medium Bearish": "#ef5350",
    "Strong Bearish": "#b71c1c",
}


def plot_full_dashboard(
    master_df: pd.DataFrame,
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    metrics,
    save_path: str | Path = "backtest_results.png",
) -> None:
    """Generate the full backtest dashboard with 4 panels."""
    fig, axes = plt.subplots(4, 1, figsize=(20, 16), height_ratios=[3, 1.5, 1.5, 2])
    fig.suptitle("Capital Flow Backtest Dashboard", fontsize=16, fontweight="bold", y=0.98)

    _plot_price_regime(axes[0], master_df, trade_log)
    _plot_composite_score(axes[1], master_df)
    _plot_layer_breakdown(axes[2], master_df)
    _plot_equity_curve(axes[3], equity_curve, metrics)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dashboard saved to: {save_path}")


def _plot_price_regime(ax: plt.Axes, df: pd.DataFrame, trade_log: pd.DataFrame) -> None:
    """Panel 1: Price with EMA ribbon and regime background shading."""
    ax.set_title("BTC Price with EMA Ribbon & Market Regime", fontsize=12)

    # Regime background shading
    if "regime" in df.columns:
        _shade_regimes(ax, df)

    # EMA ribbon
    ema_cols = [c for c in df.columns if c.startswith("ema_")]
    for col in sorted(ema_cols):
        period = col.replace("ema_", "")
        ax.plot(df.index, df[col], linewidth=0.7, alpha=0.5, label=f"EMA {period}")

    # Price
    ax.plot(df.index, df["close"], color="black", linewidth=1.2, label="BTC Price")

    # Trade markers
    if not trade_log.empty:
        for _, trade in trade_log.iterrows():
            if trade["direction"] == "long":
                ax.annotate("▲", xy=(trade["entry_date"], trade["entry_price"]),
                          fontsize=10, color="green", ha="center", va="bottom")
                if pd.notna(trade.get("exit_date")):
                    ax.annotate("▼", xy=(trade["exit_date"], trade["exit_price"]),
                              fontsize=10, color="red", ha="center", va="top")

    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))


def _plot_composite_score(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 2: Composite score with confidence bands."""
    ax.set_title("Composite Capital Flow Score", fontsize=12)

    if "composite_adjusted" in df.columns:
        score = df["composite_adjusted"]
        ax.fill_between(df.index, 0, score, where=score > 0,
                        color="#2ecc71", alpha=0.4, label="Bullish")
        ax.fill_between(df.index, 0, score, where=score < 0,
                        color="#e74c3c", alpha=0.4, label="Bearish")
        ax.plot(df.index, score, color="black", linewidth=0.8)

    # Threshold lines
    ax.axhline(y=0.6, color="green", linestyle="--", linewidth=0.5, alpha=0.7, label="Strong Bullish")
    ax.axhline(y=0.3, color="green", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axhline(y=-0.3, color="red", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axhline(y=-0.6, color="red", linestyle="--", linewidth=0.5, alpha=0.7, label="Strong Bearish")
    ax.axhline(y=0, color="gray", linewidth=0.5)

    ax.set_ylabel("Score [-1, +1]")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))


def _plot_layer_breakdown(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 3: Individual layer scores."""
    ax.set_title("Layer Score Breakdown", fontsize=12)

    layer_info = [
        ("layer1_score", "L1: Macro Flow", "#e74c3c"),
        ("layer2_score", "L2: Market Structure", "#3498db"),
        ("layer3_score", "L3: Momentum", "#2ecc71"),
        ("layer4_score", "L4: Sentiment", "#9b59b6"),
    ]

    for col, label, color in layer_info:
        if col in df.columns:
            ax.plot(df.index, df[col].rolling(5).mean(), label=label,
                   color=color, linewidth=1.0, alpha=0.8)

    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_ylabel("Layer Score")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))


def _plot_equity_curve(ax: plt.Axes, equity_curve: pd.DataFrame, metrics) -> None:
    """Panel 4: Equity curve with drawdown shading."""
    ax.set_title("Portfolio Equity Curve", fontsize=12)

    if equity_curve.empty:
        ax.text(0.5, 0.5, "No trades executed", transform=ax.transAxes,
               ha="center", va="center", fontsize=14, color="gray")
        return

    equity = equity_curve["equity"]
    ax.plot(equity.index, equity, color="#2c3e50", linewidth=1.2, label="Strategy")

    # Drawdown shading
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    ax2 = ax.twinx()
    ax2.fill_between(equity.index, 0, drawdown * 100, color="#e74c3c", alpha=0.2)
    ax2.set_ylabel("Drawdown (%)", color="#e74c3c")
    ax2.set_ylim(-50, 5)

    # Add metrics text box
    m = metrics.calculate_all()
    text = (
        f"Return: {m['total_return_pct']:.1f}%  |  "
        f"CAGR: {m['cagr_pct']:.1f}%  |  "
        f"Sharpe: {m['sharpe_ratio']:.2f}  |  "
        f"Max DD: {m['max_drawdown_pct']:.1f}%  |  "
        f"Win Rate: {m['win_rate_pct']:.0f}%  |  "
        f"Trades: {m['total_trades']}"
    )
    ax.text(0.5, -0.15, text, transform=ax.transAxes, ha="center",
           fontsize=9, style="italic", color="#34495e",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", alpha=0.8))

    ax.set_ylabel("Equity (USD)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))


def _shade_regimes(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Add background color shading for market regimes."""
    regimes = df["regime"]
    dates = df.index

    current_regime = regimes.iloc[0]
    start_idx = 0

    for i in range(1, len(regimes)):
        if regimes.iloc[i] != current_regime or i == len(regimes) - 1:
            color = REGIME_COLORS.get(current_regime, "#cccccc")
            ax.axvspan(dates[start_idx], dates[i], alpha=0.1, color=color)
            current_regime = regimes.iloc[i]
            start_idx = i

    # Legend for regimes
    patches = [mpatches.Patch(color=c, alpha=0.3, label=r) for r, c in REGIME_COLORS.items()]
    regime_legend = ax.legend(handles=patches, loc="upper right", fontsize=7,
                              title="Regime", title_fontsize=8)
    ax.add_artist(regime_legend)


def plot_signal_detail(
    master_df: pd.DataFrame,
    save_path: str | Path = "signal_detail.png",
) -> None:
    """Plot detailed individual indicator signals."""
    signal_cols = [c for c in master_df.columns if c.startswith("signal_")]
    if not signal_cols:
        return

    n_signals = len(signal_cols)
    fig, axes = plt.subplots(n_signals, 1, figsize=(20, 3 * n_signals), sharex=True)
    if n_signals == 1:
        axes = [axes]

    fig.suptitle("Individual Indicator Signals", fontsize=14, fontweight="bold")

    for ax, col in zip(axes, signal_cols):
        name = col.replace("signal_", "")
        data = master_df[col].rolling(5).mean()

        ax.fill_between(master_df.index, 0, data, where=data > 0,
                        color="#2ecc71", alpha=0.5)
        ax.fill_between(master_df.index, 0, data, where=data < 0,
                        color="#e74c3c", alpha=0.5)
        ax.plot(master_df.index, data, color="black", linewidth=0.5)
        ax.set_ylabel(name, fontsize=8)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=0, color="gray", linewidth=0.3)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = Path(save_path)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Signal detail saved to: {save_path}")
