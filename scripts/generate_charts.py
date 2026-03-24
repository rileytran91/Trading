#!/usr/bin/env python3
"""Generate detailed regime analysis charts."""

from __future__ import annotations

import sys
import os
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from capitalflow.data.historical import build_realistic_master
from capitalflow.indicators.layer1_macro import (
    StablecoinSupplyChange, BTCDominanceChange, TotalMarketCapMomentum, DXYCorrelation,
)
from capitalflow.indicators.layer2_structure import (
    VolumeProfile, OpenInterestProxy, FundingRateProxy, ExchangeReserveEstimate,
)
from capitalflow.indicators.layer3_momentum import (
    MultiTimeframeRSI, MACDHistogramDivergence, EMARibbon, OnBalanceVolumeTrend, ChaikinMoneyFlow,
)
from capitalflow.indicators.layer4_sentiment import (
    FearGreedProxy, VolatilityRegime, MarketBreadth,
)

REGIME_COLORS = {
    "Accumulation": "#2ecc71",
    "Markup": "#3498db",
    "Distribution": "#e74c3c",
    "Markdown": "#e67e22",
    "Unknown": "#cccccc",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def shade_regimes(ax, dates, regimes):
    """Add regime background shading."""
    current = regimes.iloc[0]
    start_idx = 0
    for i in range(1, len(regimes)):
        if regimes.iloc[i] != current or i == len(regimes) - 1:
            color = REGIME_COLORS.get(current, "#cccccc")
            ax.axvspan(dates[start_idx], dates[min(i, len(dates)-1)], alpha=0.15, color=color)
            current = regimes.iloc[i]
            start_idx = i


def generate_master_dashboard(master, output_dir):
    """Main dashboard: price + regimes + all layer signals."""
    print("  Generating master dashboard...")

    # Compute indicators
    indicators_l1 = [StablecoinSupplyChange(), BTCDominanceChange(), TotalMarketCapMomentum(), DXYCorrelation()]
    indicators_l2 = [VolumeProfile(), OpenInterestProxy(), FundingRateProxy(), ExchangeReserveEstimate()]
    indicators_l3 = [MultiTimeframeRSI(), MACDHistogramDivergence(), EMARibbon(), OnBalanceVolumeTrend(), ChaikinMoneyFlow()]
    indicators_l4 = [FearGreedProxy(), VolatilityRegime(), MarketBreadth()]

    for ind in indicators_l1 + indicators_l2 + indicators_l3 + indicators_l4:
        try:
            master = ind.compute(master)
        except Exception:
            pass

    # Compute layer signals
    l1_sigs, l2_sigs, l3_sigs, l4_sigs = [], [], [], []
    for ind in indicators_l1:
        try: l1_sigs.append(ind.signal(master))
        except: pass
    for ind in indicators_l2:
        try: l2_sigs.append(ind.signal(master))
        except: pass
    for ind in indicators_l3:
        try: l3_sigs.append(ind.signal(master))
        except: pass
    for ind in indicators_l4:
        try: l4_sigs.append(ind.signal(master))
        except: pass

    l1_avg = pd.concat(l1_sigs, axis=1).mean(axis=1) if l1_sigs else pd.Series(0, index=master.index)
    l2_avg = pd.concat(l2_sigs, axis=1).mean(axis=1) if l2_sigs else pd.Series(0, index=master.index)
    l3_avg = pd.concat(l3_sigs, axis=1).mean(axis=1) if l3_sigs else pd.Series(0, index=master.index)
    l4_avg = pd.concat(l4_sigs, axis=1).mean(axis=1) if l4_sigs else pd.Series(0, index=master.index)

    # Optimized weights from Phase 1
    composite = (l1_avg * 0.20 + l2_avg * 0.35 + l3_avg * 0.40 + l4_avg * 0.05).rolling(3).mean()

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(5, 1, height_ratios=[3, 1.5, 1.5, 1.5, 1.5], hspace=0.25)

    dates = master.index
    regimes = master["actual_regime"]

    # Panel 1: BTC Price with EMA Ribbon and Regime Shading
    ax1 = fig.add_subplot(gs[0])
    shade_regimes(ax1, dates, regimes)

    ema_cols = sorted([c for c in master.columns if c.startswith("ema_")])
    colors_ema = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(ema_cols)))
    for col, c in zip(ema_cols, colors_ema):
        ax1.plot(dates, master[col], linewidth=0.6, alpha=0.5, color=c)

    ax1.plot(dates, master["close"], color="black", linewidth=1.3, label="BTC Price")
    ax1.set_ylabel("BTC Price (USD)", fontsize=11)
    ax1.set_yscale("log")
    ax1.set_title("BTC Price with Market Regime Shading (2020-2025)", fontsize=14, fontweight="bold")

    patches = [mpatches.Patch(color=c, alpha=0.4, label=r) for r, c in REGIME_COLORS.items() if r != "Unknown"]
    ax1.legend(handles=patches, loc="upper left", fontsize=9, title="Regime")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Composite Score
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    shade_regimes(ax2, dates, regimes)
    ax2.fill_between(dates, 0, composite, where=composite > 0, color="#2ecc71", alpha=0.5)
    ax2.fill_between(dates, 0, composite, where=composite < 0, color="#e74c3c", alpha=0.5)
    ax2.plot(dates, composite, color="black", linewidth=0.8)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.axhline(0.3, color="green", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.axhline(-0.3, color="red", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.set_ylabel("Composite Score", fontsize=10)
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_title("Composite Capital Flow Score (Optimized Weights: L1=20% L2=35% L3=40% L4=5%)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Layer 1 & 2 (Leading)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    shade_regimes(ax3, dates, regimes)
    ax3.plot(dates, l1_avg.rolling(5).mean(), color="#e74c3c", linewidth=1.0, label="L1: Macro Flow (20%)")
    ax3.plot(dates, l2_avg.rolling(5).mean(), color="#3498db", linewidth=1.0, label="L2: Market Structure (35%)")
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.set_ylabel("Signal", fontsize=10)
    ax3.set_ylim(-1.0, 1.0)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.set_title("Leading Indicators: L1 Macro + L2 Structure", fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Layer 3 & 4 (Confirming)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    shade_regimes(ax4, dates, regimes)
    ax4.plot(dates, l3_avg.rolling(5).mean(), color="#2ecc71", linewidth=1.0, label="L3: Momentum (40%)")
    ax4.plot(dates, l4_avg.rolling(5).mean(), color="#9b59b6", linewidth=1.0, label="L4: Sentiment (5%)")
    ax4.axhline(0, color="gray", linewidth=0.5)
    ax4.set_ylabel("Signal", fontsize=10)
    ax4.set_ylim(-1.0, 1.0)
    ax4.legend(loc="upper left", fontsize=9)
    ax4.set_title("Confirming Indicators: L3 Momentum + L4 Sentiment", fontsize=11)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Top 5 leading individual indicators
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    shade_regimes(ax5, dates, regimes)

    top_indicators = [
        (MACDHistogramDivergence(), "MACD Div", "#e74c3c"),
        (TotalMarketCapMomentum(), "Mcap Mom", "#ff6b35"),
        (ExchangeReserveEstimate(), "Exch Flow", "#3498db"),
        (MarketBreadth(), "Breadth", "#9b59b6"),
        (OnBalanceVolumeTrend(), "OBV", "#2ecc71"),
    ]

    for ind, label, color in top_indicators:
        try:
            sig = ind.signal(master).rolling(7).mean()
            ax5.plot(dates, sig, linewidth=0.8, alpha=0.7, color=color, label=label)
        except Exception:
            pass

    ax5.axhline(0, color="gray", linewidth=0.5)
    ax5.set_ylabel("Signal", fontsize=10)
    ax5.set_ylim(-1.0, 1.0)
    ax5.legend(loc="upper left", fontsize=8, ncol=5)
    ax5.set_title("Top 5 Leading Indicators (by transition detection rate)", fontsize=11)
    ax5.grid(True, alpha=0.3)

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    path = os.path.join(output_dir, "regime_analysis_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_indicator_heatmap(master, output_dir):
    """Heatmap of indicator signals by regime."""
    print("  Generating indicator heatmap...")

    all_indicators = [
        ("L1: Stablecoin", StablecoinSupplyChange()),
        ("L1: BTC Dominance", BTCDominanceChange()),
        ("L1: Mcap Momentum", TotalMarketCapMomentum()),
        ("L1: DXY", DXYCorrelation()),
        ("L2: Volume", VolumeProfile()),
        ("L2: OI Proxy", OpenInterestProxy()),
        ("L2: Funding", FundingRateProxy()),
        ("L2: Exchange", ExchangeReserveEstimate()),
        ("L3: RSI", MultiTimeframeRSI()),
        ("L3: MACD Div", MACDHistogramDivergence()),
        ("L3: EMA Ribbon", EMARibbon()),
        ("L3: OBV", OnBalanceVolumeTrend()),
        ("L3: CMF", ChaikinMoneyFlow()),
        ("L4: Fear/Greed", FearGreedProxy()),
        ("L4: Vol Regime", VolatilityRegime()),
        ("L4: Breadth", MarketBreadth()),
    ]

    # Compute all
    for name, ind in all_indicators:
        try:
            master = ind.compute(master)
        except:
            pass

    regimes = master["actual_regime"]
    mask = regimes != "Unknown"

    # Build heatmap data
    regime_order = ["Accumulation", "Markup", "Distribution", "Markdown"]
    data = np.zeros((len(all_indicators), 4))

    for i, (name, ind) in enumerate(all_indicators):
        try:
            sig = ind.signal(master)
            for j, regime in enumerate(regime_order):
                rmask = (regimes == regime) & mask
                if rmask.sum() > 0:
                    data[i, j] = sig[rmask].mean()
        except:
            pass

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.6, vmax=0.6)

    ax.set_xticks(range(4))
    ax.set_xticklabels(regime_order, fontsize=11)
    ax.set_yticks(range(len(all_indicators)))
    ax.set_yticklabels([n for n, _ in all_indicators], fontsize=10)

    # Add text annotations
    for i in range(len(all_indicators)):
        for j in range(4):
            color = "white" if abs(data[i, j]) > 0.3 else "black"
            ax.text(j, i, f"{data[i, j]:+.3f}", ha="center", va="center",
                   fontsize=9, fontweight="bold", color=color)

    ax.set_title("Indicator Signal Strength by Market Regime\n(Green=Bullish, Red=Bearish)",
                fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Mean Signal Strength [-1, +1]")

    plt.tight_layout()
    path = os.path.join(output_dir, "indicator_regime_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_transition_chart(master, output_dir):
    """Chart showing key regime transitions with leading signals."""
    print("  Generating transition analysis chart...")

    regimes = master["actual_regime"]
    transitions = []
    for i in range(1, len(regimes)):
        if regimes.iloc[i] != regimes.iloc[i-1] and regimes.iloc[i] != "Unknown":
            transitions.append({
                "date": regimes.index[i],
                "from": regimes.iloc[i-1],
                "to": regimes.iloc[i],
            })

    # Pick 6 most important transitions
    key_transitions = [t for t in transitions if
        (t["from"] == "Markup" and t["to"] == "Distribution") or
        (t["from"] == "Distribution" and t["to"] == "Markdown") or
        (t["from"] == "Markdown" and t["to"] == "Accumulation") or
        (t["from"] == "Accumulation" and t["to"] == "Markup")
    ][:8]

    if not key_transitions:
        key_transitions = transitions[:8]

    # Compute key indicators
    macd = MACDHistogramDivergence()
    mcap = TotalMarketCapMomentum()
    exch = ExchangeReserveEstimate()
    obv = OnBalanceVolumeTrend()

    for ind in [macd, mcap, exch, obv]:
        try: master = ind.compute(master)
        except: pass

    fig, axes = plt.subplots(len(key_transitions), 1, figsize=(20, 4 * len(key_transitions)))
    if len(key_transitions) == 1:
        axes = [axes]

    fig.suptitle("Key Regime Transitions - Leading Indicator Analysis",
                fontsize=14, fontweight="bold", y=1.01)

    for idx, trans in enumerate(key_transitions):
        ax = axes[idx]
        t_date = trans["date"]
        t_idx = master.index.get_loc(t_date)

        # Window: 40 days before to 20 days after
        start = max(0, t_idx - 40)
        end = min(len(master), t_idx + 20)
        window = master.iloc[start:end]

        # Price (normalized)
        price_norm = window["close"] / window["close"].iloc[0]
        ax.plot(window.index, price_norm, color="black", linewidth=1.5, label="BTC Price")

        # Key indicators
        for ind, label, color in [
            (macd, "MACD", "#e74c3c"),
            (mcap, "Mcap Mom", "#ff6b35"),
            (exch, "Exchange", "#3498db"),
            (obv, "OBV", "#2ecc71"),
        ]:
            try:
                sig = ind.signal(window)
                # Scale to fit on same chart
                ax.plot(window.index, 1 + sig * 0.3, color=color, linewidth=0.8, alpha=0.7, label=label)
            except:
                pass

        ax.axvline(t_date, color="red", linestyle="--", linewidth=2, alpha=0.8)
        ax.set_title(f"{trans['from']} → {trans['to']} ({t_date.strftime('%Y-%m-%d')})",
                    fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, ncol=5)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.tight_layout()
    path = os.path.join(output_dir, "transition_analysis.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("\n" + "=" * 60)
    print("  GENERATING DETAILED ANALYSIS CHARTS")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n  Loading data...")
    master = build_realistic_master("2020-01-01", "2025-03-24")
    print(f"  Data: {len(master)} rows\n")

    generate_master_dashboard(master, OUTPUT_DIR)
    generate_indicator_heatmap(master, OUTPUT_DIR)
    generate_transition_chart(master, OUTPUT_DIR)

    print(f"\n  All charts saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
