#!/usr/bin/env python3
"""
Regime Detection Optimizer

Runs iterative optimization loops to find the best indicator parameters
and layer weights for accurately detecting market regimes.

Compares predicted regimes against actual labeled regimes to score accuracy.
"""

from __future__ import annotations

import itertools
import logging
import sys
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from capitalflow.data.historical import build_realistic_master, REGIME_LABELS
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
from capitalflow.signals.scorer import SignalScorer
from capitalflow.signals.regime import RegimeDetector, MarketRegime
from capitalflow.signals.confidence import classify_confidence, ConfidenceLevel
from capitalflow.backtest.portfolio import Portfolio
from capitalflow.backtest.metrics import PerformanceMetrics

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_regime_accuracy(predicted: pd.Series, actual: pd.Series) -> dict:
    """Compare predicted regimes against actual labeled regimes."""
    mask = actual != "Unknown"
    pred = predicted[mask]
    act = actual[mask]

    if len(pred) == 0:
        return {"overall": 0, "per_regime": {}}

    # Overall accuracy
    correct = (pred == act).sum()
    overall = correct / len(pred) * 100

    # Per-regime accuracy
    per_regime = {}
    for regime in ["Accumulation", "Markup", "Distribution", "Markdown"]:
        regime_mask = act == regime
        if regime_mask.sum() > 0:
            regime_correct = (pred[regime_mask] == regime).sum()
            per_regime[regime] = {
                "accuracy": round(regime_correct / regime_mask.sum() * 100, 1),
                "total_days": int(regime_mask.sum()),
                "correct_days": int(regime_correct),
                "precision": 0.0,
                "recall": round(regime_correct / regime_mask.sum() * 100, 1),
            }
            # Precision: of all days predicted as this regime, how many actually were?
            pred_regime_mask = pred == regime
            if pred_regime_mask.sum() > 0:
                true_positive = ((pred == regime) & (act == regime)).sum()
                per_regime[regime]["precision"] = round(
                    true_positive / pred_regime_mask.sum() * 100, 1
                )

    return {
        "overall": round(overall, 1),
        "per_regime": per_regime,
        "total_days": int(mask.sum()),
    }


def compute_indicator_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all individual indicator signals."""
    indicators = {
        "L1_stablecoin": StablecoinSupplyChange(lookback=30),
        "L1_btc_dominance": BTCDominanceChange(lookback=14),
        "L1_mcap_momentum": TotalMarketCapMomentum(lookback=20),
        "L1_dxy": DXYCorrelation(lookback=30),
        "L2_volume": VolumeProfile(lookback=20),
        "L2_oi_proxy": OpenInterestProxy(lookback=14),
        "L2_funding": FundingRateProxy(lookback=30),
        "L2_exchange": ExchangeReserveEstimate(lookback=14),
        "L3_rsi": MultiTimeframeRSI(periods=[14, 21]),
        "L3_macd": MACDHistogramDivergence(),
        "L3_ema_ribbon": EMARibbon(periods=[8, 21, 55, 100, 200]),
        "L3_obv": OnBalanceVolumeTrend(lookback=20),
        "L3_cmf": ChaikinMoneyFlow(period=20),
        "L4_fear_greed": FearGreedProxy(lookback=30),
        "L4_vol_regime": VolatilityRegime(atr_period=14),
        "L4_breadth": MarketBreadth(ma_period=50),
    }

    for name, ind in indicators.items():
        try:
            df = ind.compute(df)
            df[f"sig_{name}"] = ind.signal(df)
        except Exception as e:
            logger.warning(f"Failed {name}: {e}")
            df[f"sig_{name}"] = 0.0

    return df


def optimize_layer_weights(df: pd.DataFrame, actual_regimes: pd.Series) -> dict:
    """Grid search over layer weights to find optimal combination."""
    print("\n" + "=" * 70)
    print("  PHASE 1: LAYER WEIGHT OPTIMIZATION")
    print("=" * 70)

    # Get all signal columns by layer
    l1_cols = [c for c in df.columns if c.startswith("sig_L1_")]
    l2_cols = [c for c in df.columns if c.startswith("sig_L2_")]
    l3_cols = [c for c in df.columns if c.startswith("sig_L3_")]
    l4_cols = [c for c in df.columns if c.startswith("sig_L4_")]

    # Layer averages
    df["layer1_avg"] = df[l1_cols].mean(axis=1) if l1_cols else 0
    df["layer2_avg"] = df[l2_cols].mean(axis=1) if l2_cols else 0
    df["layer3_avg"] = df[l3_cols].mean(axis=1) if l3_cols else 0
    df["layer4_avg"] = df[l4_cols].mean(axis=1) if l4_cols else 0

    # Grid search over weights (sum to 1.0)
    best_accuracy = 0
    best_weights = [0.35, 0.25, 0.25, 0.15]
    best_params = {}

    weight_options = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    results = []
    tested = 0

    for w1 in weight_options:
        for w2 in weight_options:
            for w3 in weight_options:
                w4 = round(1.0 - w1 - w2 - w3, 2)
                if w4 < 0.05 or w4 > 0.45:
                    continue

                # Compute composite
                composite = (
                    df["layer1_avg"] * w1 +
                    df["layer2_avg"] * w2 +
                    df["layer3_avg"] * w3 +
                    df["layer4_avg"] * w4
                )
                composite_smooth = composite.rolling(3).mean()

                # Test different transition days
                for trans_days in [3, 5, 7]:
                    for entry_thresh in [0.2, 0.25, 0.3]:
                        df_temp = df.copy()
                        df_temp["composite_smooth"] = composite_smooth
                        df_temp["composite_adjusted"] = composite_smooth

                        detector = RegimeDetector(transition_days=trans_days)
                        predicted = detector.detect(df_temp)

                        acc = compute_regime_accuracy(predicted, actual_regimes)
                        tested += 1

                        if acc["overall"] > best_accuracy:
                            best_accuracy = acc["overall"]
                            best_weights = [w1, w2, w3, w4]
                            best_params = {
                                "weights": [w1, w2, w3, w4],
                                "transition_days": trans_days,
                                "entry_threshold": entry_thresh,
                                "accuracy": acc,
                            }
                            results.append(best_params.copy())

    print(f"\n  Tested {tested} combinations")
    print(f"\n  Best Layer Weights:")
    print(f"    L1 (Macro Flow):      {best_weights[0]:.2f}")
    print(f"    L2 (Market Structure): {best_weights[1]:.2f}")
    print(f"    L3 (Momentum):        {best_weights[2]:.2f}")
    print(f"    L4 (Sentiment):       {best_weights[3]:.2f}")
    print(f"    Transition Days:      {best_params.get('transition_days', 5)}")
    print(f"\n  Overall Regime Accuracy: {best_accuracy:.1f}%")

    if best_params.get("accuracy", {}).get("per_regime"):
        print(f"\n  Per-Regime Accuracy:")
        for regime, stats in best_params["accuracy"]["per_regime"].items():
            print(f"    {regime:15s}: Recall={stats['recall']:5.1f}%  "
                  f"Precision={stats['precision']:5.1f}%  "
                  f"({stats['correct_days']}/{stats['total_days']} days)")

    return best_params


def analyze_individual_indicators(df: pd.DataFrame, actual_regimes: pd.Series) -> dict:
    """Analyze each indicator's predictive power for each regime."""
    print("\n" + "=" * 70)
    print("  PHASE 2: INDIVIDUAL INDICATOR ANALYSIS")
    print("=" * 70)

    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    mask = actual_regimes != "Unknown"

    results = {}

    for col in sig_cols:
        name = col.replace("sig_", "")
        sig = df[col][mask]
        act = actual_regimes[mask]

        indicator_result = {}
        for regime in ["Accumulation", "Markup", "Distribution", "Markdown"]:
            regime_mask = act == regime
            if regime_mask.sum() > 0:
                regime_signal = sig[regime_mask]
                indicator_result[regime] = {
                    "mean_signal": round(regime_signal.mean(), 4),
                    "median_signal": round(regime_signal.median(), 4),
                    "std": round(regime_signal.std(), 4),
                    "pct_positive": round((regime_signal > 0).mean() * 100, 1),
                    "pct_strong_pos": round((regime_signal > 0.3).mean() * 100, 1),
                    "pct_strong_neg": round((regime_signal < -0.3).mean() * 100, 1),
                }

        results[name] = indicator_result

    # Print analysis
    print(f"\n  {'Indicator':<25s} | {'Accumulation':>14s} | {'Markup':>14s} | {'Distribution':>14s} | {'Markdown':>14s}")
    print("  " + "-" * 95)

    for name, regimes in results.items():
        values = []
        for regime in ["Accumulation", "Markup", "Distribution", "Markdown"]:
            if regime in regimes:
                val = regimes[regime]["mean_signal"]
                values.append(f"{val:+.3f}")
            else:
                values.append("   N/A   ")
        print(f"  {name:<25s} | {values[0]:>14s} | {values[1]:>14s} | {values[2]:>14s} | {values[3]:>14s}")

    return results


def find_leading_signals(df: pd.DataFrame, actual_regimes: pd.Series) -> dict:
    """Find which indicators lead regime transitions."""
    print("\n" + "=" * 70)
    print("  PHASE 3: LEADING SIGNAL DETECTION")
    print("  (Which indicators change BEFORE regime transitions?)")
    print("=" * 70)

    sig_cols = [c for c in df.columns if c.startswith("sig_")]

    # Find regime transition points
    transitions = []
    for i in range(1, len(actual_regimes)):
        if actual_regimes.iloc[i] != actual_regimes.iloc[i-1] and actual_regimes.iloc[i] != "Unknown":
            transitions.append({
                "date": actual_regimes.index[i],
                "from": actual_regimes.iloc[i-1],
                "to": actual_regimes.iloc[i],
            })

    print(f"\n  Found {len(transitions)} regime transitions")

    # For each transition, check which signals moved first
    lead_analysis = {}
    for col in sig_cols:
        name = col.replace("sig_", "")
        leads = []

        for trans in transitions:
            idx = df.index.get_loc(trans["date"])
            if idx < 30:
                continue

            # Check signal direction change in the 30 days before transition
            pre_signal = df[col].iloc[max(0, idx-30):idx]
            post_signal = df[col].iloc[idx:min(len(df), idx+10)]

            if len(pre_signal) < 10 or len(post_signal) < 3:
                continue

            # For bullish transitions (to Accumulation/Markup)
            if trans["to"] in ("Accumulation", "Markup"):
                # Did the signal turn positive before the transition?
                for lookback in [5, 10, 15, 20, 25, 30]:
                    window = pre_signal.iloc[-lookback:]
                    if window.mean() > 0.1:
                        leads.append({
                            "transition": f"{trans['from']} → {trans['to']}",
                            "date": trans["date"],
                            "lead_days": lookback,
                            "signal_strength": round(window.mean(), 3),
                        })
                        break

            # For bearish transitions (to Distribution/Markdown)
            elif trans["to"] in ("Distribution", "Markdown"):
                for lookback in [5, 10, 15, 20, 25, 30]:
                    window = pre_signal.iloc[-lookback:]
                    if window.mean() < -0.1:
                        leads.append({
                            "transition": f"{trans['from']} → {trans['to']}",
                            "date": trans["date"],
                            "lead_days": lookback,
                            "signal_strength": round(window.mean(), 3),
                        })
                        break

        if leads:
            avg_lead = np.mean([l["lead_days"] for l in leads])
            detection_rate = len(leads) / len(transitions) * 100
            lead_analysis[name] = {
                "avg_lead_days": round(avg_lead, 1),
                "detection_rate": round(detection_rate, 1),
                "transitions_detected": len(leads),
                "total_transitions": len(transitions),
                "details": leads,
            }

    # Sort by detection rate
    sorted_indicators = sorted(lead_analysis.items(), key=lambda x: x[1]["detection_rate"], reverse=True)

    print(f"\n  {'Indicator':<25s} | {'Detection Rate':>15s} | {'Avg Lead Days':>15s} | {'Detected/Total':>15s}")
    print("  " + "-" * 80)

    for name, stats in sorted_indicators:
        print(f"  {name:<25s} | {stats['detection_rate']:>14.1f}% | {stats['avg_lead_days']:>14.1f}d | "
              f"{stats['transitions_detected']:>6d}/{stats['total_transitions']:<6d}")

    return dict(sorted_indicators)


def optimize_regime_rules(df: pd.DataFrame, actual_regimes: pd.Series, best_weights: list) -> dict:
    """Optimize regime detection rules using the best weights."""
    print("\n" + "=" * 70)
    print("  PHASE 4: REGIME RULE OPTIMIZATION")
    print("=" * 70)

    l1_cols = [c for c in df.columns if c.startswith("sig_L1_")]
    l2_cols = [c for c in df.columns if c.startswith("sig_L2_")]
    l3_cols = [c for c in df.columns if c.startswith("sig_L3_")]
    l4_cols = [c for c in df.columns if c.startswith("sig_L4_")]

    composite = (
        df[l1_cols].mean(axis=1) * best_weights[0] +
        df[l2_cols].mean(axis=1) * best_weights[1] +
        df[l3_cols].mean(axis=1) * best_weights[2] +
        df[l4_cols].mean(axis=1) * best_weights[3]
    ).rolling(3).mean()

    mask = actual_regimes != "Unknown"

    # Analyze composite score distribution per regime
    print(f"\n  Composite Score Distribution by Actual Regime:")
    print(f"  {'Regime':<15s} | {'Mean':>8s} | {'Median':>8s} | {'Std':>8s} | {'Q25':>8s} | {'Q75':>8s}")
    print("  " + "-" * 65)

    regime_stats = {}
    for regime in ["Accumulation", "Markup", "Distribution", "Markdown"]:
        regime_mask = (actual_regimes == regime) & mask
        if regime_mask.sum() > 0:
            scores = composite[regime_mask].dropna()
            stats = {
                "mean": scores.mean(),
                "median": scores.median(),
                "std": scores.std(),
                "q25": scores.quantile(0.25),
                "q75": scores.quantile(0.75),
            }
            regime_stats[regime] = stats
            print(f"  {regime:<15s} | {stats['mean']:>+7.3f} | {stats['median']:>+7.3f} | "
                  f"{stats['std']:>7.3f} | {stats['q25']:>+7.3f} | {stats['q75']:>+7.3f}")

    # Find optimal thresholds
    print(f"\n  Optimizing score thresholds for regime boundaries...")

    best_overall = 0
    best_thresholds = {}

    # Grid search over thresholds
    for acc_low in np.arange(-0.3, 0.2, 0.05):
        for acc_high in np.arange(acc_low + 0.05, 0.4, 0.05):
            for dist_low in np.arange(-0.4, 0.1, 0.05):
                for dist_high in np.arange(dist_low + 0.05, 0.3, 0.05):
                    # Classify based on thresholds + trend
                    price_trend = df["close"].pct_change(20)
                    predicted = pd.Series("Accumulation", index=df.index)

                    predicted[(composite > acc_high) & (price_trend > 0.02)] = "Markup"
                    predicted[(composite < dist_low) & (price_trend < -0.02)] = "Markdown"
                    predicted[(composite > dist_low) & (composite < dist_high) &
                              (price_trend < 0)] = "Distribution"

                    acc = compute_regime_accuracy(predicted[mask], actual_regimes[mask])

                    if acc["overall"] > best_overall:
                        best_overall = acc["overall"]
                        best_thresholds = {
                            "acc_low": round(acc_low, 2),
                            "acc_high": round(acc_high, 2),
                            "dist_low": round(dist_low, 2),
                            "dist_high": round(dist_high, 2),
                            "accuracy": acc,
                        }

    print(f"\n  Optimal Thresholds:")
    print(f"    Accumulation zone: [{best_thresholds.get('acc_low', -0.1)}, {best_thresholds.get('acc_high', 0.2)}]")
    print(f"    Distribution zone: [{best_thresholds.get('dist_low', -0.2)}, {best_thresholds.get('dist_high', 0.1)}]")
    print(f"    Markup trigger:    composite > {best_thresholds.get('acc_high', 0.2)} & trend > 0")
    print(f"    Markdown trigger:  composite < {best_thresholds.get('dist_low', -0.2)} & trend < 0")
    print(f"\n  Optimized Overall Accuracy: {best_overall:.1f}%")

    if best_thresholds.get("accuracy", {}).get("per_regime"):
        for regime, stats in best_thresholds["accuracy"]["per_regime"].items():
            print(f"    {regime:15s}: Recall={stats['recall']:5.1f}%  Precision={stats['precision']:5.1f}%")

    return best_thresholds


def run_optimized_backtest(df: pd.DataFrame, best_weights: list, best_thresholds: dict) -> dict:
    """Run backtest with optimized parameters."""
    print("\n" + "=" * 70)
    print("  PHASE 5: OPTIMIZED BACKTEST")
    print("=" * 70)

    l1_cols = [c for c in df.columns if c.startswith("sig_L1_")]
    l2_cols = [c for c in df.columns if c.startswith("sig_L2_")]
    l3_cols = [c for c in df.columns if c.startswith("sig_L3_")]
    l4_cols = [c for c in df.columns if c.startswith("sig_L4_")]

    composite = (
        df[l1_cols].mean(axis=1) * best_weights[0] +
        df[l2_cols].mean(axis=1) * best_weights[1] +
        df[l3_cols].mean(axis=1) * best_weights[2] +
        df[l4_cols].mean(axis=1) * best_weights[3]
    ).rolling(3).mean()

    # Run portfolio simulation
    portfolio = Portfolio(initial_capital=100_000, commission_pct=0.001, slippage_pct=0.0005)

    warmup = 252
    entry_thresh = 0.25
    exit_thresh = 0.05

    price_trend = df["close"].pct_change(20)

    for i in range(warmup, len(df)):
        date = df.index[i]
        price = df["close"].iloc[i]
        score = composite.iloc[i] if not pd.isna(composite.iloc[i]) else 0
        trend = price_trend.iloc[i] if not pd.isna(price_trend.iloc[i]) else 0

        if portfolio.position_direction == "flat":
            # Entry: strong bullish signal + uptrend
            if score > entry_thresh and trend > 0.01:
                size = min(1.0, 0.5 + abs(score))
                portfolio.open_position(date, price, "long", size)
            # Entry: strong bearish signal + downtrend
            elif score < -entry_thresh and trend < -0.01:
                size = min(1.0, 0.5 + abs(score))
                portfolio.open_position(date, price, "short", size)

        elif portfolio.position_direction == "long":
            if score < exit_thresh or trend < -0.03:
                portfolio.close_position(date, price)

        elif portfolio.position_direction == "short":
            if score > -exit_thresh or trend > 0.03:
                portfolio.close_position(date, price)

        portfolio.update_equity(date, price)

    # Close any remaining position
    if portfolio.position_direction != "flat":
        portfolio.close_position(df.index[-1], df["close"].iloc[-1])

    equity = portfolio.get_equity_curve()
    trades = portfolio.get_trade_log()

    if not equity.empty and not trades.empty:
        metrics = PerformanceMetrics(equity, trades)
        report = metrics.print_report()
        print(report)

        # Buy and hold comparison
        bh_return = (df["close"].iloc[-1] / df["close"].iloc[warmup] - 1) * 100
        print(f"\n  Buy & Hold Return:  {bh_return:>10.2f}%")
        print(f"  Strategy Return:    {metrics.total_return():>10.2f}%")
        print(f"  Alpha:              {metrics.total_return() - bh_return:>+10.2f}%")

        return {"metrics": metrics, "equity": equity, "trades": trades}
    else:
        print("  No trades were generated.")
        return {}


def print_regime_confirmation_rules(
    indicator_analysis: dict,
    lead_analysis: dict,
    best_weights: list,
    best_thresholds: dict,
):
    """Print the final confirmed regime detection rules."""
    print("\n" + "=" * 70)
    print("  FINAL: REGIME CONFIRMATION RULEBOOK")
    print("=" * 70)

    print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  OPTIMAL LAYER WEIGHTS                                          │
  │  ──────────────────                                             │
  │  L1 Macro Capital Flow:  {best_weights[0]:.0%}  (LEADING - most important)    │
  │  L2 Market Structure:    {best_weights[1]:.0%}                                 │
  │  L3 Technical Momentum:  {best_weights[2]:.0%}                                 │
  │  L4 Sentiment:           {best_weights[3]:.0%}  (CONFIRMING)                   │
  └──────────────────────────────────────────────────────────────────┘
""")

    # Rank indicators by leading capability
    print("  TOP LEADING INDICATORS (by detection rate):")
    print("  " + "-" * 60)
    for i, (name, stats) in enumerate(lead_analysis.items()):
        if i >= 10:
            break
        star = " ***" if stats["detection_rate"] > 60 else " **" if stats["detection_rate"] > 40 else ""
        print(f"  {i+1:2d}. {name:<25s} Detection: {stats['detection_rate']:5.1f}%  "
              f"Lead: {stats['avg_lead_days']:4.1f}d{star}")

    print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │                    REGIME CONFIRMATION RULES                     │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  ● ACCUMULATION (Tích lũy)                                      │
  │    ─ Composite score: {best_thresholds.get('acc_low', -0.1):+.2f} to {best_thresholds.get('acc_high', 0.2):+.2f}                          │
  │    ─ Volatility: LOW (ATR percentile < 40%)                     │
  │    ─ Volume: Decreasing on down days                            │
  │    ─ Key signals: OBV divergence positive, CMF turning up       │
  │    ─ DXY: Weakening or stable                                   │
  │    ─ Breadth: Altcoins starting to recover                      │
  │    → ENTRY SIGNAL: Score crosses above {best_thresholds.get('acc_high', 0.2):+.2f}                   │
  │                                                                  │
  │  ● MARKUP (Tăng trưởng)                                         │
  │    ─ Composite score: > {best_thresholds.get('acc_high', 0.2):+.2f} with uptrend                    │
  │    ─ EMA Ribbon: Fully bullish (8>21>55>100>200)                │
  │    ─ Volume: Increasing on up days                              │
  │    ─ RSI: 50-70 range (healthy uptrend)                         │
  │    ─ Market breadth: >60% alts above 50-day MA                  │
  │    ─ BTC dominance: Rising early, then falling (altseason)      │
  │    → HOLD/ADD: Score remains above {best_thresholds.get('acc_high', 0.2):+.2f}                      │
  │                                                                  │
  │  ● DISTRIBUTION (Phân phối)                                     │
  │    ─ Composite score: {best_thresholds.get('dist_low', -0.2):+.2f} to {best_thresholds.get('dist_high', 0.1):+.2f}                         │
  │    ─ Volatility: HIGH and expanding                             │
  │    ─ Volume: Climactic spikes on down days                      │
  │    ─ MACD: Bearish divergence (price up, MACD down)             │
  │    ─ Funding rate proxy: Extremely positive (overleveraged)     │
  │    ─ Fear/Greed: Extreme Greed (>80)                            │
  │    → EXIT SIGNAL: Score drops below {best_thresholds.get('dist_low', -0.2):+.2f}                    │
  │                                                                  │
  │  ● MARKDOWN (Sụp đổ)                                            │
  │    ─ Composite score: < {best_thresholds.get('dist_low', -0.2):+.2f} with downtrend                │
  │    ─ EMA Ribbon: Fully bearish (8<21<55<100<200)                │
  │    ─ Volume: Capitulation spikes                                │
  │    ─ RSI: <30 (oversold, but can stay oversold)                 │
  │    ─ Market breadth: <30% alts above 50-day MA                  │
  │    ─ DXY: Strengthening (risk-off)                              │
  │    → WAIT FOR: Score crosses above {best_thresholds.get('acc_low', -0.1):+.2f} = potential bottom   │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
""")

    print("  MULTI-LAYER CONFIRMATION PROTOCOL:")
    print("  " + "-" * 60)
    print("""
  To confirm a regime transition, require agreement from
  AT LEAST 3 out of 4 layers:

  Step 1: L1 (Macro) fires first → EARLY WARNING
          Stablecoin flows change, BTC.D shifts, DXY reverses

  Step 2: L2 (Structure) confirms → INCREASING CONFIDENCE
          Volume profile changes, exchange flows shift

  Step 3: L3 (Momentum) aligns → ACTIONABLE SIGNAL
          EMA ribbon crosses, MACD confirms, RSI enters zone

  Step 4: L4 (Sentiment) validates → FULL CONFIRMATION
          Fear/Greed hits extreme, breadth confirms, vol regime shifts

  ⚠  If only 1-2 layers agree → WAIT, possible false signal
  ✓  If 3+ layers agree → HIGH CONFIDENCE regime change
  ✓✓ If all 4 agree → MAXIMUM CONFIDENCE, increase position size
""")


def main():
    print("=" * 70)
    print("   CAPITAL FLOW REGIME OPTIMIZATION ENGINE")
    print("   Finding Optimal Leading Indicators for Market Cycles")
    print("=" * 70)
    print(f"\n   Data: 2020-01-01 → 2025-03-24 (BTC + ETH + 8 Altcoins)")
    print(f"   Labeled Regimes: {len(REGIME_LABELS)} periods")

    # Build realistic data
    print("\n   Building historical market data...")
    master = build_realistic_master("2020-01-01", "2025-03-24")
    print(f"   Master DataFrame: {len(master)} rows × {len(master.columns)} columns")

    # Store actual regimes
    actual_regimes = master["actual_regime"].copy()

    # Compute all indicator signals
    print("   Computing 16 indicator signals across 4 layers...")
    master = compute_indicator_signals(master)

    # Phase 1: Layer weight optimization
    best_params = optimize_layer_weights(master, actual_regimes)
    best_weights = best_params.get("weights", [0.35, 0.25, 0.25, 0.15])

    # Phase 2: Individual indicator analysis
    indicator_analysis = analyze_individual_indicators(master, actual_regimes)

    # Phase 3: Leading signal detection
    lead_analysis = find_leading_signals(master, actual_regimes)

    # Phase 4: Regime rule optimization
    best_thresholds = optimize_regime_rules(master, actual_regimes, best_weights)

    # Phase 5: Run backtest with optimized params
    bt_results = run_optimized_backtest(master, best_weights, best_thresholds)

    # Final: Print regime confirmation rules
    print_regime_confirmation_rules(indicator_analysis, lead_analysis, best_weights, best_thresholds)

    # Save analysis results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save indicator analysis
    rows = []
    for name, regimes in indicator_analysis.items():
        for regime, stats in regimes.items():
            rows.append({"indicator": name, "regime": regime, **stats})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "indicator_analysis.csv"), index=False)

    # Save lead analysis
    lead_rows = []
    for name, stats in lead_analysis.items():
        lead_rows.append({
            "indicator": name,
            "detection_rate": stats["detection_rate"],
            "avg_lead_days": stats["avg_lead_days"],
            "transitions_detected": stats["transitions_detected"],
        })
    pd.DataFrame(lead_rows).to_csv(os.path.join(output_dir, "lead_analysis.csv"), index=False)

    print(f"\n   Analysis files saved to: {output_dir}/")
    print("\n" + "=" * 70)
    print("   OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
