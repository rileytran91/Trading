#!/usr/bin/env python3
"""
Capital Flow Backtest System - Main Entry Point

Hệ thống backtest dòng tiền đầu tư Crypto.
Phân tích 4 lớp chỉ báo dẫn dắt để xác định chu kỳ thị trường.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --start 2021-01-01 --end 2024-12-31
    python scripts/run_backtest.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from capitalflow.config import Config
from capitalflow.backtest.engine import BacktestEngine
from capitalflow.visualization.plots import plot_full_dashboard, plot_signal_detail
from capitalflow.signals.regime import MarketRegime


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Capital Flow Backtest System for Crypto Market Cycle Detection"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("backtest")

    # Load config
    config = Config.from_yaml(args.config)
    if args.start:
        config.data.start_date = args.start
    if args.end:
        config.data.end_date = args.end

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Print header
    print("=" * 70)
    print("   CAPITAL FLOW BACKTEST SYSTEM")
    print("   Crypto Market Cycle Detection via Leading Indicators")
    print("=" * 70)
    print(f"\n   Period:     {config.data.start_date} → {config.data.end_date}")
    print(f"   Symbol:     {config.data.base_symbol}")
    print(f"   Capital:    ${config.backtest.initial_capital:,.0f}")
    print(f"   Altcoins:   {len(config.data.altcoins)} symbols")
    print()

    # Run backtest
    engine = BacktestEngine(config)

    try:
        results = engine.run(
            start=config.data.start_date,
            end=config.data.end_date,
        )
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

    # Print results
    metrics = results["metrics"]
    report = metrics.print_report()
    print(report)

    # Print regime analysis
    print("\n--- Market Regime Analysis ---")
    regime_summary = results["regime_summary"]
    for regime, stats in regime_summary.items():
        print(f"  {regime:15s}: {stats['days']:4d} days ({stats['pct_of_total']:5.1f}%) "
              f"| Avg daily return: {stats['avg_daily_return']:+.4f}%")

    # Print current signal
    master_df = results["master_df"]
    if not master_df.empty:
        last = master_df.iloc[-1]
        print(f"\n--- Latest Signal ({master_df.index[-1].strftime('%Y-%m-%d')}) ---")
        print(f"  Composite Score:    {last.get('composite_adjusted', 0):+.3f}")
        print(f"  Confidence:         {last.get('confidence', 'N/A')}")
        print(f"  Market Regime:      {last.get('regime', 'N/A')}")
        print(f"  Layer Agreement:    {last.get('layer_agreement', 0):.1%}")
        print()
        print(f"  L1 Macro Flow:      {last.get('layer1_score', 0):+.3f}")
        print(f"  L2 Market Structure: {last.get('layer2_score', 0):+.3f}")
        print(f"  L3 Momentum:        {last.get('layer3_score', 0):+.3f}")
        print(f"  L4 Sentiment:       {last.get('layer4_score', 0):+.3f}")

    # Generate visualizations
    print("\nGenerating charts...")
    try:
        plot_full_dashboard(
            master_df=results["master_df"],
            equity_curve=results["equity_curve"],
            trade_log=results["trade_log"],
            metrics=metrics,
            save_path=os.path.join(args.output, "backtest_dashboard.png"),
        )

        plot_signal_detail(
            master_df=results["master_df"],
            save_path=os.path.join(args.output, "signal_detail.png"),
        )
    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")

    # Save data
    try:
        results["trade_log"].to_csv(
            os.path.join(args.output, "trade_log.csv"), index=False
        )
        results["equity_curve"].to_csv(
            os.path.join(args.output, "equity_curve.csv")
        )
        print(f"\nResults saved to: {args.output}/")
    except Exception as e:
        logger.warning(f"Failed to save CSV: {e}")

    print("\n" + "=" * 70)
    print("   Backtest complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
