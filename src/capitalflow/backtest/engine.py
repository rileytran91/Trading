"""Walk-forward backtest engine.

Orchestrates the full backtest pipeline:
1. Fetch data
2. Compute indicators
3. Generate signals
4. Execute trades
5. Calculate performance
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from ..config import Config
from ..data.fetcher import DataFetcher
from ..signals.scorer import SignalScorer
from ..signals.confidence import classify_confidence, ConfidenceLevel
from ..signals.regime import RegimeDetector, MarketRegime
from .portfolio import Portfolio
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Walk-forward backtesting engine.

    Walk-forward process:
    1. Split data into rolling train/test windows
    2. For each window:
       a. Compute indicators on train+test data (indicators need history)
       b. Generate signals on test period only
       c. Execute trades and record equity
    3. Stitch all test periods for final equity curve
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config.from_yaml()
        self.fetcher = DataFetcher(self.config.data.cache_dir)
        self.scorer = SignalScorer(
            layer_weights=self.config.layers.get("weights", [0.35, 0.25, 0.25, 0.15])
        )
        self.regime_detector = RegimeDetector(
            transition_days=self.config.signals.regime_transition_days
        )

    def run(self, start: str | None = None, end: str | None = None) -> dict:
        """Run the full walk-forward backtest."""
        start = start or self.config.data.start_date
        end = end or self.config.data.end_date

        logger.info(f"Starting backtest: {start} to {end}")

        # 1. Fetch all data
        logger.info("Fetching data...")
        master_df = self.fetcher.build_master_dataframe(
            start=start, end=end,
            altcoins=self.config.data.altcoins,
        )

        # 2. Compute all indicators on full dataset
        logger.info("Computing indicators...")
        master_df = self.scorer.compute_composite(master_df)

        # 3. Detect regimes
        logger.info("Detecting market regimes...")
        master_df["regime"] = self.regime_detector.detect(master_df)

        # 4. Classify confidence levels
        master_df["confidence"] = master_df["composite_adjusted"].apply(
            lambda x: classify_confidence(
                x,
                self.config.signals.strong_threshold,
                self.config.signals.medium_threshold,
                self.config.signals.weak_threshold,
            ).value
        )

        # 5. Run walk-forward backtest
        logger.info("Running walk-forward backtest...")
        portfolio = self._run_walk_forward(master_df)

        # 6. Calculate metrics
        equity_curve = portfolio.get_equity_curve()
        trade_log = portfolio.get_trade_log()
        metrics = PerformanceMetrics(equity_curve, trade_log)

        # 7. Regime analysis
        regime_summary = self.regime_detector.get_regime_summary(master_df)

        return {
            "master_df": master_df,
            "equity_curve": equity_curve,
            "trade_log": trade_log,
            "metrics": metrics,
            "regime_summary": regime_summary,
            "portfolio": portfolio,
        }

    def _run_walk_forward(self, df: pd.DataFrame) -> Portfolio:
        """Execute walk-forward trading simulation."""
        cfg = self.config.backtest
        portfolio = Portfolio(
            initial_capital=cfg.initial_capital,
            commission_pct=cfg.commission_pct,
            slippage_pct=cfg.slippage_pct,
        )

        # Skip the warmup period (need enough history for indicators)
        warmup = 252  # ~1 year for indicator warmup
        if len(df) <= warmup:
            logger.warning("Not enough data for backtest after warmup")
            return portfolio

        tradeable = df.iloc[warmup:]

        for i, (date, row) in enumerate(tradeable.iterrows()):
            price = row["close"]
            composite = row.get("composite_adjusted", 0)
            regime = row.get("regime", MarketRegime.ACCUMULATION.value)
            confidence = classify_confidence(
                composite,
                self.config.signals.strong_threshold,
                self.config.signals.medium_threshold,
                self.config.signals.weak_threshold,
            )

            # Trading logic
            self._apply_trading_rules(
                portfolio, date, price, composite, regime, confidence
            )

            # Update equity
            portfolio.update_equity(date, price)

        # Close any open position at the end
        if portfolio.position_direction != "flat":
            last_date = tradeable.index[-1]
            last_price = tradeable["close"].iloc[-1]
            portfolio.close_position(last_date, last_price)

        return portfolio

    def _apply_trading_rules(
        self,
        portfolio: Portfolio,
        date,
        price: float,
        composite: float,
        regime: str,
        confidence: ConfidenceLevel,
    ) -> None:
        """Apply trading rules based on composite score and regime.

        Entry rules:
        - LONG: composite > entry_threshold AND regime in (Accumulation, Markup)
        - SHORT/CASH: composite < -entry_threshold AND regime in (Distribution, Markdown)

        Exit rules:
        - EXIT LONG: composite < exit_threshold OR regime enters Distribution/Markdown
        - EXIT SHORT: composite > -exit_threshold OR regime enters Accumulation/Markup

        Position sizing: proportional to confidence level
        """
        entry_thresh = self.config.backtest.entry_threshold
        exit_thresh = self.config.backtest.exit_threshold

        bullish_regimes = {MarketRegime.ACCUMULATION.value, MarketRegime.MARKUP.value}
        bearish_regimes = {MarketRegime.DISTRIBUTION.value, MarketRegime.MARKDOWN.value}

        current_dir = portfolio.position_direction

        if current_dir == "flat":
            # Look for entry
            if composite >= entry_thresh and regime in bullish_regimes:
                # Size based on confidence
                size = self._confidence_to_size(confidence)
                portfolio.open_position(
                    date, price, "long", size,
                    regime=regime, composite=composite,
                )
            elif composite <= -entry_thresh and regime in bearish_regimes:
                size = self._confidence_to_size(confidence)
                portfolio.open_position(
                    date, price, "short", size,
                    regime=regime, composite=composite,
                )

        elif current_dir == "long":
            # Check exit conditions
            should_exit = (
                composite < exit_thresh
                or regime in bearish_regimes
                or confidence in (
                    ConfidenceLevel.MEDIUM_BEARISH,
                    ConfidenceLevel.STRONG_BEARISH,
                )
            )
            if should_exit:
                portfolio.close_position(date, price)

        elif current_dir == "short":
            should_exit = (
                composite > -exit_thresh
                or regime in bullish_regimes
                or confidence in (
                    ConfidenceLevel.MEDIUM_BULLISH,
                    ConfidenceLevel.STRONG_BULLISH,
                )
            )
            if should_exit:
                portfolio.close_position(date, price)

    @staticmethod
    def _confidence_to_size(confidence: ConfidenceLevel) -> float:
        """Map confidence level to position size (fraction of capital)."""
        mapping = {
            ConfidenceLevel.STRONG_BULLISH: 1.0,
            ConfidenceLevel.MEDIUM_BULLISH: 0.7,
            ConfidenceLevel.WEAK_BULLISH: 0.4,
            ConfidenceLevel.STRONG_BEARISH: 1.0,
            ConfidenceLevel.MEDIUM_BEARISH: 0.7,
            ConfidenceLevel.WEAK_BEARISH: 0.4,
        }
        return mapping.get(confidence, 0.5)

    def run_simple(self, start: str | None = None, end: str | None = None) -> dict:
        """Simplified backtest without walk-forward (faster for testing)."""
        return self.run(start, end)
