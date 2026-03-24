"""Wyckoff Market Cycle Regime Detection.

Identifies 4 market phases:
1. ACCUMULATION - Smart money buying, low volatility, range-bound
2. MARKUP       - Uptrend, expanding breadth, rising volume on up days
3. DISTRIBUTION - Smart money selling, high volatility, range-bound at tops
4. MARKDOWN     - Downtrend, declining breadth, rising volume on down days
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    ACCUMULATION = "Accumulation"
    MARKUP = "Markup"
    DISTRIBUTION = "Distribution"
    MARKDOWN = "Markdown"


class RegimeDetector:
    """Detects Wyckoff market regimes using composite signals.

    Uses hysteresis (requires N consecutive qualifying days)
    to prevent whipsaw between regimes.
    """

    def __init__(self, transition_days: int = 5):
        self.transition_days = transition_days

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Detect market regime for each row in the dataframe.

        Requires columns:
        - composite_smooth: the smoothed composite score
        - atr_percentile: volatility regime (from Layer 4)
        - vol_buy_ratio: volume direction (from Layer 2)
        """
        regimes = pd.Series(MarketRegime.ACCUMULATION.value, index=df.index)

        composite = df.get("composite_smooth", pd.Series(0, index=df.index))
        atr_pct = df.get("atr_percentile", pd.Series(0.5, index=df.index))
        price_trend = df["close"].pct_change(20)

        # Score each regime's probability
        accumulation_score = pd.Series(0.0, index=df.index)
        markup_score = pd.Series(0.0, index=df.index)
        distribution_score = pd.Series(0.0, index=df.index)
        markdown_score = pd.Series(0.0, index=df.index)

        # ACCUMULATION conditions:
        # - Composite turning positive from negative
        # - Low volatility
        # - Price range-bound (low momentum)
        accumulation_score += (composite > -0.2).astype(float) * 0.3
        accumulation_score += (composite < 0.3).astype(float) * 0.2
        accumulation_score += (atr_pct < 0.4).astype(float) * 0.3
        accumulation_score += (price_trend.abs() < 0.1).astype(float) * 0.2

        # MARKUP conditions:
        # - Composite solidly positive
        # - Price uptrend
        # - Moderate to expanding volatility
        markup_score += (composite > 0.2).astype(float) * 0.4
        markup_score += (price_trend > 0.05).astype(float) * 0.3
        markup_score += ((atr_pct > 0.2) & (atr_pct < 0.7)).astype(float) * 0.3

        # DISTRIBUTION conditions:
        # - Composite turning negative from positive
        # - High volatility
        # - Price near highs but momentum fading
        distribution_score += (composite < 0.2).astype(float) * 0.3
        distribution_score += (composite > -0.3).astype(float) * 0.2
        distribution_score += (atr_pct > 0.5).astype(float) * 0.3
        distribution_score += (price_trend.abs() < 0.1).astype(float) * 0.2

        # MARKDOWN conditions:
        # - Composite solidly negative
        # - Price downtrend
        # - High volatility
        markdown_score += (composite < -0.2).astype(float) * 0.4
        markdown_score += (price_trend < -0.05).astype(float) * 0.3
        markdown_score += (atr_pct > 0.4).astype(float) * 0.3

        # Select highest-scoring regime
        scores = pd.DataFrame({
            MarketRegime.ACCUMULATION.value: accumulation_score,
            MarketRegime.MARKUP.value: markup_score,
            MarketRegime.DISTRIBUTION.value: distribution_score,
            MarketRegime.MARKDOWN.value: markdown_score,
        })

        raw_regime = scores.idxmax(axis=1)

        # Apply hysteresis: require N consecutive days in new regime before switching
        smoothed = self._apply_hysteresis(raw_regime)

        return smoothed

    def _apply_hysteresis(self, raw: pd.Series) -> pd.Series:
        """Require transition_days consecutive signals before regime change."""
        result = raw.copy()
        current_regime = raw.iloc[0]
        count = 0

        for i in range(len(raw)):
            if raw.iloc[i] == current_regime:
                count = 0
                result.iloc[i] = current_regime
            else:
                count += 1
                if count >= self.transition_days:
                    current_regime = raw.iloc[i]
                    count = 0
                result.iloc[i] = current_regime

        return result

    def get_regime_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics for each regime period."""
        if "regime" not in df.columns:
            df["regime"] = self.detect(df)

        summary = {}
        for regime in MarketRegime:
            mask = df["regime"] == regime.value
            if mask.any():
                regime_data = df[mask]
                returns = regime_data["close"].pct_change()
                summary[regime.value] = {
                    "days": int(mask.sum()),
                    "pct_of_total": round(mask.mean() * 100, 1),
                    "avg_daily_return": round(returns.mean() * 100, 4),
                    "total_return": round(
                        (regime_data["close"].iloc[-1] / regime_data["close"].iloc[0] - 1) * 100, 2
                    ) if len(regime_data) > 1 else 0,
                }

        return summary
