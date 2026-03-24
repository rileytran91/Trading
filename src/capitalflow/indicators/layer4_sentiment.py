"""Layer 4: Sentiment & Confirmation Indicators.

These indicators provide confirmation of signals from other layers
and help avoid false signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Indicator


class FearGreedProxy(Indicator):
    """Fear & Greed Index proxy built from market data.

    Composite of:
    - Volatility (25%): High vol = fear, low vol = greed
    - Momentum (25%): Price vs 125d SMA
    - Volume (25%): Buy vs sell volume
    - Market breadth (25%): Altcoin performance

    Output: 0 = Extreme Fear, 100 = Extreme Greed
    Signal: CONTRARIAN at extremes
    - Extreme Fear (<20) = BULLISH (buy when others are fearful)
    - Extreme Greed (>80) = BEARISH (sell when others are greedy)
    """

    name = "fear_greed_proxy"
    layer = 4

    def __init__(self, lookback: int = 30):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Volatility component (inverted: high vol = low score = fear)
        returns = df["close"].pct_change()
        vol = returns.rolling(self.lookback).std() * np.sqrt(252)
        vol_percentile = vol.rolling(252).rank(pct=True)
        vol_score = (1 - vol_percentile) * 100  # High vol = low score

        # 2. Momentum component
        sma_125 = df["close"].rolling(125).mean()
        momentum = (df["close"] - sma_125) / sma_125
        mom_percentile = momentum.rolling(252).rank(pct=True)
        mom_score = mom_percentile * 100

        # 3. Volume component (buy pressure)
        price_change = df["close"].diff()
        up_vol = np.where(price_change > 0, df["volume"], 0)
        down_vol = np.where(price_change < 0, df["volume"], 0)
        up_avg = pd.Series(up_vol, index=df.index).rolling(self.lookback).mean()
        down_avg = pd.Series(down_vol, index=df.index).rolling(self.lookback).mean()
        total = up_avg + down_avg
        total = total.replace(0, np.nan)
        vol_ratio = up_avg / total
        vol_score_comp = vol_ratio * 100

        # 4. Market breadth component (if altcoin data available)
        alt_cols = [c for c in df.columns if c.endswith("_close") and c not in ("eth_close",)]
        if alt_cols:
            breadth_scores = []
            for col in alt_cols:
                sma50 = df[col].rolling(50).mean()
                breadth_scores.append((df[col] > sma50).astype(float))
            breadth = pd.concat(breadth_scores, axis=1).mean(axis=1) * 100
        else:
            breadth = pd.Series(50, index=df.index)

        # Composite Fear & Greed
        df["fear_greed"] = (
            vol_score * 0.25
            + mom_score * 0.25
            + vol_score_comp * 0.25
            + breadth * 0.25
        )

        # Smooth
        df["fear_greed_smooth"] = df["fear_greed"].rolling(7).mean()

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "fear_greed_smooth" not in df.columns:
            df = self.compute(df)

        fg = df["fear_greed_smooth"]

        # CONTRARIAN signal at extremes, trend-following in middle
        signal = pd.Series(0.0, index=df.index)

        # Extreme fear = bullish
        signal = signal.where(fg >= 20, 1.0)
        # High fear = mildly bullish
        signal = signal.where(~((fg >= 20) & (fg < 35)), 0.5)
        # Neutral
        signal = signal.where(~((fg >= 35) & (fg <= 65)), 0.0)
        # High greed = mildly bearish
        signal = signal.where(~((fg > 65) & (fg <= 80)), -0.5)
        # Extreme greed = bearish
        signal = signal.where(fg <= 80, -1.0)

        return signal.rolling(3).mean().clip(-1, 1)


class VolatilityRegime(Indicator):
    """Volatility regime classification using ATR.

    Logic:
    - Low volatility regime = compression = BULLISH (breakout coming)
    - High volatility + downtrend = capitulation = BULLISH (bottom signal)
    - High volatility + uptrend = distribution = BEARISH (top signal)
    - Medium volatility = trending = follow current trend

    Wyckoff cycle alignment:
    - Low vol = Accumulation/Re-accumulation
    - Expanding vol = Markup/Markdown beginning
    - High vol = Distribution/Capitulation
    """

    name = "volatility_regime"
    layer = 4

    def __init__(self, atr_period: int = 14, percentile_window: int = 252):
        self.atr_period = atr_period
        self.percentile_window = percentile_window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # ATR calculation
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.atr_period).mean()

        # ATR as % of price
        df["atr_pct"] = df["atr"] / df["close"]

        # ATR percentile (historical context)
        df["atr_percentile"] = df["atr_pct"].rolling(self.percentile_window).rank(pct=True)

        # Volatility regime classification
        df["vol_regime_class"] = np.where(
            df["atr_percentile"] < 0.25, "low_vol",
            np.where(df["atr_percentile"] > 0.75, "high_vol", "medium_vol")
        )

        # Volatility direction (expanding or contracting?)
        df["vol_expanding"] = df["atr_pct"].diff(self.atr_period) > 0

        # Trend direction context
        df["trend_dir"] = np.sign(df["close"].pct_change(20))

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "atr_percentile" not in df.columns:
            df = self.compute(df)

        signal = pd.Series(0.0, index=df.index)

        atr_pct = df["atr_percentile"]
        trend = df["trend_dir"]
        expanding = df["vol_expanding"]

        # Low vol = accumulation/compression = mildly bullish
        low_vol = atr_pct < 0.25
        signal = signal.where(~low_vol, 0.4)

        # High vol + downtrend = capitulation = bullish (contrarian)
        capitulation = (atr_pct > 0.75) & (trend < 0)
        signal = signal.where(~capitulation, 0.6)

        # High vol + uptrend = euphoria/distribution = bearish
        euphoria = (atr_pct > 0.75) & (trend > 0)
        signal = signal.where(~euphoria, -0.6)

        # Medium vol: follow the trend
        medium_vol = (atr_pct >= 0.25) & (atr_pct <= 0.75)
        signal = signal.where(~medium_vol, trend * 0.3)

        return signal.rolling(5).mean().clip(-1, 1)


class MarketBreadth(Indicator):
    """Market breadth: % of altcoins in uptrend.

    Logic:
    - >70% alts above 50d SMA = broad rally = BULLISH
    - <30% alts above 50d SMA = broad decline = BEARISH
    - Breadth divergence from BTC = important signal:
      * BTC rising but breadth falling = narrowing rally = BEARISH
      * BTC falling but breadth rising = accumulation in alts = BULLISH

    This measures how widespread the capital flow is across the market.
    """

    name = "market_breadth"
    layer = 4

    def __init__(self, ma_period: int = 50):
        self.ma_period = ma_period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Find altcoin columns
        alt_cols = [c for c in df.columns if c.endswith("_close")]

        if not alt_cols:
            df["breadth_pct"] = 50.0
            df["breadth_signal"] = 0.0
            return df

        # Calculate % of alts above their MA
        above_ma = []
        for col in alt_cols:
            ma = df[col].rolling(self.ma_period).mean()
            above_ma.append((df[col] > ma).astype(float))

        df["breadth_pct"] = pd.concat(above_ma, axis=1).mean(axis=1) * 100

        # Breadth momentum
        df["breadth_change"] = df["breadth_pct"].diff(7)

        # BTC vs breadth divergence
        btc_trend = df["close"].pct_change(14)
        breadth_trend = df["breadth_pct"].diff(14)

        # Divergence: BTC direction vs breadth direction disagree
        df["breadth_divergence"] = (
            self.normalize(breadth_trend, method="tanh")
            - self.normalize(btc_trend, method="tanh")
        )

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "breadth_pct" not in df.columns:
            df = self.compute(df)

        # Base signal from breadth level
        breadth = (df["breadth_pct"] - 50) / 50  # -1 to +1

        # Divergence bonus/penalty
        div = df.get("breadth_divergence", pd.Series(0, index=df.index))

        combined = breadth * 0.6 + div * 0.4
        return combined.rolling(5).mean().clip(-1, 1)
