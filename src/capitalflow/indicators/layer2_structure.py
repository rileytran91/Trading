"""Layer 2: Market Structure Indicators.

Analyze volume, open interest, and exchange flow patterns
to detect institutional and whale activity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Indicator


class VolumeProfile(Indicator):
    """Volume analysis to detect accumulation/distribution phases.

    Logic:
    - Rising volume on up days = smart money buying = BULLISH
    - Rising volume on down days = smart money selling = BEARISH
    - Declining volume on rallies = distribution = BEARISH
    - Declining volume on drops = selling exhaustion = BULLISH
    - Volume spike detection for climactic events
    """

    name = "volume_profile"
    layer = 2

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Volume SMA for relative comparison
        df["vol_sma"] = df["volume"].rolling(self.lookback).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]

        # Classify volume by price direction
        price_change = df["close"].diff()
        df["up_volume"] = np.where(price_change > 0, df["volume"], 0)
        df["down_volume"] = np.where(price_change < 0, df["volume"], 0)

        # Volume-weighted price trend
        df["up_vol_ma"] = pd.Series(df["up_volume"]).rolling(self.lookback).mean()
        df["down_vol_ma"] = pd.Series(df["down_volume"]).rolling(self.lookback).mean()

        # Buy/sell volume ratio
        total_vol = df["up_vol_ma"] + df["down_vol_ma"]
        total_vol = total_vol.replace(0, np.nan)
        df["vol_buy_ratio"] = (df["up_vol_ma"] - df["down_vol_ma"]) / total_vol

        # Volume trend (is volume increasing or decreasing?)
        df["vol_trend"] = df["vol_sma"].pct_change(self.lookback)

        # Climactic volume detection (>2 std devs above mean)
        vol_std = df["volume"].rolling(self.lookback * 5).std()
        df["vol_climax"] = (df["volume"] - df["vol_sma"]) / vol_std

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "vol_buy_ratio" not in df.columns:
            df = self.compute(df)

        # Primary: buy/sell volume ratio
        buy_sell = self.normalize(df["vol_buy_ratio"], method="tanh")

        # Secondary: volume trend context
        vol_trend = self.normalize(df["vol_trend"], method="tanh")

        combined = buy_sell * 0.7 + vol_trend * 0.3
        return combined.clip(-1, 1)


class OpenInterestProxy(Indicator):
    """Open Interest proxy using volume and price patterns.

    Since direct OI data requires futures APIs, we approximate:
    - Rising volume + rising price = new longs opening = BULLISH
    - Rising volume + falling price = new shorts opening = BEARISH
    - Falling volume + rising price = short covering (weak rally) = NEUTRAL-BEARISH
    - Falling volume + falling price = long liquidation exhaustion = NEUTRAL-BULLISH

    This is the classic volume-price relationship from Wyckoff analysis.
    """

    name = "open_interest_proxy"
    layer = 2

    def __init__(self, lookback: int = 14):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Volume change
        vol_change = df["volume"].pct_change(self.lookback)
        price_change = df["close"].pct_change(self.lookback)

        # Classify regime
        vol_up = vol_change > 0
        price_up = price_change > 0

        # OI proxy score
        df["oi_proxy"] = np.where(
            vol_up & price_up, 1.0,      # New longs - bullish
            np.where(
                vol_up & ~price_up, -1.0,   # New shorts - bearish
                np.where(
                    ~vol_up & price_up, -0.3,  # Short covering - weak
                    0.3                        # Long liquidation exhaustion
                )
            )
        )

        # Smooth to avoid noise
        df["oi_proxy_smooth"] = df["oi_proxy"].rolling(5).mean()

        # Volume-price divergence intensity
        df["vol_price_div"] = self.normalize(vol_change, method="tanh") * np.sign(price_change)

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "oi_proxy_smooth" not in df.columns:
            df = self.compute(df)
        return df["oi_proxy_smooth"].clip(-1, 1)


class FundingRateProxy(Indicator):
    """Funding rate proxy using price premium/discount patterns.

    Logic (contrarian):
    - Sustained positive premium (price > MA) = overleveraged longs = BEARISH
    - Sustained negative premium (price < MA) = overleveraged shorts = BULLISH
    - Uses z-score to detect extreme positioning

    This captures the same information as actual funding rates:
    when price persistently deviates from fair value, it indicates
    leveraged positioning that tends to mean-revert.
    """

    name = "funding_rate_proxy"
    layer = 2

    def __init__(self, lookback: int = 30):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Premium: price deviation from SMA (proxy for futures vs spot spread)
        sma = df["close"].rolling(self.lookback).mean()
        df["price_premium"] = (df["close"] - sma) / sma

        # Z-score of premium
        premium_mean = df["price_premium"].rolling(90).mean()
        premium_std = df["price_premium"].rolling(90).std().replace(0, np.nan)
        df["premium_zscore"] = (df["price_premium"] - premium_mean) / premium_std

        # Sustained deviation detection
        df["premium_duration"] = 0.0
        premium_positive = (df["price_premium"] > 0).astype(float)
        premium_negative = (df["price_premium"] < 0).astype(float)

        # Count consecutive days of positive/negative premium
        pos_streak = premium_positive.groupby(
            (premium_positive != premium_positive.shift()).cumsum()
        ).cumsum()
        neg_streak = premium_negative.groupby(
            (premium_negative != premium_negative.shift()).cumsum()
        ).cumsum()

        df["premium_streak"] = pos_streak - neg_streak

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "premium_zscore" not in df.columns:
            df = self.compute(df)

        # CONTRARIAN: high positive z-score = overleveraged longs = bearish
        contrarian = -self.normalize(df["premium_zscore"], method="tanh")

        return contrarian.clip(-1, 1)


class ExchangeReserveEstimate(Indicator):
    """Exchange reserve estimation using volume patterns.

    Logic:
    - Volume spike with price drop = exchange inflows (selling) = BEARISH
    - Volume spike with price rise = exchange outflows (buying) = BULLISH
    - Sustained high volume = active redistribution phase
    - Low volume = hodling phase = accumulation

    This approximates on-chain exchange flow data using publicly
    available price and volume data.
    """

    name = "exchange_reserve"
    layer = 2

    def __init__(self, lookback: int = 14):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Volume z-score for spike detection
        vol_mean = df["volume"].rolling(self.lookback * 3).mean()
        vol_std = df["volume"].rolling(self.lookback * 3).std().replace(0, np.nan)
        df["vol_zscore"] = (df["volume"] - vol_mean) / vol_std

        # Direction of volume spikes
        price_dir = np.sign(df["close"].diff())
        df["directed_vol_zscore"] = df["vol_zscore"] * price_dir

        # Smoothed exchange flow proxy
        df["exchange_flow_proxy"] = df["directed_vol_zscore"].rolling(self.lookback).mean()

        # Accumulation/distribution score based on volume regime
        vol_percentile = df["volume"].rolling(252).rank(pct=True)
        df["vol_regime"] = np.where(
            vol_percentile < 0.3, 0.5,   # Low volume = accumulation
            np.where(vol_percentile > 0.8, -0.3, 0.0)  # High volume = uncertain
        )

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "exchange_flow_proxy" not in df.columns:
            df = self.compute(df)

        flow = self.normalize(df["exchange_flow_proxy"], method="tanh")
        regime = pd.Series(df["vol_regime"], index=df.index)

        combined = flow * 0.7 + regime * 0.3
        return combined.clip(-1, 1)
