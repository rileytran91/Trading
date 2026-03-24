"""Layer 1: Macro Capital Flow Indicators (LEADING).

These are the most important indicators - they track actual money flow
into and out of the crypto ecosystem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Indicator


class StablecoinSupplyChange(Indicator):
    """Tracks stablecoin supply changes as a proxy for capital inflow/outflow.

    Logic:
    - Increasing stablecoin supply = new money entering crypto = BULLISH
    - Decreasing stablecoin supply = money leaving crypto = BEARISH
    - Rate of change matters more than absolute level
    - Uses 7d and 30d rate of change for different time horizons

    Proxy: We use BTC trading volume as an approximation for stablecoin activity,
    since direct stablecoin market cap data requires paid APIs.
    """

    name = "stablecoin_supply"
    layer = 1

    def __init__(self, lookback: int = 30):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        vol = df.get("stablecoin_volume", df["volume"].rolling(7).mean())

        # Short-term flow (7 days)
        df["stable_flow_7d"] = vol.pct_change(7)

        # Medium-term flow (30 days)
        df["stable_flow_30d"] = vol.pct_change(self.lookback)

        # Volume momentum: accelerating vs decelerating
        df["stable_flow_accel"] = df["stable_flow_7d"] - df["stable_flow_7d"].shift(7)

        # Dollar volume (price * volume) as better proxy for total capital deployed
        df["dollar_volume"] = df["close"] * df["volume"]
        df["dollar_vol_change"] = df["dollar_volume"].pct_change(self.lookback)

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "stable_flow_30d" not in df.columns:
            df = self.compute(df)

        # Combine short and medium term signals
        short = self.normalize(df["stable_flow_7d"], method="tanh")
        medium = self.normalize(df["stable_flow_30d"], method="tanh")
        accel = self.normalize(df["stable_flow_accel"], method="tanh")

        combined = short * 0.3 + medium * 0.4 + accel * 0.3
        return combined.clip(-1, 1)


class BTCDominanceChange(Indicator):
    """BTC Dominance changes as indicator of risk appetite.

    Logic:
    - Rising BTC.D + Rising total mcap = Early bull (money entering through BTC) = BULLISH
    - Rising BTC.D + Falling total mcap = Risk-off flight to BTC = BEARISH
    - Falling BTC.D + Rising total mcap = Altseason / Risk-on = VERY BULLISH
    - Falling BTC.D + Falling total mcap = Panic selling alts = VERY BEARISH

    The combination of BTC dominance direction + total market cap direction
    gives a 2D view of capital flow.
    """

    name = "btc_dominance"
    layer = 1

    def __init__(self, lookback: int = 14):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if "btc_dominance" not in df.columns:
            # Fallback: estimate from BTC price vs total market
            df["btc_dominance"] = 0.5  # neutral if no data

        df["btc_dom_change"] = df["btc_dominance"].pct_change(self.lookback)
        df["btc_dom_trend"] = (
            df["btc_dominance"].rolling(self.lookback).mean()
            - df["btc_dominance"].rolling(self.lookback * 2).mean()
        )

        # Total market cap change for context
        if "total_mcap" in df.columns:
            df["mcap_change"] = df["total_mcap"].pct_change(self.lookback)
        else:
            df["mcap_change"] = df["close"].pct_change(self.lookback)

        # 2D signal: combine BTC.D direction with market direction
        # Both rising = early bull (+0.5)
        # BTC.D falling + mcap rising = altseason (+1.0)
        # BTC.D rising + mcap falling = risk-off (-0.5)
        # Both falling = panic (-1.0)
        btc_d_dir = np.sign(df["btc_dom_change"])
        mcap_dir = np.sign(df["mcap_change"])

        df["dom_mcap_regime"] = np.where(
            (btc_d_dir < 0) & (mcap_dir > 0), 1.0,    # Altseason
            np.where(
                (btc_d_dir > 0) & (mcap_dir > 0), 0.5,  # Early bull
                np.where(
                    (btc_d_dir > 0) & (mcap_dir < 0), -0.5,  # Risk-off
                    -1.0  # Panic
                )
            )
        )

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "dom_mcap_regime" not in df.columns:
            df = self.compute(df)

        # Smoothed regime signal
        regime = df["dom_mcap_regime"].rolling(5).mean()
        return regime.clip(-1, 1)


class TotalMarketCapMomentum(Indicator):
    """Total crypto market cap momentum as proxy for aggregate capital flow.

    Logic:
    - Rising total mcap = net capital inflow = BULLISH
    - Falling total mcap = net capital outflow = BEARISH
    - Rate of change acceleration = strengthening/weakening flow
    - Compare short-term vs long-term momentum for regime shifts
    """

    name = "total_mcap_momentum"
    layer = 1

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        mcap = df.get("total_mcap", df["close"])

        # Multi-period rate of change
        df["mcap_roc_7"] = mcap.pct_change(7)
        df["mcap_roc_20"] = mcap.pct_change(self.lookback)
        df["mcap_roc_60"] = mcap.pct_change(60)

        # Momentum acceleration
        df["mcap_momentum_accel"] = df["mcap_roc_7"] - df["mcap_roc_7"].shift(7)

        # Trend strength: short vs long momentum
        df["mcap_trend_strength"] = df["mcap_roc_20"] - df["mcap_roc_60"]

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "mcap_roc_20" not in df.columns:
            df = self.compute(df)

        roc = self.normalize(df["mcap_roc_20"], method="tanh")
        accel = self.normalize(df["mcap_momentum_accel"], method="tanh")
        trend = self.normalize(df["mcap_trend_strength"], method="tanh")

        combined = roc * 0.4 + accel * 0.3 + trend * 0.3
        return combined.clip(-1, 1)


class DXYCorrelation(Indicator):
    """DXY (Dollar Index) inverse correlation with crypto.

    Logic:
    - Falling DXY = weaker dollar = risk assets rally = BULLISH for crypto
    - Rising DXY = stronger dollar = risk assets decline = BEARISH for crypto
    - Rolling correlation strength matters: strong negative correlation
      means DXY is a reliable signal
    - DXY trend reversal = potential crypto trend reversal
    """

    name = "dxy_correlation"
    layer = 1

    def __init__(self, lookback: int = 30):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if "dxy_close" not in df.columns:
            df["dxy_signal"] = 0.0
            df["dxy_trend"] = 0.0
            df["dxy_btc_corr"] = 0.0
            return df

        # DXY trend (inverted - falling DXY = bullish for crypto)
        df["dxy_trend"] = -df["dxy_close"].pct_change(self.lookback)

        # Rolling correlation between DXY and BTC
        btc_returns = df["close"].pct_change()
        dxy_returns = df["dxy_close"].pct_change()
        df["dxy_btc_corr"] = btc_returns.rolling(self.lookback).corr(dxy_returns)

        # DXY momentum reversal detection
        dxy_sma_short = df["dxy_close"].rolling(10).mean()
        dxy_sma_long = df["dxy_close"].rolling(30).mean()
        # DXY crossing below its MA = bullish for crypto
        df["dxy_signal"] = np.where(dxy_sma_short < dxy_sma_long, 1.0, -1.0)

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "dxy_trend" not in df.columns:
            df = self.compute(df)

        trend = self.normalize(df["dxy_trend"], method="tanh")
        crossover = pd.Series(df["dxy_signal"], index=df.index).rolling(5).mean()

        # Weight by correlation strength (stronger correlation = more reliable signal)
        corr_weight = df["dxy_btc_corr"].abs().rolling(10).mean().fillna(0.5)

        combined = (trend * 0.5 + crossover * 0.5) * corr_weight
        return combined.clip(-1, 1)
