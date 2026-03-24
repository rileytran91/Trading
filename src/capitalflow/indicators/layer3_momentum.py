"""Layer 3: Technical Momentum Indicators.

Pure price/volume based indicators that confirm capital flow direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Indicator


class MultiTimeframeRSI(Indicator):
    """RSI across multiple periods, averaged for a composite momentum signal.

    Logic:
    - RSI < 30 = oversold = bullish reversal potential
    - RSI > 70 = overbought = bearish reversal potential
    - Gradient mapping between extremes
    """

    name = "multi_rsi"
    layer = 3

    def __init__(self, periods: list[int] | None = None):
        self.periods = periods or [14, 21]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in self.periods:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)

            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()

            rs = self.safe_divide(avg_gain, avg_loss)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # Average RSI
        rsi_cols = [f"rsi_{p}" for p in self.periods]
        df["rsi_avg"] = df[rsi_cols].mean(axis=1)
        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "rsi_avg" not in df.columns:
            df = self.compute(df)
        # RSI signal: oversold=bullish, overbought=bearish
        # Map 0-100 to inverted -1 to +1 (low RSI = bullish = positive signal)
        raw = -(df["rsi_avg"] - 50) / 50
        return raw.clip(-1, 1)


class MACDHistogramDivergence(Indicator):
    """MACD Histogram with divergence detection.

    Logic:
    - Rising MACD histogram = bullish momentum
    - Falling MACD histogram = bearish momentum
    - Price new high + MACD histogram lower high = bearish divergence
    - Price new low + MACD histogram higher low = bullish divergence
    """

    name = "macd_divergence"
    layer = 3

    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()

        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(span=self.signal_period, adjust=False).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

        # Detect divergence using rolling windows
        lookback = 20
        df["macd_hist_slope"] = df["macd_histogram"].diff(5)
        df["price_slope"] = df["close"].diff(5)

        # Bearish divergence: price rising but MACD histogram falling
        df["macd_bearish_div"] = (
            (df["price_slope"] > 0) & (df["macd_hist_slope"] < 0)
        ).astype(float)

        # Bullish divergence: price falling but MACD histogram rising
        df["macd_bullish_div"] = (
            (df["price_slope"] < 0) & (df["macd_hist_slope"] > 0)
        ).astype(float)

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "macd_histogram" not in df.columns:
            df = self.compute(df)

        # Base signal from histogram
        hist_signal = self.normalize(df["macd_histogram"], method="tanh")

        # Divergence adjustments
        div_signal = df.get("macd_bullish_div", 0) * 0.3 - df.get("macd_bearish_div", 0) * 0.3

        combined = (hist_signal * 0.7 + div_signal * 0.3)
        return combined.clip(-1, 1)


class EMARibbon(Indicator):
    """EMA Ribbon (8, 21, 55, 100, 200) for trend strength.

    Logic:
    - All EMAs in order (8>21>55>100>200) = strong uptrend
    - All EMAs inverted = strong downtrend
    - EMA spread (distance between fastest and slowest) = trend strength
    - Price relative to ribbon center = momentum direction
    """

    name = "ema_ribbon"
    layer = 3

    def __init__(self, periods: list[int] | None = None):
        self.periods = periods or [8, 21, 55, 100, 200]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        for p in self.periods:
            df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()

        # Calculate ribbon ordering score
        ema_cols = [f"ema_{p}" for p in self.periods]
        ema_values = df[ema_cols]

        # Count how many pairs are in "bullish order" (shorter > longer)
        n_pairs = 0
        bullish_pairs = pd.Series(0.0, index=df.index)
        for i in range(len(self.periods)):
            for j in range(i + 1, len(self.periods)):
                n_pairs += 1
                bullish_pairs += (
                    df[f"ema_{self.periods[i]}"] > df[f"ema_{self.periods[j]}"]
                ).astype(float)

        df["ema_ribbon_score"] = (bullish_pairs / n_pairs) * 2 - 1  # -1 to +1

        # Ribbon spread as % of price
        df["ema_ribbon_spread"] = (
            (df[f"ema_{self.periods[0]}"] - df[f"ema_{self.periods[-1]}"])
            / df["close"]
        )

        # Price position relative to ribbon center
        ribbon_center = ema_values.mean(axis=1)
        df["price_vs_ribbon"] = (df["close"] - ribbon_center) / ribbon_center

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "ema_ribbon_score" not in df.columns:
            df = self.compute(df)
        # Combine ribbon ordering with price position
        ribbon = df["ema_ribbon_score"] * 0.6
        position = self.normalize(df["price_vs_ribbon"], method="tanh") * 0.4
        return (ribbon + position).clip(-1, 1)


class OnBalanceVolumeTrend(Indicator):
    """On-Balance Volume with trend divergence detection.

    Logic:
    - OBV rising with price = confirmed uptrend (bullish)
    - OBV falling with price rising = distribution (bearish)
    - OBV rising with price falling = accumulation (bullish)
    """

    name = "obv_trend"
    layer = 3

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate OBV
        obv_direction = np.sign(df["close"].diff())
        df["obv"] = (obv_direction * df["volume"]).cumsum()

        # OBV slope (normalized)
        df["obv_slope"] = df["obv"].diff(self.lookback) / df["obv"].rolling(self.lookback).std()

        # Price slope for divergence
        df["price_slope_obv"] = df["close"].pct_change(self.lookback)

        # Divergence score
        df["obv_divergence"] = df["obv_slope"] * np.sign(df["obv_slope"]) - (
            self.normalize(df["price_slope_obv"], method="tanh")
        )

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "obv_slope" not in df.columns:
            df = self.compute(df)

        # OBV slope as primary signal
        obv_sig = self.normalize(df["obv_slope"], method="tanh")

        return obv_sig.clip(-1, 1)


class ChaikinMoneyFlow(Indicator):
    """Chaikin Money Flow - measures buying/selling pressure.

    Logic:
    - CMF > 0 = buying pressure = accumulation = bullish
    - CMF < 0 = selling pressure = distribution = bearish
    - CMF magnitude indicates strength
    """

    name = "cmf"
    layer = 3

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Money Flow Multiplier
        hl_range = df["high"] - df["low"]
        hl_range = hl_range.replace(0, np.nan)
        mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range

        # Money Flow Volume
        mf_volume = mf_multiplier * df["volume"]

        # CMF = sum(MF Volume, period) / sum(Volume, period)
        df["cmf"] = (
            mf_volume.rolling(self.period).sum()
            / df["volume"].rolling(self.period).sum()
        )

        return df

    def signal(self, df: pd.DataFrame) -> pd.Series:
        if "cmf" not in df.columns:
            df = self.compute(df)

        # CMF is already in roughly -1 to +1 range
        # Apply mild normalization for consistency
        return self.normalize(df["cmf"], method="tanh").clip(-1, 1)
