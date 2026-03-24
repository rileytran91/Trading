"""Abstract base class for all indicators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Indicator(ABC):
    """Base class for all capital flow indicators.

    Every indicator must:
    1. compute() - Add raw indicator columns to the master DataFrame
    2. signal()  - Return a Series of floats in [-1.0, +1.0] (bearish to bullish)
    """

    name: str = "base"
    layer: int = 0

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to df and return df."""
        ...

    @abstractmethod
    def signal(self, df: pd.DataFrame) -> pd.Series:
        """Return normalized signal in [-1, +1]."""
        ...

    @staticmethod
    def normalize(series: pd.Series, method: str = "tanh") -> pd.Series:
        """Normalize a series to [-1, +1] range."""
        if series.empty:
            return series

        if method == "tanh":
            # z-score then tanh compression
            mean = series.rolling(252, min_periods=30).mean()
            std = series.rolling(252, min_periods=30).std()
            std = std.replace(0, np.nan)
            z = (series - mean) / std
            return np.tanh(z)

        elif method == "minmax":
            roll_min = series.rolling(252, min_periods=30).min()
            roll_max = series.rolling(252, min_periods=30).max()
            span = roll_max - roll_min
            span = span.replace(0, np.nan)
            return ((series - roll_min) / span) * 2 - 1

        elif method == "rsi":
            # Specific for RSI: 0-100 -> -1 to +1
            return (series - 50) / 50

        return series

    @staticmethod
    def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
        """Division with zero protection."""
        return a / b.replace(0, np.nan)
