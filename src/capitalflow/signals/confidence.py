"""Confidence level classification for composite signals."""

from __future__ import annotations

from enum import Enum

import pandas as pd


class ConfidenceLevel(Enum):
    STRONG_BULLISH = "Strong Bullish"
    MEDIUM_BULLISH = "Medium Bullish"
    WEAK_BULLISH = "Weak Bullish"
    NEUTRAL = "Neutral"
    WEAK_BEARISH = "Weak Bearish"
    MEDIUM_BEARISH = "Medium Bearish"
    STRONG_BEARISH = "Strong Bearish"


def classify_confidence(
    score: float,
    strong: float = 0.6,
    medium: float = 0.3,
    weak: float = 0.1,
) -> ConfidenceLevel:
    """Classify a composite score into a confidence level."""
    if score >= strong:
        return ConfidenceLevel.STRONG_BULLISH
    elif score >= medium:
        return ConfidenceLevel.MEDIUM_BULLISH
    elif score >= weak:
        return ConfidenceLevel.WEAK_BULLISH
    elif score > -weak:
        return ConfidenceLevel.NEUTRAL
    elif score > -medium:
        return ConfidenceLevel.WEAK_BEARISH
    elif score > -strong:
        return ConfidenceLevel.MEDIUM_BEARISH
    else:
        return ConfidenceLevel.STRONG_BEARISH


def classify_series(
    scores: pd.Series,
    strong: float = 0.6,
    medium: float = 0.3,
    weak: float = 0.1,
) -> pd.Series:
    """Classify a series of composite scores into confidence levels."""
    return scores.apply(lambda x: classify_confidence(x, strong, medium, weak).value)


def confidence_to_position(level: ConfidenceLevel) -> float:
    """Map confidence level to position size multiplier (0 to 1)."""
    mapping = {
        ConfidenceLevel.STRONG_BULLISH: 1.0,
        ConfidenceLevel.MEDIUM_BULLISH: 0.7,
        ConfidenceLevel.WEAK_BULLISH: 0.3,
        ConfidenceLevel.NEUTRAL: 0.0,
        ConfidenceLevel.WEAK_BEARISH: -0.3,
        ConfidenceLevel.MEDIUM_BEARISH: -0.7,
        ConfidenceLevel.STRONG_BEARISH: -1.0,
    }
    return mapping.get(level, 0.0)
