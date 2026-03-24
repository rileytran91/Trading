"""Weighted signal aggregation across all indicator layers."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..indicators import (
    ALL_LAYERS,
    LAYER1_INDICATORS,
    LAYER2_INDICATORS,
    LAYER3_INDICATORS,
    LAYER4_INDICATORS,
)

logger = logging.getLogger(__name__)


class SignalScorer:
    """Aggregates signals from all layers into a composite score.

    Architecture:
    1. Each indicator produces a signal in [-1, +1]
    2. Within each layer, indicators are weighted (configurable)
    3. Layers are weighted: L1(0.35) > L2(0.25) = L3(0.25) > L4(0.15)
    4. Final composite score is in [-1, +1]
    5. Layer agreement boosts/dampens confidence
    """

    def __init__(
        self,
        layer_weights: list[float] | None = None,
        config: dict | None = None,
    ):
        self.layer_weights = layer_weights or [0.35, 0.25, 0.25, 0.15]
        self.config = config or {}
        self._indicators: list[list] = []
        self._build_indicators()

    def _build_indicators(self) -> None:
        """Instantiate all indicator objects."""
        self._indicators = []
        for layer_classes in ALL_LAYERS:
            layer_instances = []
            for cls in layer_classes:
                try:
                    instance = cls()
                    layer_instances.append(instance)
                except Exception as e:
                    logger.warning(f"Failed to instantiate {cls.__name__}: {e}")
            self._indicators.append(layer_instances)

    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all indicators and add their columns to the dataframe."""
        for layer_idx, layer in enumerate(self._indicators):
            for indicator in layer:
                try:
                    df = indicator.compute(df)
                    logger.debug(f"Computed: {indicator.name}")
                except Exception as e:
                    logger.warning(f"Failed to compute {indicator.name}: {e}")
        return df

    def compute_layer_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute weighted score for each layer."""
        for layer_idx, layer in enumerate(self._indicators):
            signals = []
            for indicator in layer:
                try:
                    sig = indicator.signal(df)
                    signals.append(sig)
                    df[f"signal_{indicator.name}"] = sig
                except Exception as e:
                    logger.warning(f"Failed to get signal from {indicator.name}: {e}")

            if signals:
                # Equal weight within layer (can be customized)
                layer_score = pd.concat(signals, axis=1).mean(axis=1)
                df[f"layer{layer_idx + 1}_score"] = layer_score
            else:
                df[f"layer{layer_idx + 1}_score"] = 0.0

        return df

    def compute_composite(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the final composite score from all layers."""
        df = df.copy()  # Avoid fragmentation warnings
        df = self.compute_all_indicators(df)
        df = self.compute_layer_scores(df)

        # Weighted composite
        composite = pd.Series(0.0, index=df.index)
        for i, weight in enumerate(self.layer_weights):
            col = f"layer{i + 1}_score"
            if col in df.columns:
                composite += df[col] * weight

        df["composite_score"] = composite
        df["composite_smooth"] = composite.rolling(3).mean()

        # Layer agreement metric
        layer_cols = [f"layer{i+1}_score" for i in range(4) if f"layer{i+1}_score" in df.columns]
        if layer_cols:
            layer_signs = df[layer_cols].apply(np.sign)
            agreement = layer_signs.apply(
                lambda row: abs(row.sum()) / len(row), axis=1
            )
            df["layer_agreement"] = agreement
            agreement_multiplier = 0.5 + agreement * 0.5
            df["composite_adjusted"] = df["composite_smooth"] * agreement_multiplier
        else:
            df["layer_agreement"] = 0.5
            df["composite_adjusted"] = df["composite_smooth"]

        return df

    def get_current_signal(self, df: pd.DataFrame) -> dict:
        """Get the latest signal summary."""
        if "composite_adjusted" not in df.columns:
            df = self.compute_composite(df)

        last = df.iloc[-1]
        return {
            "composite_score": last.get("composite_adjusted", 0),
            "layer1_macro": last.get("layer1_score", 0),
            "layer2_structure": last.get("layer2_score", 0),
            "layer3_momentum": last.get("layer3_score", 0),
            "layer4_sentiment": last.get("layer4_score", 0),
            "layer_agreement": last.get("layer_agreement", 0),
            "date": df.index[-1],
        }
