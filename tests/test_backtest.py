"""Quick smoke tests for the backtest system."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np

from capitalflow.data.generator import generate_btc_data, build_synthetic_master
from capitalflow.indicators.base import Indicator
from capitalflow.indicators.layer3_momentum import MultiTimeframeRSI, EMARibbon, ChaikinMoneyFlow
from capitalflow.indicators.layer1_macro import TotalMarketCapMomentum
from capitalflow.signals.scorer import SignalScorer
from capitalflow.signals.confidence import classify_confidence, ConfidenceLevel
from capitalflow.signals.regime import RegimeDetector
from capitalflow.backtest.portfolio import Portfolio
from capitalflow.backtest.metrics import PerformanceMetrics


def test_generate_btc_data():
    df = generate_btc_data("2022-01-01", "2023-12-31")
    assert len(df) > 700
    assert all(c in df.columns for c in ["open", "high", "low", "close", "volume"])
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["volume"] > 0).all()


def test_build_synthetic_master():
    master = build_synthetic_master("2022-01-01", "2023-06-30")
    assert "dxy_close" in master.columns
    assert "btc_dominance" in master.columns
    assert "eth_close" in master.columns
    assert len(master) > 500


def test_rsi_indicator():
    df = generate_btc_data("2022-01-01", "2023-12-31")
    rsi = MultiTimeframeRSI(periods=[14])
    df = rsi.compute(df)
    sig = rsi.signal(df)
    assert "rsi_14" in df.columns
    assert sig.min() >= -1.0
    assert sig.max() <= 1.0


def test_ema_ribbon():
    df = generate_btc_data("2022-01-01", "2023-12-31")
    ribbon = EMARibbon(periods=[8, 21, 55])
    df = ribbon.compute(df)
    sig = ribbon.signal(df)
    assert "ema_ribbon_score" in df.columns
    assert sig.dropna().min() >= -1.0
    assert sig.dropna().max() <= 1.0


def test_confidence_classification():
    assert classify_confidence(0.8) == ConfidenceLevel.STRONG_BULLISH
    assert classify_confidence(0.4) == ConfidenceLevel.MEDIUM_BULLISH
    assert classify_confidence(0.0) == ConfidenceLevel.NEUTRAL
    assert classify_confidence(-0.8) == ConfidenceLevel.STRONG_BEARISH


def test_portfolio():
    port = Portfolio(initial_capital=100_000)
    from datetime import datetime
    d1 = datetime(2023, 1, 1)
    d2 = datetime(2023, 2, 1)
    port.open_position(d1, 20000, "long", 1.0)
    assert port.position_direction == "long"
    pnl = port.close_position(d2, 22000)
    assert port.position_direction == "flat"
    assert pnl > 0  # Price went up, should be profit


def test_signal_scorer():
    master = build_synthetic_master("2022-01-01", "2023-12-31")
    scorer = SignalScorer()
    result = scorer.compute_composite(master)
    assert "composite_adjusted" in result.columns
    assert "layer1_score" in result.columns
    assert "layer_agreement" in result.columns


def test_regime_detector():
    master = build_synthetic_master("2022-01-01", "2023-12-31")
    scorer = SignalScorer()
    master = scorer.compute_composite(master)
    detector = RegimeDetector(transition_days=3)
    regimes = detector.detect(master)
    assert len(regimes) == len(master)
    valid_regimes = {"Accumulation", "Markup", "Distribution", "Markdown"}
    assert set(regimes.unique()).issubset(valid_regimes)


if __name__ == "__main__":
    tests = [
        test_generate_btc_data,
        test_build_synthetic_master,
        test_rsi_indicator,
        test_ema_ribbon,
        test_confidence_classification,
        test_portfolio,
        test_signal_scorer,
        test_regime_detector,
    ]
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            print(f"  FAIL: {test.__name__} - {e}")
