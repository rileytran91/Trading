"""Configuration loader for the backtest system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    start_date: str = "2020-01-01"
    end_date: str = "2026-03-24"
    base_symbol: str = "BTC/USDT"
    altcoins: list[str] = field(default_factory=list)
    cache_dir: str = "~/.cache/capitalflow"
    exchange: str = "binance"
    timeframe: str = "1d"


@dataclass
class BacktestConfig:
    train_days: int = 365
    test_days: int = 90
    step_days: int = 90
    initial_capital: float = 100_000
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    entry_threshold: float = 0.3
    exit_threshold: float = 0.1


@dataclass
class SignalConfig:
    strong_threshold: float = 0.6
    medium_threshold: float = 0.3
    weak_threshold: float = 0.1
    regime_transition_days: int = 5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    layers: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> Config:
        if path is None:
            path = Path(__file__).parents[2] / "config" / "default.yaml"
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        data_raw = raw.get("data", {})
        data_cfg = DataConfig(
            start_date=data_raw.get("start_date", "2020-01-01"),
            end_date=data_raw.get("end_date", "2026-03-24"),
            base_symbol=data_raw.get("base_symbol", "BTC/USDT"),
            altcoins=data_raw.get("altcoins", []),
            cache_dir=os.path.expanduser(data_raw.get("cache_dir", "~/.cache/capitalflow")),
            exchange=data_raw.get("exchange", "binance"),
            timeframe=data_raw.get("timeframe", "1d"),
        )

        bt_raw = raw.get("backtest", {})
        wf = bt_raw.get("walk_forward", {})
        bt_cfg = BacktestConfig(
            train_days=wf.get("train_days", 365),
            test_days=wf.get("test_days", 90),
            step_days=wf.get("step_days", 90),
            initial_capital=bt_raw.get("initial_capital", 100_000),
            commission_pct=bt_raw.get("commission_pct", 0.001),
            slippage_pct=bt_raw.get("slippage_pct", 0.0005),
            entry_threshold=bt_raw.get("entry_threshold", 0.3),
            exit_threshold=bt_raw.get("exit_threshold", 0.1),
        )

        sig_raw = raw.get("signals", {})
        sig_cfg = SignalConfig(
            strong_threshold=sig_raw.get("strong_threshold", 0.6),
            medium_threshold=sig_raw.get("medium_threshold", 0.3),
            weak_threshold=sig_raw.get("weak_threshold", 0.1),
            regime_transition_days=sig_raw.get("regime_transition_days", 5),
        )

        return cls(
            data=data_cfg,
            backtest=bt_cfg,
            signals=sig_cfg,
            layers=raw.get("layers", {}),
        )
