"""Local parquet cache layer for market data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


class DataCache:
    """Caches DataFrames as parquet files for fast reload."""

    def __init__(self, cache_dir: str = "~/.cache/capitalflow"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, symbol: str, timeframe: str, start: str, end: str) -> str:
        raw = f"{symbol}_{timeframe}_{start}_{end}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def get(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame | None:
        key = self._key(symbol, timeframe, start, end)
        path = self._path(key)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                path.unlink(missing_ok=True)
        return None

    def put(self, df: pd.DataFrame, symbol: str, timeframe: str, start: str, end: str) -> None:
        if df.empty:
            return
        key = self._key(symbol, timeframe, start, end)
        path = self._path(key)
        df.to_parquet(path, index=True)

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
