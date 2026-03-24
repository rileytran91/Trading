"""Unified data fetcher wrapping yfinance and ccxt."""

from __future__ import annotations

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from .cache import DataCache
from .symbols import YFINANCE_SYMBOLS, DEFAULT_ALTCOINS, ccxt_to_yfinance

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches crypto and macro data from multiple sources."""

    def __init__(self, cache_dir: str = "~/.cache/capitalflow"):
        self.cache = DataCache(cache_dir)

    def fetch_ohlcv(
        self, symbol: str, start: str, end: str, timeframe: str = "1d"
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol. Uses yfinance as primary source."""
        cached = self.cache.get(symbol, timeframe, start, end)
        if cached is not None:
            logger.info(f"Cache hit: {symbol}")
            return cached

        ticker = self._resolve_ticker(symbol)
        logger.info(f"Fetching {symbol} ({ticker}) from yfinance...")

        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.index.name = "date"
            df.index = pd.to_datetime(df.index)

            self.cache.put(df, symbol, timeframe, start, end)
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return pd.DataFrame()

    def fetch_multi_symbols(
        self, symbols: list[str], start: str, end: str, timeframe: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        result = {}
        for sym in symbols:
            df = self.fetch_ohlcv(sym, start, end, timeframe)
            if not df.empty:
                result[sym] = df
            time.sleep(0.3)  # Rate limiting
        return result

    def fetch_btc(self, start: str, end: str) -> pd.DataFrame:
        return self.fetch_ohlcv("BTC/USDT", start, end)

    def fetch_dxy(self, start: str, end: str) -> pd.DataFrame:
        return self.fetch_ohlcv("DXY", start, end)

    def fetch_altcoins(
        self, start: str, end: str, altcoins: list[str] | None = None
    ) -> dict[str, pd.DataFrame]:
        symbols = altcoins or DEFAULT_ALTCOINS
        return self.fetch_multi_symbols(symbols, start, end)

    def fetch_stablecoin_proxy(self, start: str, end: str) -> pd.DataFrame:
        """Approximate stablecoin supply using USDT/USD pair volume as proxy.

        Higher USDT trading volume + price stability near $1 = more stablecoin activity.
        We use total crypto market cap minus BTC+ETH as an alternative proxy.
        """
        total = self.fetch_ohlcv("TOTAL_MCAP", start, end)
        btc = self.fetch_ohlcv("BTC/USDT", start, end)

        if total.empty or btc.empty:
            return pd.DataFrame()

        # Use BTC volume as proxy for stablecoin flow
        # High volume = more stablecoin being used for trading
        df = pd.DataFrame(index=btc.index)
        df["stablecoin_proxy"] = btc["volume"].rolling(7).mean()
        df["stablecoin_proxy_change"] = df["stablecoin_proxy"].pct_change(30)
        return df.dropna()

    def fetch_total_mcap(self, start: str, end: str) -> pd.DataFrame:
        return self.fetch_ohlcv("TOTAL_MCAP", start, end)

    def build_master_dataframe(
        self, start: str, end: str, altcoins: list[str] | None = None
    ) -> pd.DataFrame:
        """Build the master dataframe with all required data for indicators."""
        logger.info("Building master dataframe...")

        # Primary BTC data
        btc = self.fetch_btc(start, end)
        if btc.empty:
            logger.warning("Could not fetch live data, falling back to synthetic data")
            from .generator import build_synthetic_master
            return build_synthetic_master(start, end)

        master = btc.copy()

        # DXY
        dxy = self.fetch_dxy(start, end)
        if not dxy.empty:
            master["dxy_close"] = dxy["close"].reindex(master.index, method="ffill")

        # Total market cap proxy
        total = self.fetch_total_mcap(start, end)
        if not total.empty:
            master["total_mcap"] = total["close"].reindex(master.index, method="ffill")
        else:
            # Fallback: use BTC price as rough market cap proxy
            master["total_mcap"] = master["close"] * 19_500_000  # approx BTC supply

        # ETH for BTC dominance estimation
        eth = self.fetch_ohlcv("ETH/USDT", start, end)
        if not eth.empty:
            eth_mcap = eth["close"] * 120_000_000  # approx ETH supply
            btc_mcap = master["close"] * 19_500_000
            if "total_mcap" in master.columns:
                master["btc_dominance"] = btc_mcap / master["total_mcap"]
            else:
                master["btc_dominance"] = btc_mcap / (btc_mcap + eth_mcap)
            master["eth_close"] = eth["close"].reindex(master.index, method="ffill")

        # Altcoin data for market breadth
        alt_data = self.fetch_altcoins(start, end, altcoins)
        for sym, df in alt_data.items():
            col_name = sym.replace("/USDT", "").lower() + "_close"
            master[col_name] = df["close"].reindex(master.index, method="ffill")

        # Stablecoin proxy
        master["stablecoin_volume"] = master["volume"].rolling(7).mean()

        # Forward fill any remaining NaNs from alignment
        master = master.ffill().dropna(subset=["close"])

        logger.info(f"Master dataframe: {len(master)} rows, {len(master.columns)} columns")
        return master

    @staticmethod
    def _resolve_ticker(symbol: str) -> str:
        if symbol in YFINANCE_SYMBOLS:
            return YFINANCE_SYMBOLS[symbol]
        if "/" in symbol:
            return ccxt_to_yfinance(symbol)
        return symbol
