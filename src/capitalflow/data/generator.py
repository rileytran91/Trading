"""Synthetic market data generator for offline testing.

Generates realistic crypto price data with market cycles,
volume patterns, and correlated altcoin prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_btc_data(
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    initial_price: float = 7200.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic BTC OHLCV data with market cycles.

    Simulates:
    - Bull/bear cycles (~1-2 years each)
    - Volatility clustering
    - Volume patterns correlated with price moves
    - Realistic OHLC spreads
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    # Base trend: combination of cycles
    t = np.linspace(0, 1, n)

    # Multi-cycle trend (simulates BTC market cycles)
    # Major cycle: ~4 years (halving cycle)
    major_cycle = np.sin(2 * np.pi * t * 1.2 - np.pi / 4) * 0.8
    # Minor cycle: ~1 year
    minor_cycle = np.sin(2 * np.pi * t * 3.5) * 0.3
    # Micro cycle: ~3 months
    micro_cycle = np.sin(2 * np.pi * t * 12) * 0.1

    trend = major_cycle + minor_cycle + micro_cycle

    # Generate returns with trend, volatility clustering, and fat tails
    base_vol = 0.03  # ~3% daily volatility
    vol_regime = 1.0 + 0.5 * np.abs(np.sin(2 * np.pi * t * 4))  # Volatility cycles

    # GARCH-like volatility clustering
    vol = np.zeros(n)
    vol[0] = base_vol
    for i in range(1, n):
        vol[i] = 0.9 * vol[i - 1] + 0.1 * base_vol * vol_regime[i]

    # Returns: trend component + noise
    trend_returns = np.diff(trend, prepend=trend[0]) * 0.02
    noise = rng.standard_t(df=5, size=n) * vol  # Fat tails
    returns = trend_returns + noise

    # Generate prices
    log_prices = np.log(initial_price) + np.cumsum(returns)
    close = np.exp(log_prices)

    # Generate OHLC from close
    daily_range = vol * close  # Range proportional to volatility
    high = close + rng.uniform(0.3, 0.7, n) * daily_range
    low = close - rng.uniform(0.3, 0.7, n) * daily_range
    open_price = close + rng.uniform(-0.3, 0.3, n) * daily_range

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    # Volume: correlated with price moves and volatility
    base_volume = 30_000_000_000  # ~$30B daily
    vol_factor = vol / base_vol
    move_factor = 1 + 2 * np.abs(returns)  # Big moves = more volume
    trend_factor = 1 + 0.5 * np.maximum(0, trend)  # Bull markets = more volume
    volume = base_volume * vol_factor * move_factor * trend_factor
    volume *= rng.uniform(0.7, 1.3, n)  # Random variation
    volume = np.maximum(volume, 1_000_000)

    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    df.index.name = "date"

    return df


def generate_altcoin_data(
    btc_df: pd.DataFrame,
    symbol: str = "ETH",
    beta: float = 1.3,
    correlation: float = 0.75,
    initial_price: float = 200.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate correlated altcoin data based on BTC price.

    Args:
        btc_df: BTC price DataFrame
        symbol: Altcoin symbol
        beta: Beta relative to BTC (>1 = more volatile)
        correlation: Correlation with BTC returns
        initial_price: Starting price
        seed: Random seed
    """
    rng = np.random.RandomState(seed)
    n = len(btc_df)

    btc_returns = btc_df["close"].pct_change().fillna(0).values

    # Correlated returns: part BTC-correlated, part independent
    independent_noise = rng.standard_t(df=5, size=n) * 0.03
    alt_returns = (
        correlation * beta * btc_returns
        + (1 - correlation) * independent_noise
    )

    # Add altcoin-specific cycles (e.g., altseason rotation)
    t = np.linspace(0, 1, n)
    alt_cycle = np.sin(2 * np.pi * t * 5 + rng.uniform(0, 2 * np.pi)) * 0.005
    alt_returns += alt_cycle

    # Generate prices
    log_prices = np.log(initial_price) + np.cumsum(alt_returns)
    close = np.exp(log_prices)

    vol = np.abs(alt_returns) * close
    high = close + rng.uniform(0.3, 0.7, n) * vol * 10
    low = close - rng.uniform(0.3, 0.7, n) * vol * 10
    open_price = close + rng.uniform(-0.3, 0.3, n) * vol * 10

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    low = np.maximum(low, 0.001)  # Prevent negative prices

    volume = btc_df["volume"].values * rng.uniform(0.05, 0.3, n)

    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=btc_df.index)
    df.index.name = "date"

    return df


def generate_dxy_data(
    btc_df: pd.DataFrame,
    initial_value: float = 96.0,
    seed: int = 123,
) -> pd.DataFrame:
    """Generate DXY data inversely correlated with BTC."""
    rng = np.random.RandomState(seed)
    n = len(btc_df)

    btc_returns = btc_df["close"].pct_change().fillna(0).values

    # DXY moves inversely to BTC with moderate correlation
    dxy_returns = -0.3 * btc_returns + rng.normal(0, 0.003, n)

    # Add DXY-specific trend
    t = np.linspace(0, 1, n)
    dxy_trend = np.sin(2 * np.pi * t * 1.5) * 0.001
    dxy_returns += dxy_trend

    log_prices = np.log(initial_value) + np.cumsum(dxy_returns)
    close = np.exp(log_prices)

    vol = np.abs(dxy_returns) * close * 5
    high = close + rng.uniform(0.2, 0.5, n) * np.maximum(vol, 0.1)
    low = close - rng.uniform(0.2, 0.5, n) * np.maximum(vol, 0.1)
    open_price = close + rng.uniform(-0.2, 0.2, n) * np.maximum(vol, 0.1)

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.uniform(1e6, 5e6, n),
    }, index=btc_df.index)
    df.index.name = "date"

    return df


def build_synthetic_master(
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a complete synthetic master dataframe for testing."""
    btc = generate_btc_data(start, end, seed=seed)
    master = btc.copy()

    # DXY
    dxy = generate_dxy_data(btc, seed=seed + 1)
    master["dxy_close"] = dxy["close"]

    # Total market cap (BTC is ~40-60% of total)
    master["total_mcap"] = master["close"] * 19_500_000 / 0.45

    # BTC dominance
    master["btc_dominance"] = 0.45 + 0.1 * np.sin(
        np.linspace(0, 4 * np.pi, len(btc))
    )

    # Altcoins
    altcoin_configs = [
        ("eth", 1.3, 0.80, 200.0),
        ("bnb", 1.2, 0.70, 35.0),
        ("sol", 1.8, 0.65, 2.0),
        ("ada", 1.5, 0.60, 0.15),
        ("xrp", 1.4, 0.55, 0.20),
        ("dot", 1.6, 0.65, 5.0),
        ("avax", 1.7, 0.60, 3.0),
        ("link", 1.4, 0.65, 8.0),
        ("uni", 1.5, 0.55, 5.0),
        ("matic", 1.6, 0.50, 0.02),
    ]

    for i, (name, beta, corr, price) in enumerate(altcoin_configs):
        alt = generate_altcoin_data(btc, name, beta, corr, price, seed=seed + 10 + i)
        master[f"{name}_close"] = alt["close"]

    # Stablecoin volume proxy
    master["stablecoin_volume"] = master["volume"].rolling(7).mean()

    # Forward fill
    master = master.ffill().dropna(subset=["close"])

    return master
