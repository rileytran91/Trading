"""Symbol mappings and proxy definitions.

Maps logical indicator names to concrete data sources.
"""

# yfinance tickers for macro data
YFINANCE_SYMBOLS = {
    "DXY": "DX-Y.NYB",           # US Dollar Index
    "BTC_USD": "BTC-USD",         # Bitcoin in USD
    "ETH_USD": "ETH-USD",         # Ethereum in USD
    "TOTAL_MCAP": "^CMC200",      # Total crypto market cap proxy (CMC Crypto 200)
    "SP500": "^GSPC",             # S&P 500 for correlation
    "GOLD": "GC=F",              # Gold futures
    "US10Y": "^TNX",             # 10-Year Treasury yield
}

# Stablecoin CoinGecko IDs for market cap tracking
STABLECOIN_IDS = ["tether", "usd-coin"]

# CCXT symbol mappings
CCXT_BASE_PAIRS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
}

# Default altcoins for market breadth analysis
DEFAULT_ALTCOINS = [
    "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT",
    "DOT/USDT", "AVAX/USDT", "LINK/USDT", "UNI/USDT", "MATIC/USDT",
]


def ccxt_to_yfinance(symbol: str) -> str:
    """Convert CCXT symbol format to yfinance ticker."""
    base = symbol.replace("/USDT", "").replace("/USD", "")
    return f"{base}-USD"
