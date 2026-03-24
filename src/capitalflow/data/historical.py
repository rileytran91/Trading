"""Realistic historical price data matching actual BTC/ETH cycles 2020-2025.

This generates synthetic data that closely follows the actual market structure:
- Jan 2020: BTC ~$7,200, ETH ~$130
- Mar 2020: COVID crash to ~$3,800 / ~$90
- Dec 2020: Recovery to ~$29,000 / ~$730
- Apr 2021: First peak ~$64,000 / ~$4,300
- Jul 2021: Summer correction ~$29,000 / ~$1,700
- Nov 2021: ATH ~$69,000 / ~$4,800
- Jun 2022: Bear market ~$17,600 / ~$880
- Nov 2022: FTX crash bottom ~$15,500 / ~$1,070
- Jan 2024: ETF approval rally ~$46,000 / ~$2,300
- Mar 2024: New ATH ~$73,000 / ~$4,000
- Dec 2024: Bull run ~$95,000 / ~$3,400
- Mar 2025: Correction ~$84,000 / ~$2,000
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Actual BTC price milestones (date, price) - key turning points
BTC_MILESTONES = [
    ("2020-01-01", 7200),
    ("2020-02-15", 10300),
    ("2020-03-12", 3800),    # COVID crash
    ("2020-05-11", 8700),    # Halving
    ("2020-07-27", 11100),
    ("2020-10-01", 10600),
    ("2020-12-31", 29000),   # Year-end rally
    ("2021-01-10", 40000),
    ("2021-02-21", 57000),
    ("2021-03-13", 61000),
    ("2021-04-14", 64800),   # First peak
    ("2021-05-19", 30000),   # May crash
    ("2021-06-22", 28800),
    ("2021-07-20", 29300),   # Summer bottom
    ("2021-09-07", 52700),
    ("2021-10-20", 66000),
    ("2021-11-10", 69000),   # ATH
    ("2021-12-04", 42000),
    ("2021-12-31", 46300),
    ("2022-01-24", 33000),
    ("2022-03-28", 47500),   # Dead cat bounce
    ("2022-05-12", 26700),   # LUNA crash
    ("2022-06-18", 17600),   # Capitulation
    ("2022-08-15", 24400),
    ("2022-09-21", 18500),
    ("2022-11-09", 15500),   # FTX crash
    ("2022-12-31", 16500),
    ("2023-01-14", 21200),
    ("2023-03-14", 26500),
    ("2023-04-14", 30400),
    ("2023-06-23", 30700),
    ("2023-07-13", 31400),
    ("2023-08-17", 26000),
    ("2023-10-16", 28500),
    ("2023-10-24", 34500),   # ETF hype
    ("2023-12-08", 44000),
    ("2023-12-31", 42200),
    ("2024-01-11", 46500),   # ETF approval
    ("2024-02-29", 62000),
    ("2024-03-14", 73100),   # New ATH
    ("2024-04-20", 64000),   # Halving
    ("2024-05-01", 57000),
    ("2024-06-07", 71000),
    ("2024-07-05", 54000),
    ("2024-08-05", 49500),   # Yen carry trade unwind
    ("2024-09-28", 65800),
    ("2024-11-06", 69500),   # Election
    ("2024-11-22", 98800),
    ("2024-12-05", 96000),
    ("2024-12-17", 107000),  # Peak
    ("2024-12-31", 95000),
    ("2025-01-20", 106000),  # Trump inauguration
    ("2025-02-03", 98000),
    ("2025-02-28", 84000),   # Tariff fears
    ("2025-03-11", 77000),
    ("2025-03-24", 84000),
]

# ETH/BTC ratio milestones
ETH_BTC_RATIO = [
    ("2020-01-01", 0.018),
    ("2020-03-12", 0.024),
    ("2020-07-01", 0.025),
    ("2020-12-31", 0.025),
    ("2021-02-05", 0.035),
    ("2021-04-14", 0.040),
    ("2021-05-12", 0.082),   # ETH rally
    ("2021-06-22", 0.065),
    ("2021-09-03", 0.076),
    ("2021-11-10", 0.070),
    ("2021-12-31", 0.080),
    ("2022-06-18", 0.055),
    ("2022-09-15", 0.080),   # Merge
    ("2022-11-09", 0.075),
    ("2022-12-31", 0.073),
    ("2023-04-14", 0.068),
    ("2023-07-13", 0.064),
    ("2023-12-31", 0.054),
    ("2024-03-14", 0.050),
    ("2024-06-01", 0.054),
    ("2024-09-15", 0.040),
    ("2024-12-17", 0.036),
    ("2024-12-31", 0.036),
    ("2025-01-20", 0.031),
    ("2025-03-24", 0.024),
]

# DXY milestones
DXY_MILESTONES = [
    ("2020-01-01", 96.5),
    ("2020-03-20", 103.0),   # COVID flight to safety
    ("2020-08-01", 93.0),
    ("2020-12-31", 89.9),
    ("2021-05-25", 89.6),
    ("2021-09-30", 94.2),
    ("2021-12-31", 95.7),
    ("2022-05-01", 103.2),
    ("2022-09-28", 114.1),   # Peak
    ("2022-12-31", 103.5),
    ("2023-07-14", 99.6),
    ("2023-10-03", 107.0),
    ("2023-12-28", 101.0),
    ("2024-04-16", 106.3),
    ("2024-09-27", 100.4),
    ("2024-12-31", 108.0),
    ("2025-01-13", 110.0),
    ("2025-03-24", 104.0),
]

# Actual market regimes (manually labeled from historical analysis)
REGIME_LABELS = [
    ("2020-01-01", "2020-03-12", "Distribution"),    # Pre-COVID distribution
    ("2020-03-12", "2020-03-23", "Markdown"),         # COVID crash
    ("2020-03-23", "2020-07-27", "Accumulation"),     # Post-crash accumulation
    ("2020-07-27", "2021-04-14", "Markup"),            # Bull run phase 1
    ("2021-04-14", "2021-05-19", "Distribution"),      # First top distribution
    ("2021-05-19", "2021-07-20", "Markdown"),          # May-July crash
    ("2021-07-20", "2021-09-07", "Accumulation"),      # Summer accumulation
    ("2021-09-07", "2021-11-10", "Markup"),            # Bull run phase 2
    ("2021-11-10", "2022-01-24", "Distribution"),      # Top distribution
    ("2022-01-24", "2022-03-28", "Markdown"),          # Jan crash
    ("2022-03-28", "2022-05-12", "Distribution"),      # Dead cat bounce
    ("2022-05-12", "2022-06-18", "Markdown"),          # LUNA crash
    ("2022-06-18", "2022-08-15", "Accumulation"),      # Summer bottom
    ("2022-08-15", "2022-11-09", "Markdown"),          # FTX prelude
    ("2022-11-09", "2023-01-14", "Accumulation"),      # FTX bottom
    ("2023-01-14", "2023-04-14", "Markup"),            # Early 2023 rally
    ("2023-04-14", "2023-10-16", "Accumulation"),      # Sideways accumulation
    ("2023-10-16", "2024-03-14", "Markup"),            # ETF rally
    ("2024-03-14", "2024-05-01", "Distribution"),      # Post-ATH distribution
    ("2024-05-01", "2024-08-05", "Accumulation"),      # Summer accumulation
    ("2024-08-05", "2024-12-17", "Markup"),            # Election + bull run
    ("2024-12-17", "2025-02-03", "Distribution"),      # Top distribution
    ("2025-02-03", "2025-03-24", "Markdown"),          # Tariff crash
]


def interpolate_milestones(
    milestones: list[tuple[str, float]],
    start: str,
    end: str,
    noise_pct: float = 0.015,
    seed: int = 42,
) -> pd.Series:
    """Interpolate between price milestones with realistic noise."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start, end, freq="D")

    # Build milestone series
    ms_dates = pd.to_datetime([m[0] for m in milestones])
    ms_values = [m[1] for m in milestones]
    ms_series = pd.Series(ms_values, index=ms_dates)

    # Reindex to daily and interpolate
    ms_daily = ms_series.reindex(dates)
    ms_daily = ms_daily.interpolate(method="cubicspline")

    # Forward/backward fill edges
    ms_daily = ms_daily.ffill().bfill()

    # Add realistic noise (autocorrelated)
    n = len(dates)
    raw_noise = rng.normal(0, 1, n)
    # Autocorrelate the noise
    smooth_noise = pd.Series(raw_noise).ewm(span=3).mean().values
    noise = smooth_noise * noise_pct * ms_daily.values

    result = ms_daily + noise
    result = result.clip(lower=ms_daily.values * 0.85, upper=ms_daily.values * 1.15)

    return result


def build_realistic_master(
    start: str = "2020-01-01",
    end: str = "2025-03-24",
    seed: int = 42,
) -> pd.DataFrame:
    """Build master dataframe with realistic BTC/ETH data following actual market cycles."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    # BTC price (interpolated from milestones)
    btc_close = interpolate_milestones(BTC_MILESTONES, start, end, noise_pct=0.012, seed=seed)

    # Generate OHLC from close
    daily_returns = btc_close.pct_change().fillna(0)
    daily_vol = daily_returns.rolling(20).std().fillna(0.02)

    btc_range = daily_vol * btc_close
    btc_high = btc_close + rng.uniform(0.2, 0.8, n) * btc_range
    btc_low = btc_close - rng.uniform(0.2, 0.8, n) * btc_range
    btc_open = btc_close.shift(1).fillna(btc_close.iloc[0]) + rng.uniform(-0.3, 0.3, n) * btc_range

    btc_high = np.maximum(btc_high, np.maximum(btc_open, btc_close))
    btc_low = np.minimum(btc_low, np.minimum(btc_open, btc_close))

    # Volume: higher during volatile periods and uptrends
    base_vol = 25_000_000_000
    vol_mult = (1 + 3 * daily_vol / 0.03) * (1 + np.abs(daily_returns))
    volume = base_vol * vol_mult * rng.uniform(0.7, 1.3, n)

    master = pd.DataFrame({
        "open": btc_open.values,
        "high": btc_high.values,
        "low": btc_low.values,
        "close": btc_close.values,
        "volume": volume.values if hasattr(volume, 'values') else volume,
    }, index=dates)
    master.index.name = "date"

    # ETH price (from BTC * ETH/BTC ratio)
    eth_ratio = interpolate_milestones(ETH_BTC_RATIO, start, end, noise_pct=0.008, seed=seed + 1)
    master["eth_close"] = btc_close.values * eth_ratio.values

    # DXY
    dxy = interpolate_milestones(DXY_MILESTONES, start, end, noise_pct=0.003, seed=seed + 2)
    master["dxy_close"] = dxy.values

    # Total market cap (BTC is ~40-50% of total, varies)
    btc_dom_base = interpolate_milestones([
        ("2020-01-01", 0.68), ("2020-08-01", 0.58), ("2021-01-01", 0.70),
        ("2021-02-15", 0.62), ("2021-05-12", 0.40), ("2021-06-15", 0.45),
        ("2021-09-15", 0.42), ("2021-11-10", 0.42), ("2022-01-01", 0.40),
        ("2022-06-18", 0.47), ("2022-12-31", 0.41), ("2023-06-01", 0.48),
        ("2023-10-16", 0.52), ("2024-01-11", 0.52), ("2024-04-01", 0.52),
        ("2024-12-01", 0.57), ("2025-01-01", 0.58), ("2025-03-24", 0.61),
    ], start, end, noise_pct=0.005, seed=seed + 3)

    master["btc_dominance"] = btc_dom_base.values
    btc_mcap = btc_close.values * 19_500_000
    master["total_mcap"] = btc_mcap / btc_dom_base.values

    # Altcoins (correlated with ETH but different betas)
    altcoin_configs = [
        ("bnb", 1.1, 0.75, 15, [("2020-01-01", 15), ("2020-03-12", 8), ("2021-05-10", 680),
                                   ("2021-11-10", 660), ("2022-06-18", 200), ("2022-12-31", 245),
                                   ("2023-06-01", 305), ("2024-03-14", 610), ("2024-12-17", 720),
                                   ("2025-03-24", 600)]),
        ("sol", 2.0, 0.60, 1, [("2020-01-01", 0.5), ("2020-09-01", 3.5), ("2021-04-01", 45),
                                 ("2021-11-06", 260), ("2022-06-18", 25), ("2022-12-31", 10),
                                 ("2023-10-01", 24), ("2024-03-14", 190), ("2024-11-22", 260),
                                 ("2025-01-19", 280), ("2025-03-24", 130)]),
        ("ada", 1.5, 0.55, 0.03, [("2020-01-01", 0.03), ("2020-03-12", 0.02), ("2021-02-27", 1.45),
                                     ("2021-05-16", 1.70), ("2021-09-02", 3.10), ("2021-12-31", 1.30),
                                     ("2022-06-18", 0.43), ("2022-12-31", 0.25), ("2023-12-31", 0.60),
                                     ("2024-03-14", 0.76), ("2024-12-17", 1.10), ("2025-03-24", 0.68)]),
        ("xrp", 1.4, 0.45, 0.19, [("2020-01-01", 0.19), ("2020-03-12", 0.13), ("2021-04-14", 1.85),
                                     ("2021-06-22", 0.58), ("2021-11-10", 1.20), ("2022-06-18", 0.30),
                                     ("2022-12-31", 0.35), ("2023-07-13", 0.80), ("2024-03-14", 0.68),
                                     ("2024-12-01", 2.50), ("2025-01-15", 3.20), ("2025-03-24", 2.30)]),
        ("dot", 1.6, 0.60, 4.0, [("2020-08-20", 4.0), ("2021-04-14", 47), ("2021-05-23", 18),
                                    ("2021-11-04", 55), ("2022-06-18", 6.5), ("2022-12-31", 4.3),
                                    ("2023-12-31", 8.5), ("2024-03-14", 10.5), ("2025-03-24", 4.2)]),
        ("avax", 1.8, 0.55, 3.0, [("2020-09-22", 3.0), ("2021-02-10", 57), ("2021-05-23", 10),
                                     ("2021-11-21", 145), ("2022-06-18", 14), ("2022-12-31", 10.5),
                                     ("2023-12-31", 40), ("2024-03-14", 55), ("2024-12-06", 55),
                                     ("2025-03-24", 20)]),
        ("link", 1.3, 0.60, 2.0, [("2020-01-01", 2.0), ("2020-08-16", 19.8), ("2020-12-31", 12),
                                     ("2021-05-10", 50), ("2021-11-10", 35), ("2022-06-18", 5.5),
                                     ("2022-12-31", 5.6), ("2023-12-31", 15.5), ("2024-03-14", 20),
                                     ("2024-12-12", 28), ("2025-03-24", 14)]),
        ("uni", 1.5, 0.55, 4.0, [("2020-09-17", 4.0), ("2021-05-03", 45), ("2021-06-22", 16),
                                    ("2021-11-10", 26), ("2022-06-18", 3.8), ("2022-12-31", 5.1),
                                    ("2023-12-31", 7.3), ("2024-03-14", 14), ("2024-12-08", 17),
                                    ("2025-03-24", 8)]),
        ("matic", 1.7, 0.50, 0.015, [("2020-01-01", 0.015), ("2020-08-24", 0.025),
                                        ("2021-02-15", 0.15), ("2021-05-18", 2.45), ("2021-06-22", 1.0),
                                        ("2021-12-27", 2.90), ("2022-06-18", 0.35), ("2022-12-31", 0.76),
                                        ("2023-12-31", 1.0), ("2024-03-14", 1.20), ("2024-12-31", 0.50),
                                        ("2025-03-24", 0.22)]),
    ]

    for name, beta, corr, init_price, milestones in altcoin_configs:
        try:
            alt_price = interpolate_milestones(milestones, start, end, noise_pct=0.02, seed=seed + hash(name) % 1000)
            master[f"{name}_close"] = alt_price.values
        except Exception:
            # Fallback if milestones don't cover full range
            master[f"{name}_close"] = master["eth_close"] * (init_price / 130) * rng.uniform(0.8, 1.2, n)

    # Stablecoin volume proxy
    master["stablecoin_volume"] = master["volume"].rolling(7).mean()

    # Actual regime labels
    master["actual_regime"] = _label_regimes(dates, REGIME_LABELS)

    master = master.ffill().bfill()
    return master


def _label_regimes(dates: pd.DatetimeIndex, labels: list[tuple]) -> pd.Series:
    """Create regime labels from date ranges."""
    regimes = pd.Series("Unknown", index=dates)
    for start_str, end_str, regime in labels:
        start_dt = pd.Timestamp(start_str)
        end_dt = pd.Timestamp(end_str)
        mask = (dates >= start_dt) & (dates < end_dt)
        regimes[mask] = regime
    return regimes
