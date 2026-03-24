"""Capital flow indicators organized in 4 layers."""

from .base import Indicator
from .layer1_macro import (
    StablecoinSupplyChange,
    BTCDominanceChange,
    TotalMarketCapMomentum,
    DXYCorrelation,
)
from .layer2_structure import (
    VolumeProfile,
    OpenInterestProxy,
    FundingRateProxy,
    ExchangeReserveEstimate,
)
from .layer3_momentum import (
    MultiTimeframeRSI,
    MACDHistogramDivergence,
    EMARibbon,
    OnBalanceVolumeTrend,
    ChaikinMoneyFlow,
)
from .layer4_sentiment import (
    FearGreedProxy,
    VolatilityRegime,
    MarketBreadth,
)

LAYER1_INDICATORS = [
    StablecoinSupplyChange,
    BTCDominanceChange,
    TotalMarketCapMomentum,
    DXYCorrelation,
]

LAYER2_INDICATORS = [
    VolumeProfile,
    OpenInterestProxy,
    FundingRateProxy,
    ExchangeReserveEstimate,
]

LAYER3_INDICATORS = [
    MultiTimeframeRSI,
    MACDHistogramDivergence,
    EMARibbon,
    OnBalanceVolumeTrend,
    ChaikinMoneyFlow,
]

LAYER4_INDICATORS = [
    FearGreedProxy,
    VolatilityRegime,
    MarketBreadth,
]

ALL_LAYERS = [LAYER1_INDICATORS, LAYER2_INDICATORS, LAYER3_INDICATORS, LAYER4_INDICATORS]
