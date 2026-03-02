"""Feature engineering registry and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FeatureCategory(Enum):
    PRICE_VOLUME = "price_volume"
    MICROSTRUCTURE = "microstructure"
    CROSS_ASSET = "cross_asset"
    DERIVED = "derived"


@dataclass
class FeatureConfig:
    name: str
    category: FeatureCategory
    window: int           # bars of lookback needed
    description: str


FEATURE_REGISTRY: list[FeatureConfig] = [
    # --- Price / Volume ---
    FeatureConfig("returns_1", FeatureCategory.PRICE_VOLUME, 2,
                  "1-bar log return"),
    FeatureConfig("returns_5", FeatureCategory.PRICE_VOLUME, 6,
                  "5-bar log return"),
    FeatureConfig("returns_20", FeatureCategory.PRICE_VOLUME, 21,
                  "20-bar log return"),
    FeatureConfig("realized_vol_20", FeatureCategory.PRICE_VOLUME, 21,
                  "20-bar realized volatility"),
    FeatureConfig("parkinson_vol_20", FeatureCategory.PRICE_VOLUME, 21,
                  "Parkinson high-low volatility estimator"),
    FeatureConfig("garman_klass_vol_20", FeatureCategory.PRICE_VOLUME, 21,
                  "Garman-Klass OHLC volatility estimator"),
    FeatureConfig("momentum_roc_10", FeatureCategory.PRICE_VOLUME, 11,
                  "Rate of change over 10 bars"),
    FeatureConfig("momentum_roc_50", FeatureCategory.PRICE_VOLUME, 51,
                  "Rate of change over 50 bars"),
    FeatureConfig("vwap_deviation", FeatureCategory.PRICE_VOLUME, 1,
                  "Distance from VWAP: (close - vwap) / vwap"),
    FeatureConfig("volume_sma_ratio", FeatureCategory.PRICE_VOLUME, 21,
                  "Volume relative to 20-bar SMA"),
    FeatureConfig("close_sma20_ratio", FeatureCategory.PRICE_VOLUME, 21,
                  "Price relative to 20-bar SMA"),
    FeatureConfig("close_sma50_ratio", FeatureCategory.PRICE_VOLUME, 51,
                  "Price relative to 50-bar SMA"),
    FeatureConfig("atr_14", FeatureCategory.PRICE_VOLUME, 15,
                  "Average True Range (14-bar EMA)"),
    FeatureConfig("high_low_range", FeatureCategory.PRICE_VOLUME, 1,
                  "Normalized bar range: (H-L)/C"),
    # --- Microstructure ---
    FeatureConfig("kyle_lambda_20", FeatureCategory.MICROSTRUCTURE, 21,
                  "Kyle's lambda: price impact per unit of order flow"),
    FeatureConfig("amihud_20", FeatureCategory.MICROSTRUCTURE, 21,
                  "Amihud illiquidity: avg |return|/dollar_volume"),
    FeatureConfig("buy_sell_imbalance", FeatureCategory.MICROSTRUCTURE, 1,
                  "Order flow direction: (buy - sell) / total"),
    FeatureConfig("order_flow_imbalance_5", FeatureCategory.MICROSTRUCTURE, 6,
                  "Cumulative signed volume over 5 bars"),
    FeatureConfig("bar_duration_ratio", FeatureCategory.MICROSTRUCTURE, 21,
                  "Bar duration relative to 20-bar average"),
    FeatureConfig("tick_count_ratio", FeatureCategory.MICROSTRUCTURE, 21,
                  "Tick count relative to 20-bar average"),
    # --- Cross-asset ---
    FeatureConfig("vix_level", FeatureCategory.CROSS_ASSET, 1,
                  "VIX value (0 if unavailable)"),
    FeatureConfig("vix_change_5", FeatureCategory.CROSS_ASSET, 6,
                  "5-bar VIX change (0 if unavailable)"),
    # --- Derived ---
    FeatureConfig("vol_of_vol_20", FeatureCategory.DERIVED, 41,
                  "Volatility of volatility"),
    FeatureConfig("return_vol_corr_20", FeatureCategory.DERIVED, 21,
                  "Correlation between returns and volume"),
]


def get_max_lookback() -> int:
    return max(f.window for f in FEATURE_REGISTRY)


# Features excluded from HMM training — no real signal for MES futures
EXCLUDED_FEATURES: set[str] = {
    "vix_level",                # Always 0.0 — no VIX data feed
    "vix_change_5",             # Always 0.0 — derived from vix_level
    "buy_sell_imbalance",       # Synthetic: buy_volume = 0.55 * volume
    "order_flow_imbalance_5",   # Synthetic: derived from fake buy/sell split
}


def get_feature_names() -> list[str]:
    """Return active feature names (excluding dead/synthetic features)."""
    return [f.name for f in FEATURE_REGISTRY if f.name not in EXCLUDED_FEATURES]


def get_all_feature_names() -> list[str]:
    """Return all feature names including excluded ones (for raw computation)."""
    return [f.name for f in FEATURE_REGISTRY]
