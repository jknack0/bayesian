"""Feature pipeline: transforms dollar bars into normalized feature vectors."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from bayesbot.data.models import FeatureVector
from bayesbot.features import get_all_feature_names, get_feature_names, get_max_lookback
from bayesbot.features.normalizer import NormalizerState, RollingZScoreNormalizer
from bayesbot.indicators import (
    compute_amihud_illiquidity,
    compute_atr,
    compute_bar_duration_ratio,
    compute_buy_sell_imbalance,
    compute_close_sma_ratio,
    compute_garman_klass_volatility,
    compute_high_low_range,
    compute_kyle_lambda,
    compute_log_returns,
    compute_order_flow_imbalance,
    compute_parkinson_volatility,
    compute_rate_of_change,
    compute_realized_volatility,
    compute_tick_count_ratio,
    compute_volume_sma_ratio,
    compute_vwap_deviation,
)


class FeaturePipeline:
    """Computes all features from dollar bars and normalizes them.

    Two modes:
    1. **Batch** (backtesting): pass an entire DataFrame.
    2. **Streaming** (live): pass the most recent ``required_lookback + 1`` bars.
    """

    def __init__(
        self,
        normalizer_window: int = 252,
        normalizer_min_samples: int = 50,
    ):
        self.normalizer = RollingZScoreNormalizer(
            normalizer_window, normalizer_min_samples
        )
        self.required_lookback = get_max_lookback()
        self.all_feature_names = get_all_feature_names()  # all 24 for raw computation
        self.feature_names = get_feature_names()           # active 20 for HMM

    # ------------------------------------------------------------------
    # Batch (backtest)
    # ------------------------------------------------------------------

    def compute_features_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for a DataFrame of dollar bars.

        Returns columns ``raw_<name>`` and ``norm_<name>`` for each feature.
        First ``required_lookback`` rows will have mostly-zero features.
        """
        features = pd.DataFrame(index=df.index)

        # Price / Volume
        features["returns_1"] = compute_log_returns(df, 1)
        features["returns_5"] = compute_log_returns(df, 5)
        features["returns_20"] = compute_log_returns(df, 20)
        features["realized_vol_20"] = compute_realized_volatility(df, 20)
        features["parkinson_vol_20"] = compute_parkinson_volatility(df, 20)
        features["garman_klass_vol_20"] = compute_garman_klass_volatility(df, 20)
        features["momentum_roc_10"] = compute_rate_of_change(df, 10)
        features["momentum_roc_50"] = compute_rate_of_change(df, 50)
        features["vwap_deviation"] = compute_vwap_deviation(df)
        features["volume_sma_ratio"] = compute_volume_sma_ratio(df, 20)
        features["close_sma20_ratio"] = compute_close_sma_ratio(df, 20)
        features["close_sma50_ratio"] = compute_close_sma_ratio(df, 50)
        features["atr_14"] = compute_atr(df, 14)
        features["high_low_range"] = compute_high_low_range(df)

        # Microstructure
        features["kyle_lambda_20"] = compute_kyle_lambda(df, 20)
        features["amihud_20"] = compute_amihud_illiquidity(df, 20)
        features["buy_sell_imbalance"] = compute_buy_sell_imbalance(df)
        features["order_flow_imbalance_5"] = compute_order_flow_imbalance(df, 5)
        features["bar_duration_ratio"] = compute_bar_duration_ratio(df, 20)
        features["tick_count_ratio"] = compute_tick_count_ratio(df, 20)

        # Cross-asset placeholders
        features["vix_level"] = df.get("vix_level", 0.0)
        features["vix_change_5"] = df.get("vix_change_5", 0.0)

        # Derived
        features["vol_of_vol_20"] = features["realized_vol_20"].rolling(20).std()
        features["return_vol_corr_20"] = features["returns_1"].rolling(20).corr(
            df["volume"].astype(float)
        )

        features = features.fillna(0.0)

        # Normalize (sequential — no look-ahead)
        normalized = self.normalizer.normalize_dataframe(features, self.feature_names)

        result = pd.concat(
            [features.add_prefix("raw_"), normalized.add_prefix("norm_")], axis=1
        )
        logger.info(
            "Computed {} features × {} bars ({} normalized columns)",
            len(self.feature_names),
            len(df),
            len(normalized.columns),
        )
        return result

    # ------------------------------------------------------------------
    # Single-bar (live)
    # ------------------------------------------------------------------

    def compute_features_single(
        self, recent_bars: pd.DataFrame
    ) -> FeatureVector | None:
        """Compute features for the latest bar using recent history.

        Pass the most recent ``required_lookback + 1`` bars.
        """
        if len(recent_bars) < self.required_lookback + 1:
            return None

        all_features = self.compute_features_batch(recent_bars)
        last = all_features.iloc[-1]
        raw = {name: float(last[f"raw_{name}"]) for name in self.all_feature_names}
        norm = {name: float(last[f"norm_{name}"]) for name in self.feature_names}
        bar = recent_bars.iloc[-1]

        return FeatureVector(
            timestamp=float(bar.get("timestamp", 0)),
            bar_index=int(bar.get("bar_index", 0)),
            symbol=str(bar.get("symbol", "MES")),
            features=raw,
            normalized_features=norm,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        return self.normalizer.get_state().__dict__

    def load_state(self, state: dict) -> None:
        self.normalizer.load_state(NormalizerState(**state))
