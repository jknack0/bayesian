"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from bayesbot.features import EXCLUDED_FEATURES, get_all_feature_names, get_feature_names, get_max_lookback
from bayesbot.features.normalizer import RollingZScoreNormalizer
from bayesbot.features.pipeline import FeaturePipeline
from bayesbot.indicators import (
    compute_atr,
    compute_garman_klass_volatility,
    compute_log_returns,
    compute_parkinson_volatility,
    compute_realized_volatility,
)


class TestPriceVolumeFeatures:
    def test_log_returns(self):
        df = pd.DataFrame({"close": [100.0, 101.0, 99.0, 102.0]})
        ret = compute_log_returns(df, 1)
        assert np.isnan(ret.iloc[0])
        assert abs(ret.iloc[1] - np.log(101 / 100)) < 1e-10

    def test_realized_volatility(self, synthetic_dollar_bars):
        vol = compute_realized_volatility(synthetic_dollar_bars, 20)
        valid = vol.dropna()
        assert len(valid) > 0
        assert all(valid >= 0)

    def test_parkinson_positive(self, synthetic_dollar_bars):
        vol = compute_parkinson_volatility(synthetic_dollar_bars, 20)
        valid = vol.dropna()
        assert all(valid >= 0)

    def test_garman_klass_positive(self, synthetic_dollar_bars):
        vol = compute_garman_klass_volatility(synthetic_dollar_bars, 20)
        valid = vol.dropna()
        assert all(valid >= 0)

    def test_atr_positive(self, synthetic_dollar_bars):
        atr = compute_atr(synthetic_dollar_bars, 14)
        valid = atr.dropna()
        assert all(valid >= 0)


class TestNormalizer:
    def test_cold_start_returns_zero(self):
        norm = RollingZScoreNormalizer(window=100, min_samples=50)
        for i in range(49):
            result = norm.normalize({"feature_a": float(i)})
            assert result["feature_a"] == 0.0

    def test_normal_operation(self):
        norm = RollingZScoreNormalizer(window=100, min_samples=10)
        # Feed 100 values from N(0,1)
        np.random.seed(42)
        for _ in range(100):
            norm.normalize({"x": float(np.random.randn())})
        # A value near the mean should have z-score near 0
        result = norm.normalize({"x": 0.0})
        assert abs(result["x"]) < 2.0

    def test_outlier_clamping(self):
        norm = RollingZScoreNormalizer(window=100, min_samples=10, clamp_range=(-4, 4))
        for _ in range(100):
            norm.normalize({"x": 0.0})
        result = norm.normalize({"x": 100.0})  # extreme outlier
        assert result["x"] == 4.0

    def test_state_roundtrip(self):
        norm = RollingZScoreNormalizer(window=50, min_samples=5)
        for i in range(20):
            norm.normalize({"a": float(i), "b": float(i * 2)})
        state = norm.get_state()

        norm2 = RollingZScoreNormalizer(window=50, min_samples=5)
        norm2.load_state(state)

        r1 = norm.normalize({"a": 10.0, "b": 20.0})
        r2 = norm2.normalize({"a": 10.0, "b": 20.0})
        assert r1 == r2


class TestFeaturePipeline:
    def test_batch_computation(self, synthetic_dollar_bars):
        pipeline = FeaturePipeline(normalizer_min_samples=10)
        result = pipeline.compute_features_batch(synthetic_dollar_bars)

        assert len(result) == len(synthetic_dollar_bars)
        feature_names = get_feature_names()
        for fn in feature_names:
            assert f"raw_{fn}" in result.columns
            assert f"norm_{fn}" in result.columns
        # Excluded features have raw columns but no norm columns
        for fn in EXCLUDED_FEATURES:
            assert f"raw_{fn}" in result.columns
            assert f"norm_{fn}" not in result.columns

    def test_normalized_features_in_range(self, synthetic_dollar_bars):
        pipeline = FeaturePipeline(normalizer_min_samples=10)
        result = pipeline.compute_features_batch(synthetic_dollar_bars)
        feature_names = get_feature_names()

        for fn in feature_names:
            col = result[f"norm_{fn}"]
            assert col.min() >= -4.0
            assert col.max() <= 4.0

    def test_single_bar_matches_batch(self, synthetic_dollar_bars):
        if len(synthetic_dollar_bars) < get_max_lookback() + 2:
            pytest.skip("Not enough bars")

        pipeline = FeaturePipeline(normalizer_min_samples=10)
        batch = pipeline.compute_features_batch(synthetic_dollar_bars)

        # Reset and compute single
        pipeline2 = FeaturePipeline(normalizer_min_samples=10)
        lookback = get_max_lookback()
        recent = synthetic_dollar_bars.iloc[-(lookback + 1):].reset_index(drop=True)
        fv = pipeline2.compute_features_single(recent)

        assert fv is not None
        # Raw features should match the last row of batch (all features, including excluded)
        for fn in get_all_feature_names():
            batch_val = batch.iloc[-1][f"raw_{fn}"]
            single_val = fv.features[fn]
            if not (np.isnan(batch_val) and np.isnan(single_val)):
                assert abs(batch_val - single_val) < 1e-3, f"{fn}: batch={batch_val}, single={single_val}"
