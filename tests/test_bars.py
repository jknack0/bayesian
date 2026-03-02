"""Tests for dollar bar construction."""

import numpy as np
import pandas as pd
import pytest

from bayesbot.data.bars import DollarBarBuilder
from bayesbot.data.models import DollarBarConfig, RawTick


class TestDollarBarBuilderTicks:
    def test_normal_bar_completion(self):
        config = DollarBarConfig(base_threshold=10_000)
        builder = DollarBarBuilder(config)

        # Each tick: price=100, volume=50 → $5,000/tick → 2 ticks = 1 bar
        result = builder.process_tick(RawTick(1.0, 100.0, 50, "buy"))
        assert result is None  # not enough yet

        result = builder.process_tick(RawTick(2.0, 100.0, 50, "sell"))
        assert result is not None
        assert result.dollar_volume >= 10_000
        assert result.open == 100.0
        assert result.buy_volume == 50
        assert result.sell_volume == 50

    def test_bar_index_increments(self):
        config = DollarBarConfig(base_threshold=5_000)
        builder = DollarBarBuilder(config)

        bars = []
        for i in range(20):
            result = builder.process_tick(RawTick(float(i), 100.0, 100, "buy"))
            if result is not None:
                bars.append(result)

        assert len(bars) >= 2
        for i in range(1, len(bars)):
            assert bars[i].bar_index == bars[i - 1].bar_index + 1


class TestDollarBarBuilderDataFrame:
    def test_basic_construction(self, synthetic_raw_bars):
        config = DollarBarConfig(base_threshold=500_000)
        builder = DollarBarBuilder(config)
        result = builder.process_dataframe(synthetic_raw_bars)

        assert len(result) > 0
        assert "open" in result.columns
        assert "dollar_volume" in result.columns
        assert all(result["dollar_volume"] > 0)

    def test_zero_volume_bars_skipped(self):
        df = pd.DataFrame({
            "timestamp": [1.0, 2.0, 3.0, 4.0],
            "bar_start": [1.0, 2.0, 3.0, 4.0],
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.0] * 4,
            "volume": [1000, 0, 0, 1000],
            "vwap": [100.0] * 4,
        })
        config = DollarBarConfig(base_threshold=50_000)
        builder = DollarBarBuilder(config)
        result = builder.process_dataframe(df)
        # Should not crash and may produce bars from the non-zero rows
        assert isinstance(result, pd.DataFrame)

    def test_threshold_adaptation(self, synthetic_raw_bars):
        config = DollarBarConfig(base_threshold=500_000, ema_window=20)
        builder = DollarBarBuilder(config)
        builder.process_dataframe(synthetic_raw_bars)

        # After processing data, threshold should have adapted
        assert builder._threshold != config.base_threshold or builder.bars_generated == 0

    def test_bars_generated_property(self, synthetic_raw_bars):
        config = DollarBarConfig(base_threshold=500_000)
        builder = DollarBarBuilder(config)
        result = builder.process_dataframe(synthetic_raw_bars)
        assert builder.bars_generated == len(result)

    def test_reset_clears_state(self, synthetic_raw_bars):
        config = DollarBarConfig(base_threshold=500_000)
        builder = DollarBarBuilder(config)
        builder.process_dataframe(synthetic_raw_bars)
        assert builder.bars_generated > 0

        builder.reset()
        assert builder.bars_generated == 0
        assert builder._threshold == config.base_threshold
