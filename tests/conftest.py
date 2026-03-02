"""Shared test fixtures — synthetic MES data generators."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_raw_bars():
    """Generate 5,000 synthetic 1-second bars for MES near 5500."""
    np.random.seed(42)
    n = 5000
    base_price = 5500.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.25)

    df = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float) + 1_700_000_000.0,
        "bar_start": np.arange(n, dtype=float) + 1_700_000_000.0,
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 0.5),
        "low": prices - np.abs(np.random.randn(n) * 0.5),
        "close": prices + np.random.randn(n) * 0.1,
        "volume": np.random.randint(100, 3000, n),
        "vwap": prices + np.random.randn(n) * 0.05,
    })
    # Ensure high >= close >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


@pytest.fixture
def synthetic_dollar_bars(synthetic_raw_bars):
    """Build dollar bars from synthetic raw bars."""
    from bayesbot.data.bars import DollarBarBuilder
    from bayesbot.data.models import DollarBarConfig

    config = DollarBarConfig(base_threshold=500_000)
    builder = DollarBarBuilder(config)
    dollar_df = builder.process_dataframe(synthetic_raw_bars)

    # Add fields microstructure features expect
    if "buy_volume" not in dollar_df.columns:
        dollar_df["buy_volume"] = (dollar_df["volume"] * 0.55).astype(int)
        dollar_df["sell_volume"] = dollar_df["volume"] - dollar_df["buy_volume"]
    if "tick_count" not in dollar_df.columns:
        dollar_df["tick_count"] = np.random.randint(5, 50, len(dollar_df))
    if "symbol" not in dollar_df.columns:
        dollar_df["symbol"] = "MES"

    return dollar_df
