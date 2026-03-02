#!/usr/bin/env python
"""Generate synthetic MES data for testing the full pipeline.

This creates a realistic-looking 1-second bar dataset with:
- Three distinct regime periods (trending, mean-reverting, volatile)
- Realistic MES price levels (~5500), volume (~100-3000), and dynamics

Usage:
    python scripts/generate_synthetic_data.py --days 90 --output data/MES_synthetic.csv
"""

import click
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_mes(n_days: int = 90, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 1-second bars for MES across multiple regimes."""
    np.random.seed(seed)

    bars_per_day = 23400  # 6.5 hours × 3600 seconds
    total_bars = n_days * bars_per_day

    # Generate regime sequence (each regime lasts 5-30 days)
    regime_labels = []
    remaining = total_bars
    while remaining > 0:
        regime = np.random.choice(["trending_up", "trending_down", "mean_reverting", "volatile"])
        duration = np.random.randint(5, 30) * bars_per_day
        duration = min(duration, remaining)
        regime_labels.extend([regime] * duration)
        remaining -= duration

    regime_labels = regime_labels[:total_bars]

    # Generate prices with regime-specific dynamics
    prices = np.zeros(total_bars)
    prices[0] = 5500.0

    for i in range(1, total_bars):
        regime = regime_labels[i]
        if regime == "trending_up":
            drift = 0.00001
            vol = 0.0003
        elif regime == "trending_down":
            drift = -0.00001
            vol = 0.0003
        elif regime == "mean_reverting":
            drift = -0.000005 * (prices[i - 1] - 5500.0) / 100
            vol = 0.0002
        else:  # volatile
            drift = 0.0
            vol = 0.0008

        prices[i] = prices[i - 1] * (1 + drift + vol * np.random.randn())

    # Build OHLCV
    # Group into 1-second bars (they're already 1-per-second)
    start_ts = 1_700_000_000.0  # arbitrary epoch
    timestamps = start_ts + np.arange(total_bars, dtype=float)

    # Intrabar noise for OHLC
    noise = np.abs(np.random.randn(total_bars)) * 0.25
    highs = prices + noise
    lows = prices - noise
    opens = prices + np.random.randn(total_bars) * 0.1

    # Volume: higher in volatile/trending, lower in mean-reverting
    base_vol = np.ones(total_bars) * 500
    for i, r in enumerate(regime_labels):
        if r == "volatile":
            base_vol[i] = 1500
        elif "trending" in r:
            base_vol[i] = 800
        else:
            base_vol[i] = 400
    volumes = (base_vol + np.abs(np.random.randn(total_bars)) * 200).astype(int)
    volumes = np.maximum(volumes, 1)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.round(opens, 2),
        "high": np.round(np.maximum.reduce([opens, highs, prices]), 2),
        "low": np.round(np.minimum.reduce([opens, lows, prices]), 2),
        "close": np.round(prices, 2),
        "volume": volumes,
        "vwap": np.round(prices + np.random.randn(total_bars) * 0.05, 2),
    })

    # Ensure OHLC sanity
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


@click.command()
@click.option("--days", default=90, help="Number of trading days")
@click.option("--output", default="data/MES_synthetic.csv", help="Output path")
@click.option("--seed", default=42, help="Random seed")
def main(days: int, output: str, seed: int):
    """Generate synthetic MES 1-second data."""
    click.echo(f"Generating {days} days of synthetic MES data...")
    df = generate_synthetic_mes(days, seed)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    click.echo(f"Saved {len(df):,} rows to {output}")
    click.echo(f"  Price range: {df['close'].min():.2f} — {df['close'].max():.2f}")
    click.echo(f"  Date range:  {days} trading days")


if __name__ == "__main__":
    main()
