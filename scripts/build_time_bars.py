#!/usr/bin/env python
"""Build time bars from raw 1-second data for ORB backtesting.

Builds bars at multiple intervals (1-min, 3-min, 5-min) filtered to RTH only.
Reads from parquet for speed (1.35 GB vs 7.8 GB CSV).

Usage:
    python scripts/build_time_bars.py
    python scripts/build_time_bars.py --intervals 60 300  # 1-min and 5-min only
    python scripts/build_time_bars.py --last-years 5      # only recent 5 years
"""

import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from bayesbot.data.time_bars import TimeBarBuilder

_ET = ZoneInfo("America/New_York")

# RTH boundaries for ES/MES
RTH_OPEN_HOUR, RTH_OPEN_MIN = 9, 30
RTH_CLOSE_HOUR, RTH_CLOSE_MIN = 16, 0

INTERVAL_NAMES = {60: "1m", 180: "3m", 300: "5m"}


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Regular Trading Hours (9:30-16:00 ET) only."""
    ts = df["timestamp"].values
    # Convert to ET and filter
    mask = []
    for t in ts:
        dt = datetime.fromtimestamp(t, tz=timezone.utc).astimezone(_ET)
        in_rth = (
            (dt.hour > RTH_OPEN_HOUR or (dt.hour == RTH_OPEN_HOUR and dt.minute >= RTH_OPEN_MIN))
            and (dt.hour < RTH_CLOSE_HOUR or (dt.hour == RTH_CLOSE_HOUR and dt.minute == 0))
        )
        mask.append(in_rth)
    return df[mask].reset_index(drop=True)


def filter_rth_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to RTH using vectorized pandas operations (much faster)."""
    dts = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(_ET)
    hours = dts.dt.hour
    minutes = dts.dt.minute
    time_minutes = hours * 60 + minutes
    rth_start = RTH_OPEN_HOUR * 60 + RTH_OPEN_MIN  # 9:30 = 570
    rth_end = RTH_CLOSE_HOUR * 60 + RTH_CLOSE_MIN   # 16:00 = 960
    mask = (time_minutes >= rth_start) & (time_minutes < rth_end)
    return df[mask].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Build time bars from raw data")
    parser.add_argument(
        "--intervals", type=int, nargs="+", default=[60, 180, 300],
        help="Bar intervals in seconds (default: 60 180 300)"
    )
    parser.add_argument(
        "--last-years", type=int, default=0,
        help="Only use the last N years of data (0 = all)"
    )
    parser.add_argument(
        "--source", default="data/ESMES_15yr_databento.parquet",
        help="Source data file (parquet or CSV)"
    )
    args = parser.parse_args()

    # Load raw data
    logger.info("Loading raw data from {}...", args.source)
    if args.source.endswith(".parquet"):
        df = pd.read_parquet(args.source)
    else:
        df = pd.read_csv(args.source)
    logger.info("Loaded {:,} rows", len(df))

    # Filter to recent years if requested
    if args.last_years > 0:
        cutoff = df["timestamp"].max() - args.last_years * 365.25 * 86400
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
        logger.info("Filtered to last {} years: {:,} rows", args.last_years, len(df))

    # Filter to RTH
    logger.info("Filtering to RTH (9:30-16:00 ET)...")
    df = filter_rth_vectorized(df)
    logger.info("RTH rows: {:,}", len(df))

    if "symbol" not in df.columns:
        df["symbol"] = "MES"

    # Build time bars at each interval
    for interval in args.intervals:
        name = INTERVAL_NAMES.get(interval, f"{interval}s")
        logger.info("Building {}-second ({}) time bars...", interval, name)

        builder = TimeBarBuilder(interval_seconds=interval, symbol="MES")
        bars_df = builder.process_dataframe(df)

        if bars_df.empty:
            logger.warning("No bars produced for interval {}", interval)
            continue

        # Add buy/sell volume (synthetic, same as dollar bars)
        bars_df["buy_volume"] = (bars_df["volume"] * 0.55).astype(int)
        bars_df["sell_volume"] = bars_df["volume"] - bars_df["buy_volume"]

        out_path = f"data/MES_time_{name}.csv"
        bars_df.to_csv(out_path, index=False)

        # Summary
        price_min = bars_df["close"].min()
        price_max = bars_df["close"].max()
        ts_start = datetime.fromtimestamp(bars_df["timestamp"].iloc[0], tz=timezone.utc)
        ts_end = datetime.fromtimestamp(bars_df["timestamp"].iloc[-1], tz=timezone.utc)

        print(f"\n{'='*50}")
        print(f"  {name} bars: {len(bars_df):,} bars")
        print(f"  Price range: {price_min:.2f} - {price_max:.2f}")
        print(f"  Date range: {ts_start:%Y-%m-%d} to {ts_end:%Y-%m-%d}")
        print(f"  Saved to: {out_path}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
