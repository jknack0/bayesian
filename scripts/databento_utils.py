"""Shared Databento download utilities used by ingest_databento.py and full_pipeline.py."""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
import pandas as pd


def _resolve_api_key(api_key: str | None = None) -> str:
    """Resolve API key from argument, env var, or .env file."""
    key = api_key or os.environ.get("DATABENTO_API_KEY")
    if not key:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("DATABENTO_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if not key:
        click.echo("ERROR: No Databento API key found.")
        click.echo("  Option 1: Add DATABENTO_API_KEY=db-xxxx to your .env file")
        click.echo("  Option 2: Pass --databento-key db-xxxx")
        click.echo("  Option 3: Set DATABENTO_API_KEY environment variable")
        click.echo("")
        click.echo("  Sign up at https://databento.com for $125 free credits")
        sys.exit(1)
    return key


def download_databento_data(
    symbol: str,
    months: int,
    api_key: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Download OHLCV 1-second bars from Databento and return as DataFrame.

    Also saves the raw CSV to output_dir/{symbol}_databento.csv for reuse.
    """
    try:
        import databento as db
    except ImportError:
        click.echo("ERROR: databento not installed. Run: pip install databento")
        sys.exit(1)

    key = _resolve_api_key(api_key)
    client = db.Historical(key)

    end = datetime.utcnow() - timedelta(days=1)  # Historical only, exclude today
    start = end - timedelta(days=months * 30)

    import time as _time

    click.echo(f"  Requesting {symbol} 1-second OHLCV bars...")
    click.echo(f"  Range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    click.echo(f"  This may take a few minutes for {months} months of data...")

    t0 = _time.time()
    click.echo(f"  [1/4] Streaming from Databento API...", nl=False)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[f"{symbol}.c.0"],
        stype_in="continuous",
        schema="ohlcv-1s",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    t1 = _time.time()
    click.echo(f" done ({t1 - t0:.1f}s)")

    click.echo(f"  [2/4] Converting to DataFrame...", nl=False)
    raw = data.to_df()
    t2 = _time.time()
    click.echo(f" done ({t2 - t1:.1f}s) -- {len(raw):,} bars")

    click.echo(f"  [3/4] Converting prices and cleaning...", nl=False)
    # to_df() defaults to price_type="float" (already converted from fixed-point)
    # and puts ts_event in the index as a DatetimeIndex
    result = pd.DataFrame({
        "timestamp": raw.index.astype("int64") / 1e9,
        "open": raw["open"].values,
        "high": raw["high"].values,
        "low": raw["low"].values,
        "close": raw["close"].values,
        "volume": raw["volume"].values,
    })

    result["vwap"] = (result["high"] + result["low"] + result["close"]) / 3
    result["bar_start"] = result["timestamp"]
    result = result[result["volume"] > 0].sort_values("timestamp").reset_index(drop=True)
    t3 = _time.time()
    click.echo(f" done ({t3 - t2:.1f}s)")

    median_close = result["close"].median()
    click.echo(f"  Price range: {result['close'].min():.2f} - {result['close'].max():.2f}")
    click.echo(f"  Median close: {median_close:.2f}")
    click.echo(f"  Clean bars: {len(result):,}")

    if output_dir is not None:
        click.echo(f"  [4/4] Saving CSV...", nl=False)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{symbol}_databento.csv"
        result.to_csv(csv_path, index=False)
        t4 = _time.time()
        click.echo(f" done ({t4 - t3:.1f}s)")
        click.echo(f"  Saved -> {csv_path}")

    total = _time.time() - t0
    click.echo(f"  Total download time: {total:.1f}s")

    return result


def estimate_databento_cost(
    symbol: str,
    months: int,
    api_key: str | None = None,
) -> float:
    """Estimate cost in USD for a Databento download."""
    try:
        import databento as db
    except ImportError:
        click.echo("ERROR: databento not installed. Run: pip install databento")
        sys.exit(1)

    key = _resolve_api_key(api_key)
    client = db.Historical(key)

    end = datetime.utcnow() - timedelta(days=1)  # Historical only, exclude today
    start = end - timedelta(days=months * 30)

    cost = client.metadata.get_cost(
        dataset="GLBX.MDP3",
        symbols=[f"{symbol}.c.0"],
        stype_in="continuous",
        schema="ohlcv-1s",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    # API returns cost in USD as a float (not cents)
    return float(cost)
