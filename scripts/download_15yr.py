#!/usr/bin/env python
"""Download 15 years of ES/MES 1-second OHLCV data from Databento.

MES launched May 2019, so this uses ES for earlier years.
Downloads in yearly chunks so a crash doesn't lose everything.
Saves each chunk as Parquet for efficient storage (~10x smaller than CSV).

Usage:
    # Download all 15 years (ES 2011-2019, MES 2019-present):
    python scripts/download_15yr.py

    # Download ES only (consistent single contract):
    python scripts/download_15yr.py --symbol ES --years 15

    # Resume after a crash (skips already-downloaded chunks):
    python scripts/download_15yr.py  # just re-run, it skips existing files

    # Combine chunks into single CSV for pipeline:
    python scripts/download_15yr.py --combine-only
"""

import os
import sys
import time
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
        click.echo("  Set DATABENTO_API_KEY in .env or pass --api-key")
        sys.exit(1)
    return key


def _download_chunk(
    client,
    symbol: str,
    start_date: str,
    end_date: str,
    output_path: Path,
) -> int:
    """Download one chunk of data and save as Parquet. Returns row count."""
    t0 = time.time()
    click.echo(f"    Streaming {symbol} {start_date} -> {end_date}...", nl=False)

    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[f"{symbol}.c.0"],
        stype_in="continuous",
        schema="ohlcv-1s",
        start=start_date,
        end=end_date,
    )
    t1 = time.time()
    click.echo(f" API={t1 - t0:.0f}s", nl=False)

    df = data.to_df()
    t2 = time.time()
    click.echo(f" df={t2 - t1:.0f}s", nl=False)

    # to_df() puts ts_event in the index, prices already float
    result = pd.DataFrame({
        "timestamp": df.index.astype("int64") / 1e9,
        "open": df["open"].values,
        "high": df["high"].values,
        "low": df["low"].values,
        "close": df["close"].values,
        "volume": df["volume"].values,
    })
    result["vwap"] = (result["high"] + result["low"] + result["close"]) / 3
    result = result[result["volume"] > 0].sort_values("timestamp").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False, engine="pyarrow")
    t3 = time.time()

    click.echo(f" save={t3 - t2:.0f}s -- {len(result):,} bars")
    return len(result)


def _generate_chunks(years: int, mes_start_year: int = 2019) -> list[dict]:
    """Generate yearly download chunks.

    Returns list of dicts with: symbol, start, end, filename
    """
    now = datetime.utcnow()
    end_date = now - timedelta(days=1)
    start_date = datetime(end_date.year - years, end_date.month, 1)

    chunks = []
    current = start_date
    while current < end_date:
        year_end = datetime(current.year + 1, 1, 1)
        chunk_end = min(year_end, end_date)

        # MES launched May 2019; use ES before that
        if current.year < mes_start_year:
            symbol = "ES"
        elif current.year == mes_start_year and current.month < 5:
            # Jan-Apr 2019: still ES
            symbol = "ES"
        else:
            symbol = "MES"

        chunks.append({
            "symbol": symbol,
            "start": current.strftime("%Y-%m-%d"),
            "end": chunk_end.strftime("%Y-%m-%d"),
            "filename": f"{symbol}_{current.year}.parquet",
        })
        current = year_end

    return chunks


def _combine_chunks(chunk_dir: Path, output_csv: Path, output_parquet: Path) -> int:
    """Combine all Parquet chunks into single CSV and Parquet files."""
    parquet_files = sorted(chunk_dir.glob("*.parquet"))
    if not parquet_files:
        click.echo("  ERROR: No Parquet files found to combine")
        return 0

    click.echo(f"  Combining {len(parquet_files)} chunks...")
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        click.echo(f"    {pf.name}: {len(df):,} bars, price {df['close'].median():.2f}")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # Normalize ES prices to MES scale (they're the same, just different tick values)
    # ES and MES have the same price quotes, no scaling needed

    # Remove any duplicates at chunk boundaries
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="first")
    n_dupes = n_before - len(combined)
    if n_dupes > 0:
        click.echo(f"  Removed {n_dupes:,} duplicate timestamps at chunk boundaries")

    click.echo(f"  Total: {len(combined):,} bars")
    click.echo(f"  Date range: {pd.to_datetime(combined['timestamp'].iloc[0], unit='s')} -> "
               f"{pd.to_datetime(combined['timestamp'].iloc[-1], unit='s')}")
    click.echo(f"  Price range: {combined['close'].min():.2f} - {combined['close'].max():.2f}")

    click.echo(f"  Saving combined Parquet...", nl=False)
    t0 = time.time()
    combined.to_parquet(output_parquet, index=False, engine="pyarrow")
    click.echo(f" done ({time.time() - t0:.0f}s)")

    click.echo(f"  Saving combined CSV...", nl=False)
    t0 = time.time()
    combined.to_csv(output_csv, index=False)
    click.echo(f" done ({time.time() - t0:.0f}s)")

    click.echo(f"  -> {output_parquet}")
    click.echo(f"  -> {output_csv}")
    return len(combined)


@click.command()
@click.option("--api-key", default=None, help="Databento API key")
@click.option("--symbol", default=None, help="Force single symbol (ES or MES). Default: auto-switch")
@click.option("--years", default=15, help="Years of history to download")
@click.option("--chunk-dir", default="data/chunks", help="Directory for yearly chunks")
@click.option("--output-name", default=None, help="Output filename stem (default: based on symbol/years)")
@click.option("--combine-only", is_flag=True, help="Skip download, just combine existing chunks")
@click.option("--skip-combine", is_flag=True, help="Download only, don't combine into single file")
@click.option("--cost-only", is_flag=True, help="Estimate cost for each chunk without downloading")
def main(api_key, symbol, years, chunk_dir, output_name, combine_only, skip_combine, cost_only):
    """Download 15 years of ES/MES data in yearly chunks."""
    chunk_path = Path(chunk_dir)
    chunk_path.mkdir(parents=True, exist_ok=True)

    if output_name is None:
        if symbol:
            output_name = f"{symbol}_{years}yr"
        else:
            output_name = f"ESMES_{years}yr"

    output_csv = Path("data") / f"{output_name}_databento.csv"
    output_parquet = Path("data") / f"{output_name}_databento.parquet"

    if combine_only:
        click.echo(f"\nCombining existing chunks from {chunk_path}/")
        _combine_chunks(chunk_path, output_csv, output_parquet)
        return

    # Generate download plan
    if symbol:
        # Single symbol mode — override auto-switching
        chunks = _generate_chunks(years, mes_start_year=9999 if symbol == "ES" else 0)
        for c in chunks:
            c["symbol"] = symbol
            c["filename"] = f"{symbol}_{c['start'][:4]}.parquet"
    else:
        chunks = _generate_chunks(years)

    # Show plan
    click.echo(f"\nDownload plan: {years} years in {len(chunks)} yearly chunks")
    click.echo(f"Chunk directory: {chunk_path}/")
    click.echo(f"Output: {output_csv}")
    click.echo("")
    for c in chunks:
        tag = " [DONE]" if (chunk_path / c["filename"]).exists() else ""
        click.echo(f"  {c['symbol']}  {c['start']} -> {c['end']}  {c['filename']}{tag}")

    # Cost estimation mode
    if cost_only:
        try:
            import databento as db
        except ImportError:
            click.echo("ERROR: databento not installed. Run: pip install databento")
            sys.exit(1)

        key = _resolve_api_key(api_key)
        client = db.Historical(key)

        click.echo(f"\nEstimating cost per chunk...\n")
        total_cost = 0.0
        for c in chunks:
            try:
                cost = client.metadata.get_cost(
                    dataset="GLBX.MDP3",
                    symbols=[f"{c['symbol']}.c.0"],
                    stype_in="continuous",
                    schema="ohlcv-1s",
                    start=c["start"],
                    end=c["end"],
                )
                cost_usd = float(cost)
                total_cost += cost_usd
                click.echo(f"  {c['symbol']}  {c['start'][:4]}  ${cost_usd:>8.2f}")
            except Exception as e:
                click.echo(f"  {c['symbol']}  {c['start'][:4]}  ERROR: {e}")

        click.echo(f"  {'':─<30}")
        click.echo(f"  {'TOTAL':<16}  ${total_cost:>8.2f}")
        click.echo(f"\n  Note: With a Databento subscription, historical downloads are included.")
        click.echo(f"  Without subscription, this would be charged per request.")
        return

    # Check which chunks already exist (resume support)
    pending = [c for c in chunks if not (chunk_path / c["filename"]).exists()]
    if not pending:
        click.echo(f"\nAll {len(chunks)} chunks already downloaded!")
    else:
        skip_count = len(chunks) - len(pending)
        if skip_count > 0:
            click.echo(f"\nSkipping {skip_count} already-downloaded chunks")
        click.echo(f"Downloading {len(pending)} remaining chunks...\n")

        try:
            import databento as db
        except ImportError:
            click.echo("ERROR: databento not installed. Run: pip install databento")
            sys.exit(1)

        key = _resolve_api_key(api_key)
        client = db.Historical(key)

        total_bars = 0
        t_start = time.time()
        for i, chunk in enumerate(pending, 1):
            click.echo(f"  [{i}/{len(pending)}] {chunk['symbol']} {chunk['start'][:4]}:")
            try:
                n = _download_chunk(
                    client,
                    chunk["symbol"],
                    chunk["start"],
                    chunk["end"],
                    chunk_path / chunk["filename"],
                )
                total_bars += n
            except Exception as e:
                click.echo(f"\n  ERROR on {chunk['filename']}: {e}")
                click.echo(f"  Re-run the script to resume from this chunk.")
                sys.exit(1)

        elapsed = time.time() - t_start
        click.echo(f"\nDownload complete: {total_bars:,} new bars in {elapsed:.0f}s")

    # Combine
    if not skip_combine:
        click.echo(f"\nCombining all chunks...")
        total = _combine_chunks(chunk_path, output_csv, output_parquet)
        click.echo(f"\nDone! {total:,} total 1-second bars ready.")
        click.echo(f"Run pipeline with: python scripts/ingest_databento.py --file {output_csv}")


if __name__ == "__main__":
    main()
