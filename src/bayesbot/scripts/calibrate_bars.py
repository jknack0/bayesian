"""CLI: Find optimal dollar bar threshold.

Usage:
    bayesbot calibrate --file data/MES_raw.csv --symbol MES --target-bars 100
"""

import click
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from tabulate import tabulate

from bayesbot.data.bars import DollarBarBuilder
from bayesbot.data.models import DollarBarConfig


@click.command("calibrate")
@click.option("--file", "file_path", required=True, help="Path to raw 1s CSV")
@click.option("--symbol", default="MES")
@click.option("--target-bars", default=100, help="Target bars per session")
def calibrate(file_path: str, symbol: str, target_bars: int):
    """Find optimal dollar bar threshold for target bars per session."""
    raw_df = pd.read_csv(file_path)
    if "vwap" not in raw_df.columns:
        raw_df["vwap"] = raw_df["close"]
    if "bar_start" not in raw_df.columns:
        raw_df["bar_start"] = raw_df["timestamp"]

    # Test logarithmically spaced thresholds
    thresholds = np.logspace(np.log10(500_000), np.log10(10_000_000), 15)

    results = []
    for thresh in thresholds:
        config = DollarBarConfig(base_threshold=thresh)
        builder = DollarBarBuilder(config)
        dollar_df = builder.process_dataframe(raw_df)

        if dollar_df.empty:
            continue

        # Bars per day
        dollar_df["date"] = pd.to_datetime(dollar_df["timestamp"], unit="s").dt.date
        bars_per_day = dollar_df.groupby("date").size()

        # Return statistics
        ret = np.log(dollar_df["close"] / dollar_df["close"].shift(1)).dropna()
        if len(ret) < 10:
            continue

        jb_stat, jb_p = jarque_bera(ret)
        autocorr = float(ret.autocorr(1)) if len(ret) > 1 else 0

        # Average bar duration
        if "bar_start" in dollar_df.columns:
            dur = (dollar_df["timestamp"] - dollar_df["bar_start"])
            avg_dur_min = dur.mean() / 60
        else:
            avg_dur_min = 0

        results.append({
            "threshold": f"${thresh:,.0f}",
            "avg_bars/day": f"{bars_per_day.mean():.1f}",
            "std_bars": f"{bars_per_day.std():.1f}",
            "avg_dur_min": f"{avg_dur_min:.1f}",
            "mean_ret": f"{ret.mean():.6f}",
            "std_ret": f"{ret.std():.6f}",
            "autocorr(1)": f"{autocorr:.4f}",
            "JB_stat": f"{jb_stat:.1f}",
            "JB_p": f"{jb_p:.4f}",
            "_raw_thresh": thresh,
            "_raw_bars": bars_per_day.mean(),
            "_raw_jb": jb_stat,
        })

    if not results:
        click.echo("ERROR: No thresholds produced valid bars")
        return

    # Find best: closest to target with lowest JB
    for r in results:
        dist = abs(r["_raw_bars"] - target_bars)
        r["_score"] = dist + r["_raw_jb"] * 0.01  # penalize non-normality

    results.sort(key=lambda x: x["_score"])
    best = results[0]

    # Display
    display = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    click.echo(f"\n  Calibration Results for {symbol} (target={target_bars} bars/session)\n")
    click.echo(tabulate(display, headers="keys", tablefmt="simple"))
    click.echo(f"\n  ★ Recommended threshold: {best['threshold']}")
    click.echo(f"    ({best['avg_bars/day']} bars/day, JB={best['JB_stat']}, autocorr={best['autocorr(1)']})")
