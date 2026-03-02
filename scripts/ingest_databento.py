#!/usr/bin/env python
"""Ingest MES futures data from Databento API and run the full pipeline.

Setup:
    1. Sign up at https://databento.com (you get $125 free credits)
    2. Get your API key from the portal (starts with db-)
    3. pip install databento
    4. Set DATABENTO_API_KEY in your .env or pass --api-key

Usage:
    # Download 12 months of MES 1-second bars and run pipeline:
    python scripts/ingest_databento.py --months 12

    # Download only, no pipeline:
    python scripts/ingest_databento.py --months 6 --download-only

    # Use existing downloaded data:
    python scripts/ingest_databento.py --file data/MES_databento.csv

    # Estimate cost before downloading:
    python scripts/ingest_databento.py --months 12 --cost-only
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
import pandas as pd
from loguru import logger


def _get_client(api_key: str | None = None):
    """Initialize Databento client with API key from arg, env, or .env file."""
    try:
        import databento as db
    except ImportError:
        click.echo("ERROR: databento not installed. Run: pip install databento")
        sys.exit(1)

    key = api_key or os.environ.get("DATABENTO_API_KEY")
    if not key:
        # Try loading from .env
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("DATABENTO_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
                    break

    if not key:
        click.echo("ERROR: No API key found.")
        click.echo("  Set DATABENTO_API_KEY in .env or pass --api-key")
        click.echo("  Sign up at https://databento.com for $125 free credits")
        sys.exit(1)

    return db.Historical(key)


def estimate_cost(client, symbol: str, months: int) -> dict:
    """Estimate the data cost before downloading."""
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
    return {"estimated_cost_usd": float(cost), "start": start, "end": end}


def download_data(client, symbol: str, months: int, output_path: Path) -> pd.DataFrame:
    """Download OHLCV 1-second bars from Databento."""
    import time as _time

    end = datetime.utcnow() - timedelta(days=1)  # Historical only, exclude today
    start = end - timedelta(days=months * 30)

    click.echo(f"  Requesting {symbol} 1-second bars...")
    click.echo(f"  Date range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    click.echo(f"  This may take a few minutes for {months} months of data...")

    t0 = _time.time()
    click.echo(f"  [1/4] Streaming data from Databento API...", nl=False)
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
    df = data.to_df()
    t2 = _time.time()
    click.echo(f" done ({t2 - t1:.1f}s) -- {len(df):,} bars received")

    click.echo(f"  [3/4] Converting prices and cleaning...", nl=False)
    # to_df() defaults to price_type="float" (already converted from fixed-point)
    # and puts ts_event in the index as a DatetimeIndex
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
    t3 = _time.time()
    click.echo(f" done ({t3 - t2:.1f}s)")

    median_close = result["close"].median()
    click.echo(f"  Price range: {result['close'].min():.2f} - {result['close'].max():.2f}")
    click.echo(f"  Median close: {median_close:.2f}")
    click.echo(f"  Clean bars: {len(result):,}")

    if not (1000 < median_close < 10000):
        click.echo(f"  WARNING: Median close {median_close:.2f} outside expected MES range")

    click.echo(f"  [4/4] Saving CSV...", nl=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    t4 = _time.time()
    click.echo(f" done ({t4 - t3:.1f}s)")
    click.echo(f"  Saved {len(result):,} bars -> {output_path}")
    click.echo(f"  Total download time: {t4 - t0:.1f}s")

    return result


@click.command()
@click.option("--api-key", default=None, help="Databento API key (or set DATABENTO_API_KEY)")
@click.option("--symbol", default="MES", help="Futures symbol")
@click.option("--months", default=12, help="Months of history to download")
@click.option("--file", "file_path", default=None, help="Use existing CSV instead of downloading")
@click.option("--output", default="data", help="Output directory")
@click.option("--download-only", is_flag=True, help="Download data but don't run pipeline")
@click.option("--cost-only", is_flag=True, help="Just estimate cost, don't download")
@click.option("--target-bars", default=100, help="Target dollar bars per session")
@click.option("--capital", default=25000.0, help="Initial capital for backtest")
def main(api_key, symbol, months, file_path, output, download_only, cost_only, target_bars, capital):
    """Ingest MES data from Databento and run the full pipeline."""
    out_dir = Path(output)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{symbol}_databento.csv"

    # ---------------------------------------------------------------
    # Step 1: Get the data
    # ---------------------------------------------------------------
    if file_path:
        click.echo(f"\n[1] Loading existing data from {file_path}...")
        raw_df = pd.read_csv(file_path)
        click.echo(f"  Loaded {len(raw_df):,} bars")
    else:
        client = _get_client(api_key)

        if cost_only:
            click.echo(f"\nEstimating cost for {months} months of {symbol} 1s bars...")
            info = estimate_cost(client, symbol, months)
            click.echo(f"  Date range: {info['start'].strftime('%Y-%m-%d')} to {info['end'].strftime('%Y-%m-%d')}")
            click.echo(f"  Estimated cost: ${info['estimated_cost_usd']:.2f}")
            click.echo(f"\n  Your free credits: $125.00")
            remaining = 125.0 - info["estimated_cost_usd"]
            if remaining > 0:
                click.echo(f"  Remaining after download: ${remaining:.2f}")
            else:
                click.echo(f"  Additional cost: ${-remaining:.2f}")
            return

        click.echo(f"\n[1] Downloading {months} months of {symbol} from Databento...")
        raw_df = download_data(client, symbol, months, csv_path)

    if download_only:
        click.echo(f"\n  Data saved to {csv_path}")
        click.echo("  Run with --file to use this data in the pipeline")
        return

    # Ensure required columns
    if "vwap" not in raw_df.columns:
        raw_df["vwap"] = (raw_df["high"] + raw_df["low"] + raw_df["close"]) / 3
    if "bar_start" not in raw_df.columns:
        raw_df["bar_start"] = raw_df["timestamp"]

    # ---------------------------------------------------------------
    # Step 2: Build dollar bars (calibrate threshold)
    # ---------------------------------------------------------------
    click.echo(f"\n[2] Building dollar bars...")
    import numpy as np
    from scipy.stats import jarque_bera
    from bayesbot.data.bars import DollarBarBuilder
    from bayesbot.data.models import DollarBarConfig

    best_thresh = 2_000_000
    best_score = float("inf")
    for thresh in np.logspace(np.log10(500_000), np.log10(10_000_000), 10):
        builder = DollarBarBuilder(DollarBarConfig(base_threshold=thresh))
        test_df = builder.process_dataframe(raw_df)
        if len(test_df) < 20:
            continue
        ret = np.log(test_df["close"] / test_df["close"].shift(1)).dropna()
        if len(ret) < 10:
            continue
        test_df["date"] = pd.to_datetime(test_df["timestamp"], unit="s").dt.date
        avg_bars = test_df.groupby("date").size().mean()
        jb, _ = jarque_bera(ret)
        score = abs(avg_bars - target_bars) + jb * 0.01
        if score < best_score:
            best_score = score
            best_thresh = thresh

    click.echo(f"  Selected threshold: ${best_thresh:,.0f}")
    config = DollarBarConfig(base_threshold=best_thresh)
    builder = DollarBarBuilder(config)
    dollar_df = builder.process_dataframe(raw_df)

    # Add microstructure columns
    dollar_df["buy_volume"] = (dollar_df["volume"] * 0.55).astype(int)
    dollar_df["sell_volume"] = dollar_df["volume"] - dollar_df["buy_volume"]
    dollar_df["tick_count"] = np.random.randint(5, 50, len(dollar_df))
    dollar_df["symbol"] = symbol

    dollar_path = data_dir / f"{symbol}_dollar_bars.csv"
    dollar_df.to_csv(dollar_path, index=False)
    click.echo(f"  Built {len(dollar_df):,} dollar bars -> {dollar_path}")

    # ---------------------------------------------------------------
    # Step 3: Compute features + Train HMM
    # ---------------------------------------------------------------
    click.echo(f"\n[3] Computing features...")
    from bayesbot.features import get_feature_names
    from bayesbot.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(normalizer_min_samples=10)
    feature_df = pipeline.compute_features_batch(dollar_df)
    feature_names = get_feature_names()
    click.echo(f"  Computed {len(feature_names)} features x {len(dollar_df)} bars")

    click.echo(f"\n[4] Training HMM...")
    from bayesbot.regime.hmm import HMMTrainer
    from tabulate import tabulate

    matrix = np.column_stack([feature_df[f"norm_{fn}"].values for fn in feature_names])
    valid = np.any(matrix != 0, axis=1)
    matrix = matrix[valid]

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    trainer = HMMTrainer(n_restarts=5, max_iter=100)
    params, report = trainer.train(matrix, feature_names, n_states=3)

    model_path = str(model_dir / "hmm_params.json")
    HMMTrainer.save_parameters(params, model_path)

    rows = []
    for label, stats in report.state_statistics.items():
        rows.append({"State": label, "Bars": stats["n_bars"], "Fraction": f"{stats['fraction']:.1%}"})
    click.echo(tabulate(rows, headers="keys", tablefmt="simple"))
    click.echo(f"  BIC={report.best_params.metrics.get('bic', 0):.0f}, RCM={report.best_params.metrics.get('rcm', 0):.1f}")
    click.echo(f"  Model saved -> {model_path}")

    # ---------------------------------------------------------------
    # Step 4: Backtest
    # ---------------------------------------------------------------
    click.echo(f"\n[5] Running backtest...")
    from bayesbot.backtest.engine import BacktestEngine

    engine = BacktestEngine(initial_capital=capital, retrain_interval=500, min_train_bars=200)
    result = engine.run(dollar_df, initial_capital=capital)

    m = result.metrics
    perf_rows = [
        ["Total Return", f"{m.total_return_pct:.2f}%"],
        ["Sharpe Ratio", f"{m.sharpe_ratio:.3f}"],
        ["Sortino Ratio", f"{m.sortino_ratio:.3f}"],
        ["Max Drawdown", f"{m.max_drawdown_pct:.2f}%"],
        ["Total Trades", f"{m.total_trades}"],
        ["Win Rate", f"{m.win_rate:.1%}"],
        ["Profit Factor", f"{m.profit_factor:.2f}"],
    ]
    click.echo(tabulate(perf_rows, tablefmt="simple"))

    # Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "equity_curve.csv", result.equity_curve, delimiter=",", header="equity")
    if result.trades:
        trades_data = [{"pnl": t.pnl, "exit_reason": t.exit_reason, "strategy": t.strategy_name, "entry_regime": t.entry_regime} for t in result.trades]
        pd.DataFrame(trades_data).to_csv(out_dir / "trades.csv", index=False)

    # ---------------------------------------------------------------
    # Step 5: Walk-Forward Validation
    # ---------------------------------------------------------------
    click.echo(f"\n[6] Walk-forward validation...")
    import json
    from bayesbot.backtest.walk_forward import WalkForwardValidator

    n_bars = len(dollar_df)
    train_bars = min(int(n_bars * 0.5), 9000)
    test_bars = min(int(n_bars * 0.2), 3000)
    if train_bars + test_bars > n_bars:
        click.echo(f"  Not enough bars for walk-forward ({n_bars} available)")
        click.echo(f"  Running single-window backtest only")
    else:
        validator = WalkForwardValidator(
            train_bars=train_bars,
            test_bars=test_bars,
            initial_capital=capital,
        )
        wf_result = validator.run(dollar_df)

        click.echo(f"\n  Sharpe (mean +/- std): {wf_result.aggregate_sharpe_mean:.3f} +/- {wf_result.aggregate_sharpe_std:.3f}")
        click.echo(f"  Max Drawdown:        {wf_result.aggregate_max_dd:.1f}%")
        click.echo(f"  Avg Win Rate:        {wf_result.aggregate_win_rate:.1%}")

        click.echo(f"\n  GO / NO-GO:")
        for criterion, passed in wf_result.go_no_go.items():
            if criterion == "PASS":
                continue
            mark = "+" if passed else "x"
            click.echo(f"    [{mark}] {criterion}")

        overall = wf_result.go_no_go.get("PASS", False)
        if overall:
            click.echo(f"\n  VERDICT: GO -- ready for paper trading")
        else:
            click.echo(f"\n  VERDICT: NO-GO -- iterate on failing components")

        wf_report = {
            "sharpe_mean": wf_result.aggregate_sharpe_mean,
            "sharpe_std": wf_result.aggregate_sharpe_std,
            "max_dd": wf_result.aggregate_max_dd,
            "win_rate": wf_result.aggregate_win_rate,
            "go_no_go": wf_result.go_no_go,
        }
        (out_dir / "walk_forward_report.json").write_text(json.dumps(wf_report, indent=2))

    click.echo(f"\n  All results saved to {out_dir}/")
    click.echo("  Done!\n")


if __name__ == "__main__":
    main()
