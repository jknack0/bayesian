#!/usr/bin/env python
"""Full end-to-end pipeline: generate data -> import -> calibrate -> train -> backtest -> validate.

This is the single script you run to test the entire system:

    python scripts/full_pipeline.py

It will:
1. Generate synthetic MES data (or use --file for real data)
2. Build dollar bars with calibrated threshold
3. Compute features and train HMM
4. Run backtest with periodic retraining
5. Run walk-forward validation
6. Print GO/NO-GO verdict
"""

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger
from tabulate import tabulate


@click.command()
@click.option("--file", "file_path", default=None, help="CSV of raw 1s bars (generates synthetic if omitted)")
@click.option("--databento", "use_databento", is_flag=True, help="Download data from Databento API")
@click.option("--databento-key", default=None, help="Databento API key (or set DATABENTO_API_KEY)")
@click.option("--months", default=12, help="Months of Databento history to download")
@click.option("--symbol", default="MES")
@click.option("--days", default=90, help="Synthetic data days (if --file not given)")
@click.option("--target-bars", default=100, help="Target dollar bars per session")
@click.option("--capital", default=25000.0)
@click.option("--output", default="results")
def main(file_path, use_databento, databento_key, months, symbol, days, target_bars, capital, output):
    """Run the complete BayesBot pipeline end-to-end."""
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1: Data
    # ---------------------------------------------------------------
    if use_databento:
        click.echo(f"\n[1/6] Downloading {months} months of {symbol} from Databento...")
        from databento_utils import download_databento_data
        raw_df = download_databento_data(symbol, months, databento_key, data_dir)
        click.echo(f"  Downloaded {len(raw_df):,} rows")
    elif file_path is not None:
        click.echo(f"\n[1/6] Loading data from {file_path}...")
        raw_df = pd.read_csv(file_path)
        if "vwap" not in raw_df.columns:
            raw_df["vwap"] = (raw_df["high"] + raw_df["low"] + raw_df["close"]) / 3
        if "bar_start" not in raw_df.columns:
            raw_df["bar_start"] = raw_df["timestamp"]
        click.echo(f"  Loaded {len(raw_df):,} rows")
    else:
        click.echo("\n[1/6] Generating synthetic data...")
        from generate_synthetic_data import generate_synthetic_mes
        raw_df = generate_synthetic_mes(days)
        raw_path = data_dir / f"{symbol}_synthetic.csv"
        raw_df.to_csv(raw_path, index=False)
        click.echo(f"  Generated {len(raw_df):,} rows -> {raw_path}")

    # ---------------------------------------------------------------
    # Step 2: Build dollar bars (quick calibration)
    # ---------------------------------------------------------------
    click.echo("\n[2/6] Building dollar bars...")
    from bayesbot.data.bars import DollarBarBuilder
    from bayesbot.data.models import DollarBarConfig

    # Quick threshold search
    from scipy.stats import jarque_bera

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
    click.echo("\n[3/6] Computing features...")
    from bayesbot.features import get_feature_names
    from bayesbot.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(normalizer_min_samples=10)
    feature_df = pipeline.compute_features_batch(dollar_df)
    feature_names = get_feature_names()
    click.echo(f"  Computed {len(feature_names)} features x {len(dollar_df)} bars")

    click.echo("\n[4/6] Training HMM...")
    from bayesbot.regime.hmm import HMMTrainer

    matrix = np.column_stack([feature_df[f"norm_{fn}"].values for fn in feature_names])
    valid = np.any(matrix != 0, axis=1)
    matrix = matrix[valid]

    trainer = HMMTrainer(n_restarts=5, max_iter=100)
    params, report = trainer.train(matrix, feature_names, n_states=3)

    model_path = str(model_dir / "hmm_params.json")
    HMMTrainer.save_parameters(params, model_path)

    # Display state stats
    rows = []
    for label, stats in report.state_statistics.items():
        rows.append({"State": label, "Bars": stats["n_bars"], "Fraction": f"{stats['fraction']:.1%}"})
    click.echo(tabulate(rows, headers="keys", tablefmt="simple"))
    click.echo(f"  BIC={report.best_params.metrics.get('bic', 0):.0f}, RCM={report.best_params.metrics.get('rcm', 0):.1f}")
    click.echo(f"  Model saved -> {model_path}")

    # ---------------------------------------------------------------
    # Step 4: Backtest
    # ---------------------------------------------------------------
    click.echo("\n[5/6] Running backtest...")
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

    # Save equity curve
    np.savetxt(out_dir / "equity_curve.csv", result.equity_curve, delimiter=",", header="equity")
    if result.trades:
        trades_data = [{"pnl": t.pnl, "exit_reason": t.exit_reason, "strategy": t.strategy_name, "entry_regime": t.entry_regime} for t in result.trades]
        pd.DataFrame(trades_data).to_csv(out_dir / "trades.csv", index=False)

    # ---------------------------------------------------------------
    # Step 5: Walk-Forward Validation
    # ---------------------------------------------------------------
    click.echo("\n[6/6] Walk-forward validation...")
    from bayesbot.backtest.walk_forward import WalkForwardValidator

    # Adjust window sizes based on available data
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

        # Save report
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
