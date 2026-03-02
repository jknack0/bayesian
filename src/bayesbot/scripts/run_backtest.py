"""CLI: Run a full backtest on historical dollar bars.

Usage:
    bayesbot backtest --file data/MES_dollar_bars.csv --capital 25000 --output results/
"""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tabulate import tabulate

from bayesbot.backtest.engine import BacktestEngine


@click.command("backtest")
@click.option("--file", "file_path", required=True, help="Dollar bars CSV")
@click.option("--capital", default=25000.0, help="Initial capital")
@click.option("--retrain-interval", default=500, help="Bars between HMM retraining")
@click.option("--output", default="results", help="Output directory")
def backtest(file_path: str, capital: float, retrain_interval: int, output: str):
    """Run event-driven backtest with periodic HMM retraining."""
    click.echo(f"\n  Loading {file_path}...")
    df = pd.read_csv(file_path)
    click.echo(f"  {len(df)} dollar bars loaded")

    engine = BacktestEngine(
        initial_capital=capital,
        retrain_interval=retrain_interval,
    )

    click.echo(f"  Running backtest (retrain every {retrain_interval} bars)...")
    result = engine.run(df, initial_capital=capital)

    m = result.metrics
    click.echo(f"\n{'='*60}")
    click.echo(f"  Backtest Results")
    click.echo(f"{'='*60}")

    rows = [
        ["Total Return", f"{m.total_return_pct:.2f}%"],
        ["Annualized Return", f"{m.annualized_return_pct:.2f}%"],
        ["Sharpe Ratio", f"{m.sharpe_ratio:.3f}"],
        ["Sortino Ratio", f"{m.sortino_ratio:.3f}"],
        ["Calmar Ratio", f"{m.calmar_ratio:.3f}"],
        ["Max Drawdown", f"{m.max_drawdown_pct:.2f}%"],
        ["Max DD Duration (bars)", f"{m.max_drawdown_duration_bars}"],
        ["Total Trades", f"{m.total_trades}"],
        ["Win Rate", f"{m.win_rate:.1%}"],
        ["Payoff Ratio", f"{m.payoff_ratio:.2f}"],
        ["Profit Factor", f"{m.profit_factor:.2f}"],
        ["Avg Trade PnL", f"${m.avg_trade_pnl:.2f}"],
        ["VaR (95%)", f"{m.var_95:.4f}"],
        ["CVaR (95%)", f"{m.cvar_95:.4f}"],
    ]
    click.echo(tabulate(rows, tablefmt="simple"))

    # Regime breakdown
    if m.regime_metrics:
        click.echo(f"\n  Per-Regime Breakdown:")
        regime_rows = []
        for regime, stats in m.regime_metrics.items():
            regime_rows.append({
                "Regime": regime,
                "Trades": stats["trades"],
                "PnL": f"${stats['total_pnl']:.2f}",
                "Win Rate": f"{stats['win_rate']:.1%}",
                "Avg PnL": f"${stats['avg_pnl']:.2f}",
            })
        click.echo(tabulate(regime_rows, headers="keys", tablefmt="simple"))

    # Save outputs
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_dir / "equity_curve.csv", result.equity_curve, delimiter=",", header="equity")

    if result.trades:
        trades_data = [
            {
                "id": t.id,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "strategy": t.strategy_name,
                "entry_regime": t.entry_regime,
                "holding_bars": t.holding_bars,
            }
            for t in result.trades
        ]
        pd.DataFrame(trades_data).to_csv(out_dir / "trades.csv", index=False)

    if result.regime_history:
        regime_data = [
            {
                "bar_index": r.bar_index,
                "regime": r.regime_name,
                "confidence": r.confidence,
            }
            for r in result.regime_history
        ]
        pd.DataFrame(regime_data).to_csv(out_dir / "regime_history.csv", index=False)

    # JSON report
    report = {
        "total_return_pct": m.total_return_pct,
        "sharpe_ratio": m.sharpe_ratio,
        "sortino_ratio": m.sortino_ratio,
        "max_drawdown_pct": m.max_drawdown_pct,
        "total_trades": m.total_trades,
        "win_rate": m.win_rate,
        "profit_factor": m.profit_factor,
        "regime_metrics": m.regime_metrics,
    }
    (out_dir / "performance_report.json").write_text(json.dumps(report, indent=2))

    click.echo(f"\n  Results saved to {out_dir}/")
