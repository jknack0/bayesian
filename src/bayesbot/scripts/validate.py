"""CLI: Walk-forward validation — the GO/NO-GO gate.

Usage:
    bayesbot validate --file data/MES_dollar_bars.csv
"""

import click
import pandas as pd
from tabulate import tabulate

from bayesbot.backtest.walk_forward import WalkForwardValidator


@click.command("validate")
@click.option("--file", "file_path", required=True, help="Dollar bars CSV")
@click.option("--train-bars", default=9000, help="Training window size")
@click.option("--test-bars", default=3000, help="Test window size")
@click.option("--capital", default=25000.0, help="Initial capital per window")
def validate(file_path: str, train_bars: int, test_bars: int, capital: float):
    """Walk-forward validation with GO/NO-GO criteria."""
    click.echo(f"\n  Loading {file_path}...")
    df = pd.read_csv(file_path)
    click.echo(f"  {len(df)} bars loaded")
    click.echo(f"  Windows: train={train_bars}, test={test_bars}")

    validator = WalkForwardValidator(
        train_bars=train_bars,
        test_bars=test_bars,
        initial_capital=capital,
    )

    click.echo("  Running walk-forward analysis...")
    result = validator.run(df)

    click.echo(f"\n{'='*60}")
    click.echo(f"  Walk-Forward Validation Results")
    click.echo(f"{'='*60}")

    # Per-window results
    if result.window_metrics:
        rows = []
        for i, m in enumerate(result.window_metrics):
            rows.append({
                "Window": i,
                "Sharpe": f"{m.sharpe_ratio:.3f}",
                "Return": f"{m.total_return_pct:.2f}%",
                "MaxDD": f"{m.max_drawdown_pct:.1f}%",
                "Trades": m.total_trades,
                "Win Rate": f"{m.win_rate:.1%}" if m.total_trades > 0 else "N/A",
            })
        click.echo(tabulate(rows, headers="keys", tablefmt="simple"))

    click.echo(f"\n  Aggregate:")
    click.echo(f"    Sharpe (mean ± std): {result.aggregate_sharpe_mean:.3f} ± {result.aggregate_sharpe_std:.3f}")
    click.echo(f"    Max Drawdown:        {result.aggregate_max_dd:.1f}%")
    click.echo(f"    Avg Win Rate:        {result.aggregate_win_rate:.1%}")

    # GO/NO-GO
    click.echo(f"\n  GO / NO-GO Criteria:")
    for criterion, passed in result.go_no_go.items():
        if criterion == "PASS":
            continue
        status = "PASS" if passed else "FAIL"
        mark = "✓" if passed else "✗"
        click.echo(f"    {mark} {criterion}: {status}")

    overall = result.go_no_go.get("PASS", False)
    if overall:
        click.echo(f"\n  ★ VERDICT: GO — ready for paper trading")
    else:
        click.echo(f"\n  ✗ VERDICT: NO-GO — iterate on failing components")
