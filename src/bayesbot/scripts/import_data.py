"""CLI: Import historical CSV data and build dollar bars.

Usage:
    bayesbot import --file data/MES_1s.csv --symbol MES --format generic
    bayesbot import --file data/MES_databento.csv --symbol MES --format databento --threshold 2000000
"""

import click
from loguru import logger

from bayesbot.data.importers import CSVImporter
from bayesbot.data.models import DollarBarConfig


@click.command("import")
@click.option("--file", "file_path", required=True, help="Path to CSV file")
@click.option("--symbol", required=True, help="Symbol (e.g., MES)")
@click.option(
    "--format", "fmt",
    default="generic",
    type=click.Choice(["databento", "ninjatrader", "generic"]),
)
@click.option("--threshold", default=None, type=float, help="Dollar bar threshold (auto if omitted)")
@click.option("--output", default=None, help="Output path for dollar bars CSV")
def import_data(file_path: str, symbol: str, fmt: str, threshold: float | None, output: str | None):
    """Import historical data and build dollar bars."""
    config = DollarBarConfig()
    if threshold:
        config.base_threshold = threshold

    importer = CSVImporter(bar_config=config)
    raw_df, dollar_df, report = importer.import_csv(file_path, symbol, fmt=fmt, bar_config=config)

    click.echo(f"\n{'='*60}")
    click.echo(f"  Import Report: {report.file_path}")
    click.echo(f"{'='*60}")
    click.echo(f"  Symbol:              {report.symbol}")
    click.echo(f"  Raw rows imported:   {report.rows_imported:,}")
    click.echo(f"  Dollar bars created: {report.dollar_bars_generated:,}")
    click.echo(f"  Date range:          {report.date_range[0]}")
    click.echo(f"                    → {report.date_range[1]}")
    click.echo(f"  Avg bars/session:    {report.avg_bars_per_session:.1f}")
    click.echo(f"  Elapsed:             {report.elapsed_seconds:.1f}s")
    if report.warnings:
        click.echo(f"\n  Warnings:")
        for w in report.warnings:
            click.echo(f"    ⚠ {w}")
    click.echo()

    # Save outputs
    out_dir = output or "data"
    raw_path = f"{out_dir}/{symbol}_raw.csv"
    dollar_path = f"{out_dir}/{symbol}_dollar_bars.csv"

    raw_df.to_csv(raw_path, index=False)
    dollar_df.to_csv(dollar_path, index=False)
    click.echo(f"  Saved: {raw_path}")
    click.echo(f"  Saved: {dollar_path}")
