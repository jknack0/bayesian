"""Unified CLI entry point for bayesbot."""

import click


@click.group()
def cli():
    """BayesBot — Bayesian semi-supervised trading bot."""
    pass


# Import and register subcommands
from bayesbot.scripts.import_data import import_data
from bayesbot.scripts.calibrate_bars import calibrate
from bayesbot.scripts.train_hmm import train_hmm
from bayesbot.scripts.run_backtest import backtest
from bayesbot.scripts.validate import validate
from bayesbot.scripts.run_live import live

cli.add_command(import_data, "import")
cli.add_command(calibrate, "calibrate")
cli.add_command(train_hmm, "train")
cli.add_command(backtest, "backtest")
cli.add_command(validate, "validate")
cli.add_command(live, "live")


if __name__ == "__main__":
    cli()
