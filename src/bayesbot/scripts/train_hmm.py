"""CLI: Train HMM regime detector on historical dollar bars.

Usage:
    bayesbot train --file data/MES_dollar_bars.csv --output models/hmm_params.json
"""

import click
import numpy as np
import pandas as pd
from loguru import logger
from tabulate import tabulate

from bayesbot.features import get_feature_names
from bayesbot.features.pipeline import FeaturePipeline
from bayesbot.regime.hmm import HMMTrainer


@click.command("train")
@click.option("--file", "file_path", required=True, help="Dollar bars CSV")
@click.option("--states", default=3, help="Number of HMM states")
@click.option("--restarts", default=10, help="Number of random restarts")
@click.option("--output", default="models/hmm_params.json", help="Output path")
def train_hmm(file_path: str, states: int, restarts: int, output: str):
    """Train HMM regime detector."""
    click.echo(f"\n  Loading dollar bars from {file_path}...")
    df = pd.read_csv(file_path)
    click.echo(f"  Loaded {len(df)} bars")

    # Compute features
    click.echo("  Computing features...")
    pipeline = FeaturePipeline()
    feature_df = pipeline.compute_features_batch(df)
    feature_names = get_feature_names()

    # Build normalised feature matrix
    matrix = np.column_stack([feature_df[f"norm_{fn}"].values for fn in feature_names])

    # Remove warm-up rows (all zeros)
    valid = np.any(matrix != 0, axis=1)
    matrix = matrix[valid]
    click.echo(f"  Training matrix: {matrix.shape[0]} samples × {matrix.shape[1]} features")

    # Train
    click.echo(f"  Training {states}-state HMM with {restarts} restarts...")
    trainer = HMMTrainer(n_restarts=restarts)
    params, report = trainer.train(matrix, feature_names, n_states=states)

    # Display results
    click.echo(f"\n{'='*60}")
    click.echo(f"  Training Report")
    click.echo(f"{'='*60}")

    # State statistics
    rows = []
    for label, stats in report.state_statistics.items():
        rows.append({
            "State": label,
            "Bars": stats["n_bars"],
            "Fraction": f"{stats['fraction']:.1%}",
            "Mean Return": f"{stats['mean_return']:.6f}",
            "Std Return": f"{stats['std_return']:.6f}",
        })
    click.echo(tabulate(rows, headers="keys", tablefmt="simple"))

    # Model comparison
    click.echo(f"\n  Model Comparison:")
    comp_rows = []
    for ns, m in report.model_comparison.items():
        if "error" not in m:
            comp_rows.append({
                "States": ns,
                "BIC": f"{m['bic']:.0f}",
                "AIC": f"{m['aic']:.0f}",
                "RCM": f"{m.get('rcm', 0):.1f}",
                "LogL": f"{m['log_likelihood']:.0f}",
            })
    click.echo(tabulate(comp_rows, headers="keys", tablefmt="simple"))

    # Transition matrix
    click.echo(f"\n  Transition Matrix:")
    for i, label in enumerate(params.state_labels):
        probs = " ".join(f"{p:.3f}" for p in params.transition_matrix[i])
        click.echo(f"    {label:20s} → [{probs}]")

    # Save
    HMMTrainer.save_parameters(params, output)
    click.echo(f"\n  Model saved to {output}")
