"""Walk-forward validation — the gold standard for systematic strategy testing.

Slides a train/test window across history:
  Train on 90 days → Test on 30 days → Slide forward 30 days → Repeat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from bayesbot.backtest.engine import BacktestEngine, BacktestResult
from bayesbot.backtest.metrics import PerformanceMetrics


@dataclass
class WalkForwardResult:
    window_results: list[BacktestResult]
    window_metrics: list[PerformanceMetrics]
    aggregate_sharpe_mean: float
    aggregate_sharpe_std: float
    aggregate_max_dd: float
    aggregate_win_rate: float
    go_no_go: dict  # per-criterion pass/fail


class WalkForwardValidator:
    """Walk-forward analysis with configurable window sizes.

    GO/NO-GO criteria:
    - Average OOS Sharpe ≥ 0.3
    - Std of Sharpe < mean of Sharpe (consistency)
    - Max drawdown in any window < 25%
    - Win rate across windows > 50%
    """

    def __init__(
        self,
        train_bars: int = 9000,   # ~90 sessions × 100 bars
        test_bars: int = 3000,    # ~30 sessions × 100 bars
        step_bars: int = 3000,
        initial_capital: float = 25_000.0,
    ):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.initial_capital = initial_capital

    def run(self, bars_df: pd.DataFrame) -> WalkForwardResult:
        """Execute the walk-forward analysis."""
        total = len(bars_df)
        min_required = self.train_bars + self.test_bars
        if total < min_required:
            logger.warning(
                "Insufficient data ({} bars, need {}). Running single window.",
                total, min_required,
            )

        window_results: list[BacktestResult] = []
        window_metrics: list[PerformanceMetrics] = []

        start = 0
        window_num = 0
        while start + self.train_bars + self.test_bars <= total:
            train_end = start + self.train_bars
            test_end = train_end + self.test_bars

            test_df = bars_df.iloc[train_end:test_end].reset_index(drop=True)

            logger.info(
                "Walk-forward window {} — train [{}, {}), test [{}, {})",
                window_num, start, train_end, train_end, test_end,
            )

            # Pre-train HMM on training window
            from bayesbot.features import get_feature_names
            from bayesbot.features.pipeline import FeaturePipeline
            from bayesbot.regime.hmm import HMMTrainer

            train_df = bars_df.iloc[start:train_end].reset_index(drop=True)
            pipeline = FeaturePipeline()
            train_features = pipeline.compute_features_batch(train_df)
            feature_names = get_feature_names()

            train_matrix = np.column_stack(
                [train_features[f"norm_{fn}"].values for fn in feature_names]
            )
            valid = np.any(train_matrix != 0, axis=1)
            train_matrix = train_matrix[valid]

            pretrained_params = None
            if len(train_matrix) >= 200:
                trainer = HMMTrainer(n_restarts=20, max_iter=300, covariance_type="diag")
                try:
                    pretrained_params, _ = trainer.train(train_matrix, feature_names, n_states=3)
                except Exception as e:
                    logger.warning("Pre-training failed for window {}: {}", window_num, e)

            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                retrain_interval=self.test_bars + 1,  # don't retrain during test
                min_train_bars=self.train_bars,  # effectively prevents retraining on small samples
            )

            result = engine.run(
                test_df,
                initial_capital=self.initial_capital,
                pretrained_params=pretrained_params,
            )
            window_results.append(result)
            window_metrics.append(result.metrics)

            start += self.step_bars
            window_num += 1

        if not window_metrics:
            return WalkForwardResult(
                window_results=[],
                window_metrics=[],
                aggregate_sharpe_mean=0.0,
                aggregate_sharpe_std=0.0,
                aggregate_max_dd=0.0,
                aggregate_win_rate=0.0,
                go_no_go={"error": "No windows could be evaluated"},
            )

        sharpes = [m.sharpe_ratio for m in window_metrics]
        max_dds = [m.max_drawdown_pct for m in window_metrics]
        win_rates = [m.win_rate for m in window_metrics]

        sharpe_mean = float(np.mean(sharpes))
        sharpe_std = float(np.std(sharpes))
        max_dd = float(np.max(max_dds)) if max_dds else 0.0
        avg_wr = float(np.mean(win_rates)) if win_rates else 0.0

        go_no_go = {
            "sharpe_mean_ge_0.3": sharpe_mean >= 0.3,
            "sharpe_std_lt_mean": sharpe_std < sharpe_mean if sharpe_mean > 0 else False,
            "max_dd_lt_25": max_dd < 25.0,
            "win_rate_gt_50": avg_wr > 0.50,
        }
        go_no_go["PASS"] = all(
            v for k, v in go_no_go.items() if k != "PASS"
        )

        logger.info(
            "Walk-forward complete — {} windows, Sharpe={:.2f}±{:.2f}, MaxDD={:.1f}%, GO={}",
            len(window_metrics), sharpe_mean, sharpe_std, max_dd, go_no_go["PASS"],
        )

        return WalkForwardResult(
            window_results=window_results,
            window_metrics=window_metrics,
            aggregate_sharpe_mean=sharpe_mean,
            aggregate_sharpe_std=sharpe_std,
            aggregate_max_dd=max_dd,
            aggregate_win_rate=avg_wr,
            go_no_go=go_no_go,
        )
