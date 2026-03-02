"""Event-driven backtesting engine.

Processes one bar at a time — exactly as the live system will.
Periodically retrains the HMM to avoid using future information.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from bayesbot.backtest.metrics import PerformanceMetrics, compute_metrics
from bayesbot.backtest.slippage import SlippageModel
from bayesbot.data.models import CompletedTrade, Position, RegimePrediction, TradeSignal
from bayesbot.features import get_feature_names
from bayesbot.features.pipeline import FeaturePipeline
from bayesbot.regime.detector import RegimeDetector
from bayesbot.regime.hmm import HMMParameters, HMMTrainer
from bayesbot.risk.cppi import CPPIPositionSizer
from bayesbot.risk.drawdown_brake import DrawdownBrake
from bayesbot.risk.kelly import KellyCalculator
from bayesbot.risk.regime_scaler import RegimeRiskScaler
from bayesbot.strategy.base import StrategyContext
from bayesbot.strategy.selector import StrategySelector


@dataclass
class BacktestResult:
    equity_curve: np.ndarray
    trades: list[CompletedTrade]
    regime_history: list[RegimePrediction]
    metrics: PerformanceMetrics
    bars_df: pd.DataFrame = field(default_factory=pd.DataFrame)


class BacktestEngine:
    """Event-driven backtester with periodic HMM retraining."""

    def __init__(
        self,
        initial_capital: float = 25_000.0,
        point_value: float = 5.0,
        retrain_interval: int = 500,
        min_train_bars: int = 300,
    ):
        self.initial_capital = initial_capital
        self.point_value = point_value
        self.retrain_interval = retrain_interval
        self.min_train_bars = min_train_bars

        # Components
        self.feature_pipeline = FeaturePipeline()
        self.strategy_selector = StrategySelector()
        self.kelly = KellyCalculator()
        self.sizer = CPPIPositionSizer()
        self.regime_scaler = RegimeRiskScaler()
        self.brake = DrawdownBrake(initial_capital=initial_capital)
        self.slippage_model = SlippageModel(point_value=point_value)
        self.trainer = HMMTrainer(n_restarts=20, max_iter=300, covariance_type="diag")

    def run(
        self,
        bars_df: pd.DataFrame,
        initial_capital: float | None = None,
        pretrained_params: HMMParameters | None = None,
    ) -> BacktestResult:
        """Run the full event-driven backtest.

        If pretrained_params is provided, the engine starts with that model
        instead of waiting to accumulate min_train_bars.
        """
        capital = initial_capital or self.initial_capital
        self.sizer.initialize(capital)

        equity = capital
        equity_curve: list[float] = [equity]
        trades: list[CompletedTrade] = []
        regime_history: list[RegimePrediction] = []
        positions: list[Position] = []
        daily_pnl = 0.0
        current_day: str = ""
        last_exit_bar = -10  # cooldown: bars since last exit

        # Feature pipeline needs full DataFrame for batch computation
        logger.info("Computing features on {} bars...", len(bars_df))
        feature_df = self.feature_pipeline.compute_features_batch(bars_df)
        feature_names = get_feature_names()

        # Use pre-trained model if provided, otherwise train from scratch
        if pretrained_params is not None:
            hmm_params = pretrained_params
            detector = RegimeDetector(hmm_params)
            last_train_idx = 0
            logger.info("Using pre-trained HMM (skipping initial training)")
        else:
            hmm_params: HMMParameters | None = None
            detector: RegimeDetector | None = None
            last_train_idx = 0

        for i in range(len(bars_df)):
            bar = bars_df.iloc[i]
            bar_dict = bar.to_dict()
            bar_idx = int(bar.get("bar_index", i))

            # --- Periodic HMM (re)training ---
            should_train = (
                hmm_params is None and i >= self.min_train_bars
            ) or (
                hmm_params is not None
                and (i - last_train_idx) >= self.retrain_interval
                and i >= self.min_train_bars
            )

            if should_train:
                train_matrix = np.column_stack(
                    [feature_df[f"norm_{fn}"].values[:i] for fn in feature_names]
                )
                # Remove rows that are all zero (warm-up period)
                valid_mask = np.any(train_matrix != 0, axis=1)
                train_matrix = train_matrix[valid_mask]

                if len(train_matrix) >= self.min_train_bars:
                    try:
                        hmm_params, _report = self.trainer.train(
                            train_matrix, feature_names, n_states=3
                        )
                        detector = RegimeDetector(hmm_params)
                        last_train_idx = i
                        logger.info("HMM trained at bar {} ({} samples)", i, len(train_matrix))
                    except Exception as e:
                        logger.warning("HMM training failed at bar {}: {}", i, e)

            # --- Skip bars without a trained model ---
            if detector is None:
                equity_curve.append(equity)
                continue

            # --- Build feature vector for this bar ---
            obs = np.array(
                [feature_df.iloc[i][f"norm_{fn}"] for fn in feature_names]
            )

            from bayesbot.data.models import FeatureVector

            fv = FeatureVector(
                timestamp=float(bar.get("timestamp", 0)),
                bar_index=bar_idx,
                symbol=str(bar.get("symbol", "MES")),
                features={fn: float(feature_df.iloc[i][f"raw_{fn}"]) for fn in self.feature_pipeline.all_feature_names},
                normalized_features={fn: float(obs[j]) for j, fn in enumerate(feature_names)},
            )

            # --- Regime prediction ---
            regime = detector.predict(fv)
            regime_history.append(regime)

            # --- Update positions ---
            for pos in positions:
                if pos.direction == "LONG":
                    pos.current_price = float(bar["close"])
                    pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity * self.point_value
                else:
                    pos.current_price = float(bar["close"])
                    pos.unrealized_pnl = (pos.entry_price - pos.current_price) * pos.quantity * self.point_value
                pos.max_favorable_excursion = max(pos.max_favorable_excursion, pos.unrealized_pnl)
                pos.max_adverse_excursion = min(pos.max_adverse_excursion, pos.unrealized_pnl)

            # --- ATR ---
            atr = float(feature_df.iloc[i].get("raw_atr_14", 1.0))
            if atr <= 0:
                atr = 1.0

            # --- Daily PnL reset ---
            ts = float(bar.get("timestamp", bar.get("bar_end", 0)))
            if ts > 0:
                from datetime import datetime, timezone
                bar_day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                if bar_day != current_day:
                    current_day = bar_day
                    daily_pnl = 0.0

            # --- Check drawdown brake ---
            brake_status = self.brake.check(equity, daily_pnl, bar_index=i)

            # --- Manage existing positions ---
            lookback_start = max(0, i - self.feature_pipeline.required_lookback)
            recent_bars = bars_df.iloc[lookback_start : i + 1]

            ctx = StrategyContext(
                current_bar=bar_dict,
                recent_bars=recent_bars,
                features=fv,
                regime=regime,
                existing_positions=positions,
                account_equity=equity,
                daily_pnl=daily_pnl,
                atr=atr,
            )

            if brake_status.kill_switch:
                # Flatten everything
                for pos in list(positions):
                    trade = self._close_position(pos, bar_dict, regime, "KILL_SWITCH")
                    trades.append(trade)
                    equity += trade.pnl - trade.commission - trade.slippage
                    daily_pnl += trade.pnl - trade.commission - trade.slippage
                    last_exit_bar = i
                positions.clear()
            else:
                mgmt_actions = self.strategy_selector.manage_positions(
                    positions, ctx, bar_idx
                )
                for pos_id, mgmt in mgmt_actions.items():
                    pos = next((p for p in positions if p.id == pos_id), None)
                    if pos is None:
                        continue
                    if mgmt.action == "EXIT":
                        trade = self._close_position(
                            pos, bar_dict, regime, mgmt.exit_reason or "STRATEGY"
                        )
                        trades.append(trade)
                        equity += trade.pnl - trade.commission - trade.slippage
                        daily_pnl += trade.pnl - trade.commission - trade.slippage
                        positions = [p for p in positions if p.id != pos_id]
                        last_exit_bar = i
                    elif mgmt.action == "ADJUST_STOP" and mgmt.new_stop_loss is not None:
                        pos.stop_loss = mgmt.new_stop_loss
                    elif mgmt.action == "ADJUST_TARGET" and mgmt.new_profit_target is not None:
                        pos.profit_target = mgmt.new_profit_target

            # --- Generate new signals ---
            # Always call select_signal to keep strategy state updated
            # (e.g. bar-gap filters need _last_bar_index current)
            active_strategies = {p.strategy_name for p in positions}
            signal = self.strategy_selector.select_signal(ctx)
            can_enter = (
                brake_status.allow_new_entries
                and signal is not None
                and signal.strategy_name not in active_strategies
            )
            if can_enter:
                    kelly_f = self.kelly.compute(trades)
                    regime_scale = self.regime_scaler.compute_scale(regime)
                    qty = self.sizer.calculate_position_size(
                        signal, equity, atr, regime, kelly_f,
                        regime_scale, brake_status.scale, self.point_value,
                    )
                    if qty > 0:
                        pos = Position(
                            id=str(uuid.uuid4()),
                            symbol=signal.symbol,
                            direction=signal.direction,
                            entry_price=signal.entry_price,
                            current_price=signal.entry_price,
                            quantity=qty,
                            entry_time=signal.timestamp,
                            entry_bar_index=bar_idx,
                            stop_loss=signal.stop_loss,
                            profit_target=signal.profit_target,
                            time_barrier=bar_idx + signal.time_barrier_bars,
                            strategy_name=signal.strategy_name,
                            entry_regime=regime.regime_name,
                        )
                        positions.append(pos)

            equity_curve.append(equity + sum(p.unrealized_pnl for p in positions))

        # Close any remaining positions at last bar
        if positions:
            last_bar = bars_df.iloc[-1].to_dict()
            last_regime = regime_history[-1] if regime_history else RegimePrediction(
                0, 0, 0, "unknown", [0.33, 0.33, 0.34], 0.34
            )
            for pos in positions:
                trade = self._close_position(pos, last_bar, last_regime, "BACKTEST_END")
                trades.append(trade)
                equity += trade.pnl - trade.commission - trade.slippage
            positions.clear()
            equity_curve[-1] = equity

        eq_arr = np.array(equity_curve)
        metrics = compute_metrics(eq_arr, trades)

        logger.info(
            "Backtest complete — {} trades, Sharpe={:.2f}, MaxDD={:.1f}%, Return={:.1f}%",
            metrics.total_trades,
            metrics.sharpe_ratio,
            metrics.max_drawdown_pct,
            metrics.total_return_pct,
        )

        return BacktestResult(
            equity_curve=eq_arr,
            trades=trades,
            regime_history=regime_history,
            metrics=metrics,
            bars_df=bars_df,
        )

    def _close_position(
        self,
        pos: Position,
        bar: dict,
        regime: RegimePrediction,
        reason: str,
    ) -> CompletedTrade:
        exit_price = float(bar.get("close", pos.current_price))
        if pos.direction == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.quantity * self.point_value
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity * self.point_value

        vol = int(bar.get("volume", 1000))
        rng = float(bar.get("high", exit_price)) - float(bar.get("low", exit_price))
        slip = self.slippage_model.estimate_slippage(vol, rng, pos.quantity)
        comm = self.slippage_model.commission * pos.quantity * 2

        return CompletedTrade(
            id=pos.id,
            symbol=pos.symbol,
            direction=pos.direction,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=float(bar.get("timestamp", 0)),
            entry_regime=pos.entry_regime,
            exit_regime=regime.regime_name,
            pnl=pnl,
            commission=comm,
            slippage=slip,
            exit_reason=reason,
            strategy_name=pos.strategy_name,
            holding_bars=int(bar.get("bar_index", 0)) - pos.entry_bar_index,
        )
