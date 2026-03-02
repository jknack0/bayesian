"""Main live trading loop — orchestrates all components.

On each new dollar bar:
  features → regime → brake check → manage positions → signals → size → execute
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import numpy as np
import pandas as pd
from loguru import logger

from bayesbot.config import Settings
from bayesbot.data.bars import DollarBarBuilder
from bayesbot.data.models import (
    CompletedTrade,
    DollarBarConfig,
    FeatureVector,
    Position,
    RawTick,
    RegimePrediction,
)
from bayesbot.execution.broker import BaseBroker
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


class LiveTradingLoop:
    """Event-driven live trading loop."""

    def __init__(
        self,
        broker: BaseBroker,
        hmm_params: HMMParameters,
        settings: Settings | None = None,
        bar_config: DollarBarConfig | None = None,
    ):
        self.broker = broker
        self.settings = settings or Settings()

        self.bar_builder = DollarBarBuilder(bar_config or DollarBarConfig())
        self.feature_pipeline = FeaturePipeline()
        self.detector = RegimeDetector(hmm_params)
        self.strategy_selector = StrategySelector()
        self.kelly = KellyCalculator()
        self.sizer = CPPIPositionSizer()
        self.regime_scaler = RegimeRiskScaler()
        self.brake = DrawdownBrake(initial_capital=self.settings.initial_capital)

        self.sizer.initialize(self.settings.initial_capital)

        # State
        self._positions: list[Position] = []
        self._trades: list[CompletedTrade] = []
        self._bar_history: list[dict] = []
        self._equity: float = self.settings.initial_capital
        self._daily_pnl: float = 0.0
        self._running = False
        self._last_checkpoint: float = 0.0
        self._last_order_time: float = 0.0

    async def start(self) -> None:
        """Connect to broker and begin the event loop."""
        await self.broker.connect()
        self._running = True

        logger.info("Live trading loop started for {}", self.settings.symbol)

        # Subscribe to market data
        await self.broker.subscribe_ticks(self.settings.symbol, self._on_tick)

        # Keep running
        while self._running:
            await asyncio.sleep(1.0)

            # Periodic checkpoint
            if time.time() - self._last_checkpoint > 300:
                self._save_checkpoint()

    async def stop(self) -> None:
        self._running = False
        self._save_checkpoint()
        await self.broker.disconnect()
        logger.info("Live trading loop stopped")

    def _on_tick(self, tickers) -> None:
        """Callback from broker on each tick.  Feeds to bar builder."""
        for ticker in tickers:
            if ticker.last is None or ticker.lastSize is None:
                continue
            tick = RawTick(
                timestamp=time.time(),
                price=ticker.last,
                volume=int(ticker.lastSize),
                side="unknown",
            )
            bar = self.bar_builder.process_tick(tick)
            if bar is not None:
                asyncio.get_event_loop().create_task(self._on_bar(bar.__dict__))

    async def _on_bar(self, bar: dict) -> None:
        """Process a completed dollar bar — main decision loop."""
        self._bar_history.append(bar)

        # Keep only the lookback we need
        max_history = self.feature_pipeline.required_lookback + 10
        if len(self._bar_history) > max_history:
            self._bar_history = self._bar_history[-max_history:]

        recent_df = pd.DataFrame(self._bar_history)
        fv = self.feature_pipeline.compute_features_single(recent_df)
        if fv is None:
            return

        # Regime
        regime = self.detector.predict(fv)
        logger.info("Bar {} — regime={} (p={:.2f})",
                     bar.get("bar_index"), regime.regime_name, regime.confidence)

        # Update equity from broker
        acct = await self.broker.get_account_summary()
        self._equity = acct.get("equity", self._equity)

        # Update position prices
        for pos in self._positions:
            pos.current_price = float(bar.get("close", pos.current_price))
            if pos.direction == "LONG":
                pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity * self.settings.point_value
            else:
                pos.unrealized_pnl = (pos.entry_price - pos.current_price) * pos.quantity * self.settings.point_value

        atr = float(fv.features.get("atr_14", 1.0))
        if atr <= 0:
            atr = 1.0
        bar_idx = int(bar.get("bar_index", 0))

        brake_status = self.brake.check(self._equity, self._daily_pnl)

        ctx = StrategyContext(
            current_bar=bar,
            recent_bars=recent_df,
            features=fv,
            regime=regime,
            existing_positions=self._positions,
            account_equity=self._equity,
            daily_pnl=self._daily_pnl,
            atr=atr,
        )

        # Kill switch
        if brake_status.kill_switch:
            logger.error("KILL SWITCH — flattening all positions")
            for pos in list(self._positions):
                await self._close_position_live(pos, bar, regime, "KILL_SWITCH")
            self._positions.clear()
            return

        # Manage existing
        mgmt_actions = self.strategy_selector.manage_positions(
            self._positions, ctx, bar_idx
        )
        for pos_id, mgmt in mgmt_actions.items():
            pos = next((p for p in self._positions if p.id == pos_id), None)
            if pos is None:
                continue
            if mgmt.action == "EXIT":
                await self._close_position_live(pos, bar, regime, mgmt.exit_reason or "STRATEGY")
                self._positions = [p for p in self._positions if p.id != pos_id]
            elif mgmt.action == "ADJUST_STOP" and mgmt.new_stop_loss is not None:
                pos.stop_loss = mgmt.new_stop_loss

        # New signals (rate limit: 1 order per 5 seconds)
        if brake_status.allow_new_entries and len(self._positions) == 0:
            if time.time() - self._last_order_time > 5.0:
                signal = self.strategy_selector.select_signal(ctx)
                if signal is not None:
                    kelly_f = self.kelly.compute(self._trades)
                    regime_scale = self.regime_scaler.compute_scale(regime)
                    qty = self.sizer.calculate_position_size(
                        signal, self._equity, atr, regime,
                        kelly_f, regime_scale, brake_status.scale,
                        self.settings.point_value,
                    )
                    if qty > 0:
                        side = "BUY" if signal.direction == "LONG" else "SELL"
                        result = await self.broker.place_order(
                            signal.symbol, side, qty, "MARKET"
                        )
                        self._last_order_time = time.time()
                        pos = Position(
                            id=str(uuid.uuid4()),
                            symbol=signal.symbol,
                            direction=signal.direction,
                            entry_price=signal.entry_price,
                            current_price=signal.entry_price,
                            quantity=qty,
                            entry_time=time.time(),
                            entry_bar_index=bar_idx,
                            stop_loss=signal.stop_loss,
                            profit_target=signal.profit_target,
                            time_barrier=bar_idx + signal.time_barrier_bars,
                            strategy_name=signal.strategy_name,
                            entry_regime=regime.regime_name,
                        )
                        self._positions.append(pos)

    async def _close_position_live(
        self, pos: Position, bar: dict, regime: RegimePrediction, reason: str
    ) -> None:
        side = "SELL" if pos.direction == "LONG" else "BUY"
        await self.broker.place_order(pos.symbol, side, pos.quantity, "MARKET")
        self._last_order_time = time.time()

        exit_price = float(bar.get("close", pos.current_price))
        if pos.direction == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.quantity * self.settings.point_value
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity * self.settings.point_value

        trade = CompletedTrade(
            id=pos.id,
            symbol=pos.symbol,
            direction=pos.direction,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=time.time(),
            entry_regime=pos.entry_regime,
            exit_regime=regime.regime_name,
            pnl=pnl,
            commission=0.62 * pos.quantity * 2,
            slippage=0.0,
            exit_reason=reason,
            strategy_name=pos.strategy_name,
            holding_bars=int(bar.get("bar_index", 0)) - pos.entry_bar_index,
        )
        self._trades.append(trade)
        self._daily_pnl += pnl - trade.commission
        self._equity += pnl - trade.commission
        logger.info("Closed {} — PnL=${:.2f}, reason={}", pos.symbol, pnl, reason)

    def _save_checkpoint(self) -> None:
        self._last_checkpoint = time.time()
        logger.debug("Checkpoint saved ({} bars, {} trades)",
                      len(self._bar_history), len(self._trades))
