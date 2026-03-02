"""Momentum strategy — active when regime is 'trending'."""

from __future__ import annotations

from bayesbot.data.models import Position, TradeSignal
from bayesbot.strategy.base import BaseStrategy, PositionManagement, StrategyContext


class MomentumStrategy(BaseStrategy):
    def __init__(self):
        self._last_bar_index: int = -100

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def applicable_regimes(self) -> list[str]:
        return ["trending"]

    def generate_signal(self, ctx: StrategyContext) -> TradeSignal | None:
        regime = ctx.regime
        feats = ctx.features.normalized_features

        # Regime transition filter — skip first bar after switching to trending
        bar_idx = ctx.features.bar_index
        gap = bar_idx - self._last_bar_index
        self._last_bar_index = bar_idx
        if gap > 3:
            return None

        # Must be in trending regime with sufficient confidence
        trending_prob = self._get_regime_prob(regime, "trending")
        if trending_prob < 0.45:
            return None

        roc10 = feats.get("momentum_roc_10", 0.0)
        sma_ratio = feats.get("close_sma20_ratio", 0.0)
        vol_ratio = feats.get("volume_sma_ratio", 0.0)

        if vol_ratio < -0.5:
            return None  # very low volume — no conviction

        # Determine direction (z-scored: >0 = above average)
        if roc10 > 0.3 and sma_ratio > 0.3:
            direction = "LONG"
        elif roc10 < -0.3 and sma_ratio < -0.3:
            direction = "SHORT"
        else:
            return None

        # Don't double up
        for pos in ctx.existing_positions:
            if pos.direction == direction and pos.strategy_name == self.name:
                return None

        price = ctx.current_bar.get("close", 0.0)
        atr = ctx.atr
        stop_mult = 1.5
        target_mult = 2.0

        if direction == "LONG":
            stop = price - stop_mult * atr
            target = price + target_mult * atr
        else:
            stop = price + stop_mult * atr
            target = price - target_mult * atr

        return TradeSignal(
            timestamp=ctx.features.timestamp,
            symbol=ctx.current_bar.get("symbol", "MES"),
            direction=direction,
            strength=min(trending_prob, 1.0),
            strategy_name=self.name,
            regime=regime.regime_name,
            regime_confidence=regime.confidence,
            entry_price=price,
            stop_loss=stop,
            profit_target=target,
            time_barrier_bars=30,
        )

    def manage_position(
        self, position: Position, ctx: StrategyContext
    ) -> PositionManagement:
        atr = ctx.atr
        regime = ctx.regime

        # Exit if regime leaves trending — don't block other strategies
        trending_prob = self._get_regime_prob(regime, "trending")
        if trending_prob < 0.5:
            return PositionManagement(action="EXIT", exit_reason="REGIME_CHANGE")

        volatile_prob = self._get_regime_prob(regime, "volatile")
        if volatile_prob > 0.6:
            return PositionManagement(action="EXIT", exit_reason="REGIME_CHANGE")

        # Trail stop to breakeven after 1 ATR profit
        if position.direction == "LONG":
            unrealized = position.current_price - position.entry_price
        else:
            unrealized = position.entry_price - position.current_price

        if unrealized > atr and position.stop_loss != position.entry_price:
            return PositionManagement(
                action="ADJUST_STOP", new_stop_loss=position.entry_price
            )

        return PositionManagement(action="HOLD")

