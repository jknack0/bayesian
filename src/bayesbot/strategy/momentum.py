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

        # High-conviction trending only
        trending_prob = self._get_regime_prob(regime, "trending")
        if trending_prob < 0.55:
            return None

        roc10 = feats.get("momentum_roc_10", 0.0)
        roc50 = feats.get("momentum_roc_50", 0.0)
        sma_ratio = feats.get("close_sma20_ratio", 0.0)
        vol_ratio = feats.get("volume_sma_ratio", 0.0)
        bar_dur = feats.get("bar_duration_ratio", 0.0)

        if vol_ratio < -0.5:
            return None  # very low volume — no conviction

        # Fast bars only — slow bars are mean-reverting, not trending
        if bar_dur > 0.5:
            return None

        # Determine direction — multi-timeframe alignment
        if roc10 > 0.4 and sma_ratio > 0.4 and roc50 > 0.1:
            direction = "LONG"
        elif roc10 < -0.4 and sma_ratio < -0.4 and roc50 < -0.1:
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
        target_mult = 1.0

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
            time_barrier_bars=20,
            max_quantity=2,  # half size — momentum has thinner edge than MR
        )

    def manage_position(
        self, position: Position, ctx: StrategyContext
    ) -> PositionManagement:
        regime = ctx.regime

        # Exit if regime leaves trending
        trending_prob = self._get_regime_prob(regime, "trending")
        if trending_prob < 0.5:
            return PositionManagement(action="EXIT", exit_reason="REGIME_CHANGE")

        volatile_prob = self._get_regime_prob(regime, "volatile")
        if volatile_prob > 0.6:
            return PositionManagement(action="EXIT", exit_reason="REGIME_CHANGE")

        return PositionManagement(action="HOLD")
