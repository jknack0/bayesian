"""Defensive strategy — active in 'volatile' regime.  Almost always flat."""

from __future__ import annotations

from bayesbot.data.models import Position, TradeSignal
from bayesbot.strategy.base import BaseStrategy, PositionManagement, StrategyContext


class DefensiveStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "defensive"

    @property
    def applicable_regimes(self) -> list[str]:
        return ["volatile"]

    def generate_signal(self, ctx: StrategyContext) -> TradeSignal | None:
        feats = ctx.features.normalized_features
        ret20 = feats.get("returns_20", 0.0)
        imbalance = feats.get("buy_sell_imbalance", 0.0)

        # Only take a small contrarian position on extreme oversold
        if ret20 < -2.0 and imbalance > 0:
            price = ctx.current_bar.get("close", 0.0)
            atr = ctx.atr
            return TradeSignal(
                timestamp=ctx.features.timestamp,
                symbol=ctx.current_bar.get("symbol", "MES"),
                direction="LONG",
                strength=0.3,  # capped low
                strategy_name=self.name,
                regime=ctx.regime.regime_name,
                regime_confidence=ctx.regime.confidence,
                entry_price=price,
                stop_loss=price - 0.75 * atr,
                profit_target=price + 1.0 * atr,
                time_barrier_bars=15,
            )
        return None

    def manage_position(
        self, position: Position, ctx: StrategyContext
    ) -> PositionManagement:
        # Ultra-defensive: exit on any sign of further deterioration
        feats = ctx.features.normalized_features
        if feats.get("returns_1", 0.0) < -2.0:
            return PositionManagement(action="EXIT", exit_reason="BOCPD_ALERT")
        return PositionManagement(action="HOLD")
