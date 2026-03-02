"""Mean-reversion strategy — active when regime is 'mean_reverting'."""

from __future__ import annotations

from bayesbot.data.models import Position, TradeSignal
from bayesbot.strategy.base import BaseStrategy, PositionManagement, StrategyContext


class MeanReversionStrategy(BaseStrategy):
    def __init__(self):
        self._last_bar_index: int = -100  # last bar we were called on

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def applicable_regimes(self) -> list[str]:
        return ["mean_reverting"]

    def generate_signal(self, ctx: StrategyContext) -> TradeSignal | None:
        regime = ctx.regime
        feats = ctx.features.normalized_features

        # Detect regime transition: if we haven't been called for 3+ bars,
        # regime was NOT mean_reverting recently — skip first few bars to settle
        bar_idx = ctx.features.bar_index
        gap = bar_idx - self._last_bar_index
        self._last_bar_index = bar_idx
        if gap > 3:
            return None  # first bar after regime transition — skip

        mr_prob = self._get_regime_prob(regime, "mean_reverting")
        if mr_prob < 0.40:
            return None

        vwap_dev = feats.get("vwap_deviation", 0.0)
        vol = feats.get("realized_vol_20", 0.0)
        kyle = feats.get("kyle_lambda_20", 0.0)

        # Volatility and liquidity must be favourable
        if vol > 1.0:  # well-above-average vol (z-scored)
            return None
        if kyle > 1.0:  # very illiquid
            return None

        price = ctx.current_bar.get("close", 0.0)
        vwap = ctx.current_bar.get("vwap", price)
        atr = ctx.atr

        # Fade overextensions
        if vwap_dev > 0.5:
            direction = "SHORT"
        elif vwap_dev < -0.5:
            direction = "LONG"
        else:
            return None

        for pos in ctx.existing_positions:
            if pos.direction == direction and pos.strategy_name == self.name:
                return None

        stop_mult = 1.25
        if direction == "LONG":
            stop = price - stop_mult * atr
            target = vwap + atr
        else:
            stop = price + stop_mult * atr
            target = vwap - atr

        return TradeSignal(
            timestamp=ctx.features.timestamp,
            symbol=ctx.current_bar.get("symbol", "MES"),
            direction=direction,
            strength=min(mr_prob * 0.9, 1.0),
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
        regime = ctx.regime

        # EXIT IMMEDIATELY if regime flips to trending — fading a real trend
        # is the #1 way to blow up a mean-reversion strategy.
        trending_prob = self._get_regime_prob(regime, "trending")
        if trending_prob > 0.5:
            return PositionManagement(action="EXIT", exit_reason="REGIME_CHANGE")

        volatile_prob = self._get_regime_prob(regime, "volatile")
        if volatile_prob > 0.6:
            return PositionManagement(action="EXIT", exit_reason="REGIME_CHANGE")

        return PositionManagement(action="HOLD")

