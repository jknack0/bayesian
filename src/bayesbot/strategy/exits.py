"""Triple Barrier exit manager (López de Prado).

Three barriers: stop-loss, profit-target, time.
Widths adapt based on ATR and current regime.
"""

from __future__ import annotations

from dataclasses import dataclass

from bayesbot.data.models import Position, RegimePrediction
from bayesbot.strategy.base import PositionManagement


# Barrier widths (ATR multiples) per regime
BARRIER_PROFILES = {
    "trending": {"profit": 2.5, "stop": 2.0, "time_bars": 60},
    "mean_reverting": {"profit": 1.0, "stop": 1.0, "time_bars": 30},
    "volatile": {"profit": 0.75, "stop": 0.5, "time_bars": 15},
}
DEFAULT_PROFILE = {"profit": 1.5, "stop": 1.5, "time_bars": 40}


class TripleBarrierManager:
    """Check stop → profit → time barriers on every bar."""

    def create_barriers(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        regime_name: str,
    ) -> dict:
        profile = BARRIER_PROFILES.get(regime_name, DEFAULT_PROFILE)
        if direction == "LONG":
            stop = entry_price - profile["stop"] * atr
            target = entry_price + profile["profit"] * atr
        else:
            stop = entry_price + profile["stop"] * atr
            target = entry_price - profile["profit"] * atr
        return {
            "stop_loss": stop,
            "profit_target": target,
            "time_barrier_bars": profile["time_bars"],
        }

    def check_barriers(
        self,
        position: Position,
        current_bar: dict,
        current_bar_index: int,
        regime: RegimePrediction,
    ) -> PositionManagement:
        price = current_bar.get("close", position.current_price)
        high = current_bar.get("high", price)
        low = current_bar.get("low", price)

        # 1. STOP-LOSS (check first — conservative assumption)
        if position.direction == "LONG" and low <= position.stop_loss:
            return PositionManagement(action="EXIT", exit_reason="STOP_LOSS")
        if position.direction == "SHORT" and high >= position.stop_loss:
            return PositionManagement(action="EXIT", exit_reason="STOP_LOSS")

        # 2. PROFIT TARGET
        if position.direction == "LONG" and high >= position.profit_target:
            return PositionManagement(action="EXIT", exit_reason="PROFIT_TARGET")
        if position.direction == "SHORT" and low <= position.profit_target:
            return PositionManagement(action="EXIT", exit_reason="PROFIT_TARGET")

        # 3. TIME BARRIER
        bars_held = current_bar_index - position.entry_bar_index
        if bars_held >= position.time_barrier:
            return PositionManagement(action="EXIT", exit_reason="TIME_BARRIER")

        return PositionManagement(action="HOLD")
