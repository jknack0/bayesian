"""Strategy selector — blends signals from all strategies by regime probability."""

from __future__ import annotations

from loguru import logger

from bayesbot.data.models import Position, TradeSignal
from bayesbot.strategy.base import BaseStrategy, PositionManagement, StrategyContext
from bayesbot.strategy.defensive import DefensiveStrategy
from bayesbot.strategy.exits import TripleBarrierManager
from bayesbot.strategy.mean_reversion import MeanReversionStrategy
from bayesbot.strategy.orb import ORBStrategy


class StrategySelector:
    """Soft regime-to-strategy mapping.

    signal_combined = Σ P(regime_k) × strategy_k_signal
    Conflicting directions → no trade (stay flat under uncertainty).
    """

    def __init__(self):
        self.strategies: list[BaseStrategy] = [
            MeanReversionStrategy(),
            DefensiveStrategy(),
            ORBStrategy(),
        ]
        self.barrier_manager = TripleBarrierManager()

    def select_signal(self, ctx: StrategyContext) -> TradeSignal | None:
        """Run all strategies, weight by regime probability, pick strongest."""
        candidates: list[tuple[float, TradeSignal]] = []

        for strategy in self.strategies:
            # Hard gate: only run strategy if current regime matches
            if ctx.regime.regime_name not in strategy.applicable_regimes:
                continue
            signal = strategy.generate_signal(ctx)
            if signal is None:
                continue
            if signal.direction == "SHORT" and signal.strategy_name != "orb":
                continue  # long-only mode (except ORB which trades both sides)
            # Weight by regime probability of applicable regimes
            weight = 0.0
            for regime_name in strategy.applicable_regimes:
                weight += ctx.regime.regime_probabilities.get(regime_name, 0.0)
            candidates.append((weight * signal.strength, signal))

        if not candidates:
            return None

        # Check for conflicting directions
        directions = set(s.direction for _, s in candidates)
        if "LONG" in directions and "SHORT" in directions:
            logger.debug("Conflicting signals — staying flat")
            return None

        # Pick the strongest
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_weight, best_signal = candidates[0]

        if best_weight < 0.1:
            return None  # too weak

        logger.info(
            "Selected signal: {} {} (strategy={}, strength={:.2f})",
            best_signal.direction,
            best_signal.symbol,
            best_signal.strategy_name,
            best_weight,
        )
        return best_signal

    def manage_positions(
        self,
        positions: list[Position],
        ctx: StrategyContext,
        current_bar_index: int,
    ) -> dict[str, PositionManagement]:
        """Manage all open positions.  Returns position_id → action."""
        results: dict[str, PositionManagement] = {}
        for pos in positions:
            # First check triple barrier
            bar_dict = ctx.current_bar
            barrier_result = self.barrier_manager.check_barriers(
                pos, bar_dict, current_bar_index, ctx.regime
            )
            if barrier_result.action == "EXIT":
                results[pos.id] = barrier_result
                continue

            # Then let the originating strategy manage
            for strategy in self.strategies:
                if strategy.name == pos.strategy_name:
                    mgmt = strategy.manage_position(pos, ctx)
                    results[pos.id] = mgmt
                    break
            else:
                results[pos.id] = PositionManagement(action="HOLD")

        return results
