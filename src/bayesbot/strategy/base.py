"""Abstract strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from bayesbot.data.models import (
    FeatureVector,
    Position,
    RegimePrediction,
    TradeSignal,
)


@dataclass
class StrategyContext:
    """Everything a strategy needs to make a decision."""
    current_bar: dict
    recent_bars: pd.DataFrame
    features: FeatureVector
    regime: RegimePrediction
    existing_positions: list[Position]
    account_equity: float
    daily_pnl: float
    atr: float  # 14-bar ATR


@dataclass
class PositionManagement:
    action: str  # 'HOLD', 'EXIT', 'ADJUST_STOP', 'ADJUST_TARGET'
    new_stop_loss: float | None = None
    new_profit_target: float | None = None
    exit_reason: str | None = None


class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def applicable_regimes(self) -> list[str]: ...

    @abstractmethod
    def generate_signal(self, ctx: StrategyContext) -> TradeSignal | None: ...

    @abstractmethod
    def manage_position(
        self, position: Position, ctx: StrategyContext
    ) -> PositionManagement: ...

    @staticmethod
    def _get_regime_prob(regime: RegimePrediction, name: str) -> float:
        """Look up probability for a named regime using the label->prob dict."""
        return regime.regime_probabilities.get(name, 0.0)
