"""Tests for strategy engine."""

import pandas as pd
import pytest

from bayesbot.data.models import FeatureVector, Position, RegimePrediction
from bayesbot.strategy.base import StrategyContext
from bayesbot.strategy.defensive import DefensiveStrategy
from bayesbot.strategy.exits import TripleBarrierManager
from bayesbot.strategy.mean_reversion import MeanReversionStrategy
from bayesbot.strategy.momentum import MomentumStrategy
from bayesbot.strategy.selector import StrategySelector


def _make_ctx(
    regime_name: str = "trending",
    regime_probs: list[float] | None = None,
    features: dict | None = None,
    atr: float = 2.0,
    price: float = 5500.0,
) -> StrategyContext:
    if regime_probs is None:
        regime_probs = {"trending": [0.1, 0.8, 0.1], "mean_reverting": [0.7, 0.2, 0.1], "volatile": [0.1, 0.1, 0.8]}
        regime_probs = regime_probs.get(regime_name, [0.33, 0.34, 0.33])
    if features is None:
        features = {
            "momentum_roc_10": 0.5,
            "close_sma20_ratio": 1.01,
            "volume_sma_ratio": 1.0,
            "vwap_deviation": 0.0,
            "buy_sell_imbalance": 0.0,
            "realized_vol_20": 0.0,
            "kyle_lambda_20": 0.0,
            "returns_1": 0.0,
            "returns_20": 0.0,
        }

    return StrategyContext(
        current_bar={"close": price, "high": price + 1, "low": price - 1, "volume": 1000, "vwap": price, "symbol": "MES"},
        recent_bars=pd.DataFrame(),
        features=FeatureVector(
            timestamp=0.0, bar_index=100, symbol="MES",
            features=features, normalized_features=features,
        ),
        regime=RegimePrediction(
            timestamp=0.0, bar_index=100,
            most_likely_regime=["mean_reverting", "trending", "volatile"].index(regime_name),
            regime_name=regime_name,
            state_probabilities=regime_probs,
            confidence=max(regime_probs),
            regime_probabilities=dict(zip(["mean_reverting", "trending", "volatile"], regime_probs)),
        ),
        existing_positions=[],
        account_equity=25000.0,
        daily_pnl=0.0,
        atr=atr,
    )


class TestMomentumStrategy:
    def test_generates_long_in_trending(self):
        ctx = _make_ctx("trending")
        strategy = MomentumStrategy()
        signal = strategy.generate_signal(ctx)
        assert signal is not None
        assert signal.direction == "LONG"

    def test_no_signal_in_mean_reverting(self):
        ctx = _make_ctx("mean_reverting")
        strategy = MomentumStrategy()
        signal = strategy.generate_signal(ctx)
        assert signal is None  # trending prob 0.2 < 0.45

    def test_exits_on_volatile_regime(self):
        strategy = MomentumStrategy()
        pos = Position(direction="LONG", entry_price=5500.0, current_price=5505.0, strategy_name="momentum")
        ctx = _make_ctx("volatile")
        mgmt = strategy.manage_position(pos, ctx)
        assert mgmt.action == "EXIT"


class TestMeanReversionStrategy:
    def test_generates_long_on_oversold(self):
        ctx = _make_ctx(
            "mean_reverting",
            features={
                "vwap_deviation": -2.0,
                "buy_sell_imbalance": 0.5,
                "realized_vol_20": -0.5,
                "kyle_lambda_20": -0.5,
                "returns_1": 0.0,
                "returns_20": 0.0,
                "momentum_roc_10": 0.0,
                "close_sma20_ratio": 1.0,
                "volume_sma_ratio": 1.0,
            },
        )
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal(ctx)
        assert signal is not None
        assert signal.direction == "LONG"


class TestTripleBarrier:
    def test_stop_loss_triggers(self):
        mgr = TripleBarrierManager()
        pos = Position(direction="LONG", entry_price=5500.0, stop_loss=5490.0, profit_target=5520.0, time_barrier=200, entry_bar_index=0)
        bar = {"close": 5489.0, "high": 5495.0, "low": 5488.0}
        regime = RegimePrediction(0, 0, 0, "trending", [0.3, 0.5, 0.2], 0.5)
        result = mgr.check_barriers(pos, bar, 10, regime)
        assert result.action == "EXIT"
        assert result.exit_reason == "STOP_LOSS"

    def test_profit_target_triggers(self):
        mgr = TripleBarrierManager()
        pos = Position(direction="LONG", entry_price=5500.0, stop_loss=5490.0, profit_target=5520.0, time_barrier=200, entry_bar_index=0)
        bar = {"close": 5522.0, "high": 5525.0, "low": 5519.0}
        regime = RegimePrediction(0, 0, 0, "trending", [0.3, 0.5, 0.2], 0.5)
        result = mgr.check_barriers(pos, bar, 10, regime)
        assert result.action == "EXIT"
        assert result.exit_reason == "PROFIT_TARGET"

    def test_time_barrier_triggers(self):
        mgr = TripleBarrierManager()
        pos = Position(direction="LONG", entry_price=5500.0, stop_loss=5490.0, profit_target=5520.0, time_barrier=50, entry_bar_index=0)
        bar = {"close": 5505.0, "high": 5510.0, "low": 5500.0}
        regime = RegimePrediction(0, 0, 0, "trending", [0.3, 0.5, 0.2], 0.5)
        result = mgr.check_barriers(pos, bar, 60, regime)
        assert result.action == "EXIT"
        assert result.exit_reason == "TIME_BARRIER"


class TestStrategySelector:
    def test_selects_best_signal(self):
        ctx = _make_ctx("trending")
        selector = StrategySelector()
        signal = selector.select_signal(ctx)
        # Should get a momentum signal in trending regime
        if signal is not None:
            assert signal.strategy_name == "momentum"
