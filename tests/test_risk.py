"""Tests for risk management."""

import pytest

from bayesbot.data.models import CompletedTrade, RegimePrediction, TradeSignal
from bayesbot.risk.cppi import CPPIPositionSizer
from bayesbot.risk.drawdown_brake import DrawdownBrake
from bayesbot.risk.kelly import KellyCalculator
from bayesbot.risk.regime_scaler import RegimeRiskScaler


class TestKellyCalculator:
    def test_no_history_returns_zero(self):
        kelly = KellyCalculator(min_trades=30)
        assert kelly.compute([]) == 0.0

    def test_winning_strategy_returns_positive(self):
        kelly = KellyCalculator(min_trades=5)
        trades = []
        for i in range(40):
            pnl = 100.0 if i % 3 != 0 else -60.0  # ~67% win rate, 100/60 payoff
            trades.append(CompletedTrade(pnl=pnl))
        result = kelly.compute(trades, kelly_fraction=0.25)
        assert result > 0

    def test_losing_strategy_returns_zero(self):
        kelly = KellyCalculator(min_trades=5)
        trades = [CompletedTrade(pnl=-50.0) for _ in range(40)]
        trades.append(CompletedTrade(pnl=10.0))  # one small winner
        result = kelly.compute(trades)
        assert result == 0.0


class TestCPPI:
    def test_cushion_based_sizing(self):
        sizer = CPPIPositionSizer(floor_pct=0.8, max_contracts=4)
        sizer.initialize(25_000.0)

        signal = TradeSignal(
            timestamp=0, symbol="MES", direction="LONG", strength=1.0,
            strategy_name="test", regime="trending", regime_confidence=0.8,
            entry_price=5500.0, stop_loss=5490.0, profit_target=5520.0,
            time_barrier_bars=50,
        )
        qty = sizer.calculate_position_size(
            signal, equity=25_000.0, atr=2.0, regime=RegimePrediction(0, 0, 1, "trending", [0.1, 0.8, 0.1], 0.8),
            kelly_fraction=0.05, regime_scale=1.0, brake_scale=1.0,
        )
        assert 0 <= qty <= 4

    def test_exhausted_cushion_returns_zero(self):
        sizer = CPPIPositionSizer(floor_pct=0.8)
        sizer.initialize(25_000.0)

        signal = TradeSignal(
            timestamp=0, symbol="MES", direction="LONG", strength=1.0,
            strategy_name="test", regime="trending", regime_confidence=0.8,
            entry_price=5500.0, stop_loss=5490.0, profit_target=5520.0,
            time_barrier_bars=50,
        )
        # Equity at the floor
        qty = sizer.calculate_position_size(
            signal, equity=20_000.0, atr=2.0, regime=RegimePrediction(0, 0, 1, "trending", [0.1, 0.8, 0.1], 0.8),
            kelly_fraction=0.05, regime_scale=1.0, brake_scale=1.0,
        )
        assert qty == 0


class TestRegimeScaler:
    def test_volatile_gets_low_scale(self):
        scaler = RegimeRiskScaler()
        regime = RegimePrediction(0, 0, 2, "volatile", [0.1, 0.1, 0.8], 0.8)
        assert scaler.compute_scale(regime) == 0.3

    def test_trending_gets_full_scale(self):
        scaler = RegimeRiskScaler()
        regime = RegimePrediction(0, 0, 1, "trending", [0.1, 0.8, 0.1], 0.8)
        assert scaler.compute_scale(regime) == 1.0

    def test_uncertainty_penalty(self):
        scaler = RegimeRiskScaler()
        regime = RegimePrediction(0, 0, 1, "trending", [0.35, 0.40, 0.25], 0.40)
        scale = scaler.compute_scale(regime)
        assert scale == 1.0 * 0.5  # uncertainty penalty

    def test_bocpd_override(self):
        scaler = RegimeRiskScaler()
        regime = RegimePrediction(0, 0, 1, "trending", [0.1, 0.8, 0.1], 0.8)
        assert scaler.compute_scale(regime, bocpd_alert=True) == 0.2


class TestDrawdownBrake:
    def test_normal_operations(self):
        brake = DrawdownBrake(initial_capital=25_000.0)
        status = brake.check(25_000.0, 0.0)
        assert status.scale == 1.0
        assert status.allow_new_entries

    def test_5pct_drawdown(self):
        brake = DrawdownBrake(initial_capital=25_000.0)
        brake.check(25_000.0, 0.0)  # set peak
        # daily_pnl=0 so the daily loss limit doesn't fire
        status = brake.check(23_500.0, 0.0)
        assert status.scale == 0.75

    def test_20pct_kill_switch(self):
        brake = DrawdownBrake(initial_capital=25_000.0)
        brake.check(25_000.0, 0.0)
        status = brake.check(20_000.0, -5000.0)
        assert status.kill_switch

    def test_daily_loss_limit(self):
        brake = DrawdownBrake(initial_capital=25_000.0, daily_loss_limit=750.0)
        status = brake.check(24_000.0, -800.0)
        assert status.kill_switch
