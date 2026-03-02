"""Performance metrics for backtesting and live monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bayesbot.data.models import CompletedTrade


@dataclass
class PerformanceMetrics:
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_bars: int = 0
    total_trades: int = 0
    win_rate: float = 0.0
    payoff_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    regime_metrics: dict = field(default_factory=dict)


def compute_metrics(
    equity_curve: np.ndarray,
    trades: list[CompletedTrade],
    bars_per_day: float = 100.0,
) -> PerformanceMetrics:
    """Compute full performance report from an equity curve and trade list."""
    m = PerformanceMetrics()

    if len(equity_curve) < 2:
        return m

    # --- Returns ---
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return m

    m.total_return_pct = float((equity_curve[-1] / equity_curve[0] - 1) * 100)

    trading_days = len(equity_curve) / bars_per_day
    years = max(trading_days / 252, 1 / 252)
    m.annualized_return_pct = float(
        ((equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1) * 100
    )

    # --- Sharpe (annualized) ---
    daily_factor = np.sqrt(bars_per_day * 252)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    m.sharpe_ratio = float(mean_ret / std_ret * daily_factor) if std_ret > 0 else 0.0

    # --- Sortino ---
    downside = returns[returns < 0]
    down_std = np.std(downside) if len(downside) > 0 else 1e-10
    m.sortino_ratio = float(mean_ret / down_std * daily_factor) if down_std > 0 else 0.0

    # --- Max drawdown ---
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / peak
    m.max_drawdown_pct = float(np.max(dd) * 100)

    # Drawdown duration
    in_dd = dd > 0
    if in_dd.any():
        max_dur = 0
        cur_dur = 0
        for v in in_dd:
            if v:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        m.max_drawdown_duration_bars = max_dur

    # Calmar
    m.calmar_ratio = (
        float(m.annualized_return_pct / m.max_drawdown_pct)
        if m.max_drawdown_pct > 0
        else 0.0
    )

    # --- VaR / CVaR ---
    sorted_ret = np.sort(returns)
    idx_95 = max(int(0.05 * len(sorted_ret)), 0)
    m.var_95 = float(sorted_ret[idx_95]) if idx_95 < len(sorted_ret) else 0.0
    m.cvar_95 = float(np.mean(sorted_ret[: idx_95 + 1])) if idx_95 > 0 else m.var_95

    # --- Trade metrics ---
    m.total_trades = len(trades)
    if trades:
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        m.win_rate = len(winners) / len(trades)
        m.avg_trade_pnl = float(np.mean([t.pnl for t in trades]))
        m.avg_winner = float(np.mean([t.pnl for t in winners])) if winners else 0.0
        m.avg_loser = float(np.mean([t.pnl for t in losers])) if losers else 0.0
        m.payoff_ratio = (
            abs(m.avg_winner / m.avg_loser) if m.avg_loser != 0 else 0.0
        )
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        m.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        # Per-regime breakdown
        regimes: dict[str, list[CompletedTrade]] = {}
        for t in trades:
            regimes.setdefault(t.entry_regime, []).append(t)
        for regime_name, regime_trades in regimes.items():
            r_pnls = [t.pnl for t in regime_trades]
            r_wins = [p for p in r_pnls if p > 0]
            m.regime_metrics[regime_name] = {
                "trades": len(regime_trades),
                "total_pnl": float(sum(r_pnls)),
                "win_rate": len(r_wins) / len(regime_trades) if regime_trades else 0,
                "avg_pnl": float(np.mean(r_pnls)),
            }

    return m
