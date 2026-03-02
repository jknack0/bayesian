"""Quarter-Kelly position sizing from rolling trade statistics.

Full Kelly maximises log-growth but produces ~50% drawdowns.
Quarter Kelly limits P(50% DD) to ~1.6% — acceptable for sub-$25K.
"""

from __future__ import annotations

from bayesbot.data.models import CompletedTrade


class KellyCalculator:
    """Compute fractional Kelly bet size from recent trade history."""

    def __init__(self, min_trades: int = 30):
        self.min_trades = min_trades

    def compute(
        self,
        trade_history: list[CompletedTrade],
        kelly_fraction: float = 0.25,
        strategy_name: str | None = None,
    ) -> float:
        """Return the fractional Kelly bet size (0 to ~0.15 typically).

        If estimated edge is ≤ 0 or insufficient history, returns 0.
        When strategy_name is provided, only trades from that strategy
        are used — prevents cross-strategy dilution of edge estimates.
        """
        trades = trade_history
        if strategy_name:
            trades = [t for t in trades if t.strategy_name == strategy_name]
        if len(trades) < self.min_trades:
            return 0.0

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        if not wins or not losses:
            return 0.0

        win_rate = len(wins) / len(trades)
        avg_win = sum(t.pnl for t in wins) / len(wins)
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses))

        if avg_loss == 0:
            return 0.0

        payoff_ratio = avg_win / avg_loss

        # Kelly formula: f* = (b*p - q) / b
        q = 1 - win_rate
        f_star = (payoff_ratio * win_rate - q) / payoff_ratio

        if f_star <= 0:
            return 0.0  # no edge — don't trade

        return f_star * kelly_fraction
