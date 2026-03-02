"""Opening Range Breakout strategy — regime-adaptive OR duration.

Adapted from a VIX-adaptive ORB for BayesBot's HMM regime framework.
Maps HMM regimes to OR duration (like VIX levels):
  - Volatile regime  → 30-min OR (wider range, more cautious)
  - Mean-reverting   → 15-min OR (standard)
  - Trending regime  → 5-min OR  (quick breakout)

Only trades during RTH (9:30-16:00 ET).  One entry per session.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from bayesbot.data.models import Position, TradeSignal
from bayesbot.strategy.base import BaseStrategy, PositionManagement, StrategyContext

_ET = ZoneInfo("America/New_York")

# RTH boundaries for ES/MES
_RTH_OPEN = (9, 30)   # 9:30 AM ET
_RTH_CLOSE = (16, 0)  # 4:00 PM ET


class ORBStrategy(BaseStrategy):
    """Regime-adaptive Opening Range Breakout.

    Phase 1 — Build the Opening Range (OR) over the first N minutes of RTH,
              where N is determined by the current HMM regime.
    Phase 2 — After OR is set, enter LONG on a confirmed breakout above OR high
              with volume confirmation.
    Phase 3 — Manage the position with trailing stop and time-based exit.
    """

    def __init__(self):
        # Session state (reset each RTH open)
        self._session_key: str | None = None
        self._or_high: float = 0.0
        self._or_low: float = float("inf")
        self._or_complete: bool = False
        self._or_end_ts: float = 0.0      # unix ts when OR window closes
        self._rth_close_ts: float = 0.0   # unix ts of RTH close
        self._entry_used: bool = False     # one entry per session

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "orb"

    @property
    def applicable_regimes(self) -> list[str]:
        return ["trending"]

    def generate_signal(self, ctx: StrategyContext) -> TradeSignal | None:
        bar = ctx.current_bar
        ts = float(bar.get("timestamp", bar.get("bar_end", 0)))
        if ts <= 0:
            return None

        et_dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(_ET)

        rth_start = et_dt.replace(
            hour=_RTH_OPEN[0], minute=_RTH_OPEN[1], second=0, microsecond=0
        )
        rth_end = et_dt.replace(
            hour=_RTH_CLOSE[0], minute=_RTH_CLOSE[1], second=0, microsecond=0
        )

        # Only trade during RTH
        if et_dt < rth_start or et_dt >= rth_end:
            return None

        # --- Detect new session ---
        session_key = et_dt.strftime("%Y-%m-%d")
        if session_key != self._session_key:
            self._reset_session(session_key, rth_start, rth_end, ctx.regime)

        price = float(bar.get("close", 0))
        high = float(bar.get("high", price))
        low = float(bar.get("low", price))

        # --- Phase 1: Build OR ---
        if not self._or_complete:
            self._or_high = max(self._or_high, high)
            self._or_low = min(self._or_low, low)

            if ts >= self._or_end_ts:
                self._or_complete = True
                or_width = self._or_high - self._or_low
                # Reject degenerate OR (zero-width or absurdly wide)
                if or_width <= 0 or or_width > ctx.atr * 4:
                    self._entry_used = True  # disable entry for this session
            return None

        # --- Phase 2: Breakout detection ---
        if self._entry_used:
            return None

        # Don't enter in last 30 min — not enough time for measured move
        mins_to_close = (self._rth_close_ts - ts) / 60.0
        if mins_to_close < 30:
            return None

        # Already holding an ORB position?
        for pos in ctx.existing_positions:
            if pos.strategy_name == self.name:
                return None

        or_width = self._or_high - self._or_low

        # Breakout above OR high → LONG
        if price <= self._or_high:
            return None

        # Confirmation: above-average volume
        feats = ctx.features.normalized_features
        vol_ratio = feats.get("volume_sma_ratio", 0.0)
        if vol_ratio < 0.0:  # z-scored: < 0 means below average
            return None

        # Regime filter
        trending_prob = self._get_regime_prob(ctx.regime, "trending")
        mr_prob = self._get_regime_prob(ctx.regime, "mean_reverting")
        combined_prob = trending_prob + mr_prob
        if combined_prob < 0.40:
            return None

        # Risk parameters — OR-width based for symmetric R:R
        stop = price - or_width                  # 1:1 risk to OR width
        target = price + or_width                # 1:1 R:R

        # Time barrier: approximate bars remaining until 15 min before close
        bars_to_close = max(int(mins_to_close / 3), 10)

        self._entry_used = True  # one trade per session

        return TradeSignal(
            timestamp=ctx.features.timestamp,
            symbol=bar.get("symbol", "MES"),
            direction="LONG",
            strength=min(combined_prob * 0.8, 1.0),
            strategy_name=self.name,
            regime=ctx.regime.regime_name,
            regime_confidence=ctx.regime.confidence,
            entry_price=price,
            stop_loss=stop,
            profit_target=target,
            time_barrier_bars=bars_to_close,
        )

    def manage_position(
        self, position: Position, ctx: StrategyContext
    ) -> PositionManagement:
        bar = ctx.current_bar
        ts = float(bar.get("timestamp", bar.get("bar_end", 0)))
        price = float(bar.get("close", position.current_price))

        # Time stop: exit 5 min before RTH close
        if self._rth_close_ts > 0:
            mins_left = (self._rth_close_ts - ts) / 60.0
            if mins_left < 5:
                return PositionManagement(action="EXIT", exit_reason="RTH_CLOSE")

        # Regime flip to volatile → protect capital
        volatile_prob = self._get_regime_prob(ctx.regime, "volatile")
        if volatile_prob > 0.65:
            return PositionManagement(action="EXIT", exit_reason="REGIME_CHANGE")

        # Trail stop to breakeven once price reaches 50% of target
        or_width = self._or_high - self._or_low if self._or_high > self._or_low else ctx.atr
        halfway = position.entry_price + or_width * 0.5
        if position.direction == "LONG" and price >= halfway:
            be_stop = position.entry_price
            if be_stop > position.stop_loss:
                return PositionManagement(
                    action="ADJUST_STOP", new_stop_loss=be_stop
                )

        return PositionManagement(action="HOLD")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_session(self, session_key, rth_start, rth_end, regime):
        """Reset state for a new trading session."""
        self._session_key = session_key
        self._or_high = 0.0
        self._or_low = float("inf")
        self._or_complete = False
        self._entry_used = False

        or_minutes = self._or_duration_minutes(regime)
        self._or_end_ts = rth_start.timestamp() + or_minutes * 60
        self._rth_close_ts = rth_end.timestamp()

    @staticmethod
    def _or_duration_minutes(regime) -> int:
        """Map HMM regime to OR duration in minutes."""
        volatile_prob = ORBStrategy._get_regime_prob(regime, "volatile")
        trending_prob = ORBStrategy._get_regime_prob(regime, "trending")

        if volatile_prob > 0.5:
            return 30  # high uncertainty → wider OR
        if trending_prob > 0.5:
            return 5   # strong trend → quick breakout
        return 15      # default

