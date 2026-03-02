"""VIX-Adaptive Opening Range Breakout strategy.

Adapts OR duration to current VIX level:
  VIX < 15  -> 5-min OR  (low vol, quick breakout)
  VIX 15-25 -> 15-min OR (normal)
  VIX 25-35 -> 30-min OR (high vol, wider range)
  VIX > 35  -> 30-min OR (extreme, cautious sizing)

Entry: breakout above OR high (LONG) or below OR low (SHORT)
       with OLS VWAP slope confirmation.
Exits: measured-move target (0.75x OR), OR-width stop (0.3x OR),
       breakeven trail at 0.75x OR profit, flatten at 3:55 PM ET.

Only trades during RTH (9:30-16:00 ET).  One entry per session.
VIX data comes from the bar dict ("vix" column).

Matches the VIXAdaptiveORBStrategy from multi-strategy-bot with
strategy_b_params.json overrides applied.
"""

from __future__ import annotations

from datetime import datetime, time as dt_time, timezone
from zoneinfo import ZoneInfo

from bayesbot.data.models import Position, TradeSignal
from bayesbot.strategy.base import BaseStrategy, PositionManagement, StrategyContext

_ET = ZoneInfo("America/New_York")

# RTH boundaries for ES/MES
_RTH_OPEN = (9, 30)   # 9:30 AM ET
_RTH_CLOSE = (16, 0)  # 4:00 PM ET


class ORBStrategy(BaseStrategy):
    """VIX-Adaptive Opening Range Breakout.

    Phase 1 -- Build the Opening Range over the first N minutes of RTH,
              where N is determined by the current VIX level.
    Phase 2 -- After OR is set, enter on a confirmed breakout above OR high
              (LONG) or below OR low (SHORT) with OLS VWAP slope alignment.
    Phase 3 -- Manage with trailing breakeven, time stop, and measured-move
              target.
    """

    # Entry filters (strategy_b_params.json values)
    OR_WIDTH_MIN_ATR_FRAC = 0.4   # OR must be >= 40% of ATR
    VWAP_SLOPE_LOOKBACK = 5        # bars for OLS VWAP slope calc

    # Exit parameters (multiples of OR width)
    TARGET_OR_MULT = 0.75          # 2.5:1 R:R (stop is 0.3)
    STOP_OR_MULT = 0.3
    TRAIL_TRIGGER_OR = 0.75        # move to breakeven at 75% of OR profit

    # Time filters (ET)
    MAX_ENTRY = (14, 0)    # no new entries after 2:00 PM
    FLATTEN = (15, 55)     # flatten at 3:55 PM

    def __init__(self):
        # Session state (reset each RTH open)
        self._session_key: str | None = None
        self._or_high: float = 0.0
        self._or_low: float = float("inf")
        self._or_width: float = 0.0
        self._or_complete: bool = False
        self._or_end_ts: float = 0.0      # unix ts when OR window closes
        self._entry_used: bool = False     # one entry per session
        self._or_duration_minutes: int = 15

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "orb"

    @property
    def applicable_regimes(self) -> list[str]:
        return ["trending", "mean_reverting", "volatile"]

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
            vix = float(bar.get("vix", 20.0))
            self._reset_session(session_key, rth_start, vix)

        price = float(bar.get("close", 0))
        high = float(bar.get("high", price))
        low = float(bar.get("low", price))

        # --- Phase 1: Build OR ---
        if not self._or_complete:
            self._or_high = max(self._or_high, high)
            self._or_low = min(self._or_low, low)

            if ts >= self._or_end_ts:
                self._or_complete = True
                self._or_width = self._or_high - self._or_low
                # Reject degenerate OR (too narrow vs ATR)
                if self._or_width <= 0 or self._or_width < self.OR_WIDTH_MIN_ATR_FRAC * ctx.atr:
                    self._entry_used = True  # disable entry for this session
            return None

        # --- Phase 2: Breakout detection ---
        if self._entry_used:
            return None

        # Time filter: no entries after MAX_ENTRY
        et_time = et_dt.time()
        if et_time >= dt_time(self.MAX_ENTRY[0], self.MAX_ENTRY[1]):
            return None

        # Already holding an ORB position?
        for pos in ctx.existing_positions:
            if pos.strategy_name == self.name:
                return None

        # VWAP slope from recent bars (OLS regression)
        slope = self._compute_vwap_slope(ctx.recent_bars)

        # Determine breakout direction with VWAP slope confirmation
        direction: str | None = None
        if price > self._or_high and slope is not None and slope > 0:
            direction = "LONG"
        elif price < self._or_low and slope is not None and slope < 0:
            direction = "SHORT"

        if direction is None:
            return None

        # Risk parameters -- 0.3x OR stop, 0.75x OR target -> 2.5:1 R:R
        stop_dist = self.STOP_OR_MULT * self._or_width
        target_dist = self.TARGET_OR_MULT * self._or_width

        if direction == "LONG":
            stop = price - stop_dist
            target = price + target_dist
        else:
            stop = price + stop_dist
            target = price - target_dist

        # Time barrier: bars until flatten time
        bar_start = float(bar.get("bar_start", 0))
        bar_end = float(bar.get("bar_end", bar.get("timestamp", 0)))
        bar_duration_min = max((bar_end - bar_start) / 60.0, 1.0)

        flatten_ts = et_dt.replace(
            hour=self.FLATTEN[0], minute=self.FLATTEN[1], second=0, microsecond=0
        ).timestamp()
        mins_to_flatten = max((flatten_ts - ts) / 60.0, 0)
        bars_to_flatten = max(int(mins_to_flatten / bar_duration_min), 10)

        # VIX-based position scaling (3-tier matching multi-strategy-bot)
        vix = float(bar.get("vix", 20.0))
        if vix > 35:
            max_qty = 1   # extreme: ~30% scale
        elif vix > 25:
            max_qty = 2   # high: ~60% scale
        else:
            max_qty = None  # normal: no extra cap (CPPI decides up to 4)

        self._entry_used = True  # one trade per session

        return TradeSignal(
            timestamp=ctx.features.timestamp,
            symbol=bar.get("symbol", "MES"),
            direction=direction,
            strength=0.7,  # fixed -- VIX-adaptive, not regime-dependent
            strategy_name=self.name,
            regime=ctx.regime.regime_name,
            regime_confidence=ctx.regime.confidence,
            entry_price=price,
            stop_loss=stop,
            profit_target=target,
            time_barrier_bars=bars_to_flatten,
            max_quantity=max_qty,
        )

    def manage_position(
        self, position: Position, ctx: StrategyContext
    ) -> PositionManagement:
        bar = ctx.current_bar
        ts = float(bar.get("timestamp", bar.get("bar_end", 0)))
        price = float(bar.get("close", position.current_price))

        # Time stop: flatten at 3:55 PM ET
        if ts > 0:
            et_dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(_ET)
            if et_dt.time() >= dt_time(self.FLATTEN[0], self.FLATTEN[1]):
                return PositionManagement(action="EXIT", exit_reason="TIME_STOP")

        # Trail stop to breakeven after 0.75x OR width profit
        or_width = self._or_width if self._or_width > 0 else ctx.atr
        trail_trigger = self.TRAIL_TRIGGER_OR * or_width

        if position.direction == "LONG":
            unrealized = price - position.entry_price
            if unrealized >= trail_trigger and position.entry_price > position.stop_loss:
                return PositionManagement(
                    action="ADJUST_STOP", new_stop_loss=position.entry_price
                )
        else:  # SHORT
            unrealized = position.entry_price - price
            if unrealized >= trail_trigger and position.entry_price < position.stop_loss:
                return PositionManagement(
                    action="ADJUST_STOP", new_stop_loss=position.entry_price
                )

        return PositionManagement(action="HOLD")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_session(self, session_key: str, rth_start: datetime, vix: float):
        """Reset state for a new trading session."""
        self._session_key = session_key
        self._or_high = 0.0
        self._or_low = float("inf")
        self._or_width = 0.0
        self._or_complete = False
        self._entry_used = False

        self._or_duration_minutes = self._get_or_duration(vix)
        self._or_end_ts = rth_start.timestamp() + self._or_duration_minutes * 60

    @staticmethod
    def _get_or_duration(vix: float) -> int:
        """OR duration in minutes based on VIX level."""
        if vix < 15:
            return 5
        elif vix < 25:
            return 15
        elif vix < 35:
            return 30
        else:
            return 30

    def _compute_vwap_slope(self, recent_bars) -> float | None:
        """OLS VWAP slope from recent bars.

        Uses ordinary least squares matching the VWAPSlope indicator from
        multi-strategy-bot:
            slope = (N * sum(x*y) - sum(x) * sum(y)) / (N * sum(x^2) - (sum(x))^2)
        where x = [0, 1, ..., N-1] and y = VWAP values.
        """
        if recent_bars is None or len(recent_bars) < self.VWAP_SLOPE_LOOKBACK:
            return None
        col = "vwap" if "vwap" in recent_bars.columns else "close"
        vals = recent_bars[col].iloc[-self.VWAP_SLOPE_LOOKBACK:].values
        n = len(vals)
        if n < 2:
            return None

        # Precomputed x-sums for x = [0, 1, ..., n-1]
        sum_x = n * (n - 1) / 2.0
        sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0

        sum_y = 0.0
        sum_xy = 0.0
        for i in range(n):
            y = float(vals[i])
            sum_y += y
            sum_xy += i * y

        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denom
