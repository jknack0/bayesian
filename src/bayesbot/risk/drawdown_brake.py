"""Multi-tier drawdown brake system.

-5%:   reduce 25%
-10%:  reduce 50%
-15%:  no new entries
-20%:  KILL SWITCH — flatten everything, 24h cooldown

Daily loss limit: $750 (3% of $25K).
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class BrakeStatus:
    scale: float            # 0.0–1.0 multiplier on position sizes
    allow_new_entries: bool
    kill_switch: bool       # True = flatten everything NOW
    message: str


class DrawdownBrake:
    def __init__(
        self,
        initial_capital: float = 25_000.0,
        daily_loss_limit: float = 750.0,
        cooldown_bars: int = 10,  # brief pause after kill switch
    ):
        self.initial_capital = initial_capital
        self.daily_loss_limit = daily_loss_limit
        self.cooldown_bars = cooldown_bars

        self._peak_equity: float = initial_capital
        self._kill_bar: int | None = None
        self._current_bar: int = 0

    def check(
        self, current_equity: float, daily_pnl: float, bar_index: int = 0
    ) -> BrakeStatus:
        """Evaluate all brake tiers."""
        self._current_bar = bar_index

        # Cooldown active?
        if self._kill_bar is not None:
            elapsed = bar_index - self._kill_bar
            if elapsed < self.cooldown_bars:
                return BrakeStatus(
                    scale=0.0,
                    allow_new_entries=False,
                    kill_switch=True,
                    message=f"Kill-switch cooldown: {self.cooldown_bars - elapsed} bars remaining",
                )
            else:
                self._kill_bar = None
                logger.info("Kill-switch cooldown expired — resuming")

        # Update peak
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # Daily loss limit
        if daily_pnl < -self.daily_loss_limit:
            logger.error("Daily loss limit breached (${:.0f})", daily_pnl)
            self._kill_bar = bar_index
            return BrakeStatus(
                scale=0.0,
                allow_new_entries=False,
                kill_switch=True,
                message=f"Daily loss limit ${self.daily_loss_limit:.0f} breached",
            )

        drawdown_pct = (self._peak_equity - current_equity) / self._peak_equity

        if drawdown_pct >= 0.20:
            logger.error("20% drawdown kill switch")
            self._kill_bar = bar_index
            return BrakeStatus(
                scale=0.0,
                allow_new_entries=False,
                kill_switch=True,
                message="20% drawdown — kill switch activated",
            )

        if drawdown_pct >= 0.15:
            return BrakeStatus(
                scale=0.0,
                allow_new_entries=False,
                kill_switch=False,
                message="15% drawdown — entries suspended",
            )

        if drawdown_pct >= 0.10:
            return BrakeStatus(
                scale=0.50,
                allow_new_entries=True,
                kill_switch=False,
                message="10% drawdown — 50% size reduction",
            )

        if drawdown_pct >= 0.05:
            return BrakeStatus(
                scale=0.75,
                allow_new_entries=True,
                kill_switch=False,
                message="5% drawdown — 25% size reduction",
            )

        return BrakeStatus(
            scale=1.0,
            allow_new_entries=True,
            kill_switch=False,
            message="Normal operations",
        )
