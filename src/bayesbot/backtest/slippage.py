"""Slippage and commission model for MES micro futures."""

from __future__ import annotations


class SlippageModel:
    """Fixed + variable slippage per side.

    Base: 1 tick ($1.25 for MES) per side.
    Variable: inversely proportional to volume, proportional to bar range.
    Commission: $0.09 per side per contract (NinjaTrader micro).
    """

    def __init__(
        self,
        base_ticks: float = 1.0,
        tick_size: float = 0.25,
        point_value: float = 5.0,
        commission_per_side: float = 0.09,
    ):
        self.base_slippage = base_ticks * tick_size * point_value
        self.tick_size = tick_size
        self.point_value = point_value
        self.commission = commission_per_side

    def estimate_slippage(
        self, bar_volume: int, bar_range: float, quantity: int
    ) -> float:
        """Slippage in dollars for a single fill.

        MES is highly liquid during RTH — typical slippage is 1-2 ticks.
        We add a small variable component for thin volume, but cap total
        slippage at 3 ticks per side (realistic worst case for MES).
        """
        base = self.base_slippage * quantity  # 1 tick per contract
        # Small bump for thin volume (overnight/holiday bars)
        vol_factor = min(1.0, 500.0 / max(bar_volume, 1))  # 0-1 scale
        variable = self.base_slippage * quantity * vol_factor  # up to 1 extra tick
        total = base + variable
        # Cap at 3 ticks per side per contract ($3.75 for MES)
        max_slip = 3 * self.tick_size * self.point_value * quantity
        return min(total, max_slip)

    def round_trip_cost(self, quantity: int, bar_volume: int = 1000, bar_range: float = 1.0) -> float:
        """Total round-trip cost: 2 × (slippage + commission)."""
        slip = self.estimate_slippage(bar_volume, bar_range, quantity)
        comm = self.commission * quantity
        return 2 * (slip + comm)
