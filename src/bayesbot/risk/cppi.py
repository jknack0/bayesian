"""CPPI (Constant Proportion Portfolio Insurance) with TIPP ratchet.

Floor at 80% of capital.  Only trade with the "cushion" (equity − floor).
TIPP ratchet locks in profits as equity grows.
"""

from __future__ import annotations

from loguru import logger

from bayesbot.data.models import RegimePrediction, TradeSignal


class CPPIPositionSizer:
    """Full position sizing chain: CPPI → Kelly → regime → brake → hard cap."""

    def __init__(
        self,
        floor_pct: float = 0.80,
        multiplier: float = 3.0,
        max_contracts: int = 4,
        ratchet_decay_pct: float = 0.05,
    ):
        self.floor_pct = floor_pct
        self.multiplier = multiplier
        self.max_contracts = max_contracts
        self.ratchet_decay_pct = ratchet_decay_pct
        self._peak_equity: float = 0.0
        self._floor: float = 0.0

    def initialize(self, equity: float) -> None:
        self._peak_equity = equity
        self._floor = equity * self.floor_pct

    def update_equity(self, equity: float) -> None:
        if equity > self._peak_equity:
            self._peak_equity = equity
            self._floor = equity * self.floor_pct

    def calculate_position_size(
        self,
        signal: TradeSignal,
        equity: float,
        atr: float,
        regime: RegimePrediction,
        kelly_fraction: float,
        regime_scale: float,
        brake_scale: float,
        point_value: float = 5.0,
    ) -> int:
        """Return number of contracts (0 to max_contracts)."""
        self.update_equity(equity)

        cushion = max(equity - self._floor, 0.0)
        if cushion <= 0:
            logger.warning("CPPI cushion exhausted — no trading allowed")
            return 0

        # Dollar allocation from CPPI
        cppi_alloc = self.multiplier * cushion

        # Risk per contract: ATR × stop multiple × point value
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        risk_per_contract = max(stop_distance * point_value, 1.0)

        # Base contracts from CPPI allocation
        contracts_raw = cppi_alloc / risk_per_contract

        # Apply Kelly scaling
        if kelly_fraction > 0:
            contracts_raw *= kelly_fraction / 0.25  # normalise to quarter-Kelly baseline

        # Apply regime scaling
        contracts_raw *= regime_scale

        # Apply drawdown brake scaling
        contracts_raw *= brake_scale

        # Signal strength scaling
        contracts_raw *= signal.strength

        cap = signal.max_quantity if signal.max_quantity is not None else self.max_contracts
        contracts = max(0, min(int(contracts_raw), cap))

        logger.debug(
            "CPPI sizing: cushion=${:.0f}, alloc=${:.0f}, risk/ct=${:.0f}, "
            "kelly={:.3f}, regime={:.2f}, brake={:.2f} → {} contracts",
            cushion, cppi_alloc, risk_per_contract,
            kelly_fraction, regime_scale, brake_scale, contracts,
        )
        return contracts

    @property
    def floor(self) -> float:
        return self._floor

    @property
    def cushion(self) -> float:
        return max(self._peak_equity - self._floor, 0.0) if self._peak_equity else 0.0
