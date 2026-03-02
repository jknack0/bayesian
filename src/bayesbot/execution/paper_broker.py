"""Paper-trading broker simulator.  Fills immediately with slippage."""

from __future__ import annotations

import uuid

from loguru import logger

from bayesbot.backtest.slippage import SlippageModel
from bayesbot.execution.broker import BaseBroker


class PaperBroker(BaseBroker):
    def __init__(self, initial_equity: float = 25_000.0, point_value: float = 5.0):
        self.equity = initial_equity
        self.point_value = point_value
        self.positions: list[dict] = []
        self.orders: list[dict] = []
        self.slippage = SlippageModel(point_value=point_value)
        self._connected = False

    async def connect(self) -> None:
        self._connected = True
        logger.info("PaperBroker connected (equity=${:.0f})", self.equity)

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("PaperBroker disconnected")

    async def get_account_summary(self) -> dict:
        unrealized = sum(p.get("unrealized_pnl", 0) for p in self.positions)
        return {
            "equity": self.equity + unrealized,
            "cash": self.equity,
            "unrealized_pnl": unrealized,
            "positions": len(self.positions),
        }

    async def get_positions(self) -> list[dict]:
        return list(self.positions)

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
    ) -> dict:
        order_id = str(uuid.uuid4())
        slip = self.slippage.estimate_slippage(1000, 1.0, quantity)
        comm = self.slippage.commission * quantity

        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "status": "FILLED",
            "slippage": slip,
            "commission": comm,
        }
        self.orders.append(order)
        logger.info("Paper order filled: {} {} {} (slip=${:.2f}, comm=${:.2f})",
                     side, quantity, symbol, slip, comm)
        return order

    async def cancel_order(self, order_id: str) -> None:
        for o in self.orders:
            if o["id"] == order_id:
                o["status"] = "CANCELLED"

    async def subscribe_ticks(self, symbol: str, callback) -> None:
        logger.info("PaperBroker: tick subscription is a no-op (use data feed)")
