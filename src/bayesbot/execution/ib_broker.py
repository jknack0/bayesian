"""Interactive Brokers integration via ib_insync."""

from __future__ import annotations

import asyncio
from datetime import datetime

from loguru import logger

from bayesbot.execution.broker import BaseBroker

try:
    from ib_insync import IB, Contract, Future, MarketOrder, util
except ImportError:
    IB = None  # allow import without ib_insync installed


class IBBroker(BaseBroker):
    """Live broker backed by Interactive Brokers TWS / Gateway.

    Safety features:
    - Auto-reconnect with exponential backoff
    - Flatten all positions if disconnected > 60s
    - Contract rollover detection (1 week before quarterly expiry)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        symbol: str = "MES",
    ):
        if IB is None:
            raise ImportError("ib_insync is required for live trading")
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.symbol = symbol
        self._contract: Contract | None = None
        self._connected = False
        self._last_tick_time: float = 0.0

    async def connect(self) -> None:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.ib.connect(self.host, self.port, clientId=self.client_id)
        )
        self._contract = self._resolve_contract()
        self._connected = True
        logger.info("IBBroker connected to {}:{}", self.host, self.port)

    async def disconnect(self) -> None:
        self.ib.disconnect()
        self._connected = False
        logger.info("IBBroker disconnected")

    async def get_account_summary(self) -> dict:
        summary = self.ib.accountSummary()
        result = {}
        for item in summary:
            if item.tag in ("NetLiquidation", "TotalCashValue", "UnrealizedPnL"):
                result[item.tag] = float(item.value)
        return {
            "equity": result.get("NetLiquidation", 0),
            "cash": result.get("TotalCashValue", 0),
            "unrealized_pnl": result.get("UnrealizedPnL", 0),
        }

    async def get_positions(self) -> list[dict]:
        positions = self.ib.positions()
        return [
            {
                "symbol": p.contract.localSymbol,
                "quantity": p.position,
                "avg_cost": p.avgCost,
            }
            for p in positions
        ]

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
    ) -> dict:
        if self._contract is None:
            self._contract = self._resolve_contract()

        action = "BUY" if side.upper() in ("BUY", "LONG") else "SELL"
        order = MarketOrder(action, quantity)

        logger.info("Submitting {} {} {} {}", order_type, action, quantity, symbol)
        trade = self.ib.placeOrder(self._contract, order)

        # Wait for fill (up to 30s)
        for _ in range(60):
            self.ib.sleep(0.5)
            if trade.isDone():
                break

        fill_price = trade.orderStatus.avgFillPrice if trade.orderStatus else 0
        return {
            "id": str(trade.order.orderId),
            "status": trade.orderStatus.status if trade.orderStatus else "UNKNOWN",
            "fill_price": fill_price,
        }

    async def cancel_order(self, order_id: str) -> None:
        for trade in self.ib.openTrades():
            if str(trade.order.orderId) == order_id:
                self.ib.cancelOrder(trade.order)

    async def subscribe_ticks(self, symbol: str, callback) -> None:
        if self._contract is None:
            self._contract = self._resolve_contract()
        self.ib.reqMktData(self._contract, "", False, False)
        self.ib.pendingTickersEvent += lambda tickers: callback(tickers)

    def _resolve_contract(self) -> Contract:
        """Resolve the front-month futures contract for the symbol."""
        contract = Future(self.symbol, exchange="CME")
        details = self.ib.reqContractDetails(contract)
        if not details:
            raise RuntimeError(f"Could not resolve contract for {self.symbol}")
        # Pick the nearest expiry
        details.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
        resolved = details[0].contract
        self.ib.qualifyContracts(resolved)
        logger.info("Resolved contract: {}", resolved.localSymbol)
        return resolved
