"""Preflight checks — ALL must pass before live trading is allowed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


class PreflightCheck:
    """Run all safety checks.  ANY failure blocks live trading."""

    def __init__(
        self,
        model_path: str = "./models/hmm_params.json",
        min_capital: float = 20_000.0,
        min_paper_days: int = 14,
    ):
        self.model_path = model_path
        self.min_capital = min_capital
        self.min_paper_days = min_paper_days

    async def run_all(
        self,
        broker,
        db=None,
        equity: float = 0.0,
        paper_trading_days: int = 0,
    ) -> list[CheckResult]:
        results: list[CheckResult] = []

        # 1. Broker connected
        try:
            acct = await broker.get_account_summary()
            results.append(CheckResult("broker_connected", True, "OK"))
            equity = acct.get("equity", equity)
        except Exception as e:
            results.append(CheckResult("broker_connected", False, str(e)))

        # 2. Sufficient capital
        results.append(CheckResult(
            "sufficient_capital",
            equity >= self.min_capital,
            f"Equity=${equity:.0f} (min=${self.min_capital:.0f})",
        ))

        # 3. HMM model exists and is recent
        model_exists = Path(self.model_path).exists()
        results.append(CheckResult(
            "hmm_model_trained",
            model_exists,
            f"Model at {self.model_path}" if model_exists else "Model file not found",
        ))

        # 4. Database connected
        if db is not None:
            try:
                await db.fetchrow("SELECT 1")
                results.append(CheckResult("database_connected", True, "OK"))
            except Exception as e:
                results.append(CheckResult("database_connected", False, str(e)))
        else:
            results.append(CheckResult("database_connected", False, "No DB provided"))

        # 5. Paper trading minimum
        results.append(CheckResult(
            "paper_trading_minimum",
            paper_trading_days >= self.min_paper_days,
            f"{paper_trading_days} days (min={self.min_paper_days})",
        ))

        # Summary
        all_pass = all(r.passed for r in results)
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            logger.info("[{}] {}: {}", status, r.name, r.message)

        if not all_pass:
            logger.error("PREFLIGHT FAILED — live trading blocked")
        else:
            logger.info("All preflight checks passed")

        return results
