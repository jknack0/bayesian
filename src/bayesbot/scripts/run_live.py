"""CLI: Start the live trading loop.

Usage:
    bayesbot live --model models/hmm_params.json --paper
    bayesbot live --model models/hmm_params.json --ib
"""

import asyncio

import click
from loguru import logger

from bayesbot.config import get_settings
from bayesbot.execution.live_loop import LiveTradingLoop
from bayesbot.execution.preflight import PreflightCheck
from bayesbot.regime.hmm import HMMTrainer


@click.command("live")
@click.option("--model", default="models/hmm_params.json", help="HMM parameters file")
@click.option("--paper/--ib", default=True, help="Paper trading or live IB")
def live(model: str, paper: bool):
    """Start the live trading loop."""
    asyncio.run(_run(model, paper))


async def _run(model_path: str, paper: bool):
    settings = get_settings()

    # Load model
    logger.info("Loading HMM model from {}", model_path)
    params = HMMTrainer.load_parameters(model_path)

    # Create broker
    if paper:
        from bayesbot.execution.paper_broker import PaperBroker
        broker = PaperBroker(initial_equity=settings.initial_capital)
    else:
        from bayesbot.execution.ib_broker import IBBroker
        broker = IBBroker(
            host=settings.ib_host,
            port=settings.ib_port,
            client_id=settings.ib_client_id,
            symbol=settings.symbol,
        )

    # Preflight
    preflight = PreflightCheck(model_path=model_path)
    await broker.connect()
    results = await preflight.run_all(broker, equity=settings.initial_capital)
    if not all(r.passed for r in results):
        if not paper:
            logger.error("Preflight failed — aborting live trading")
            return
        logger.warning("Preflight issues detected (paper mode, continuing)")

    # Start loop
    loop = LiveTradingLoop(broker=broker, hmm_params=params, settings=settings)
    try:
        await loop.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await loop.stop()
