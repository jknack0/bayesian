"""Diagnose momentum blocking pattern."""
import pandas as pd
from loguru import logger
logger.remove()  # silence all logging
from bayesbot.backtest.engine import BacktestEngine
from bayesbot.regime.hmm import HMMTrainer

bars = pd.read_csv("data/MES_dollar_bars.csv").tail(25000).reset_index(drop=True)
params = HMMTrainer.load_parameters("models/hmm_params.json")

engine = BacktestEngine()
result = engine.run(bars, pretrained_params=params)

trades = result.trades
mr_trades = [t for t in trades if t.strategy_name == "mean_reversion"]
mom_trades = [t for t in trades if t.strategy_name == "momentum"]

print(f"MR trades: {len(mr_trades)}")
print(f"Mom trades: {len(mom_trades)}")
print()
print("Momentum trade details:")
for t in mom_trades:
    print(f"  bars_held={t.holding_bars:3d}  pnl=${t.pnl:+8.0f}  exit={t.exit_reason}")

total_mom_bars = sum(t.holding_bars for t in mom_trades)
print(f"\nTotal bars momentum blocked: {total_mom_bars}")
print(f"MR bars lost: ~{387 - len(mr_trades)} trades missing")
