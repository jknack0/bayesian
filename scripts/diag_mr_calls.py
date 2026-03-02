"""Count how many times MR generate_signal is called and why signals are rejected."""
import sys
import pandas as pd
import numpy as np
from loguru import logger
logger.remove()

# Monkey-patch MR to count calls
from bayesbot.strategy.mean_reversion import MeanReversionStrategy

original_generate = MeanReversionStrategy.generate_signal
call_count = 0
skip_reasons = {"gap": 0, "mr_prob": 0, "vol": 0, "kyle": 0, "vwap_dev": 0, "dup": 0, "signal": 0}

def patched_generate(self, ctx):
    global call_count
    call_count += 1

    regime = ctx.regime
    feats = ctx.features.normalized_features
    bar_idx = ctx.features.bar_index
    gap = bar_idx - self._last_bar_index
    self._last_bar_index = bar_idx
    if gap > 3:
        skip_reasons["gap"] += 1
        return None

    mr_prob = self._get_regime_prob(regime, "mean_reverting")
    if mr_prob < 0.40:
        skip_reasons["mr_prob"] += 1
        return None

    vwap_dev = feats.get("vwap_deviation", 0.0)
    vol = feats.get("realized_vol_20", 0.0)
    kyle = feats.get("kyle_lambda_20", 0.0)

    if vol > 1.0:
        skip_reasons["vol"] += 1
        return None
    if kyle > 1.0:
        skip_reasons["kyle"] += 1
        return None

    if vwap_dev > 0.5:
        direction = "SHORT"
    elif vwap_dev < -0.5:
        direction = "LONG"
    else:
        skip_reasons["vwap_dev"] += 1
        return None

    for pos in ctx.existing_positions:
        if pos.direction == direction and pos.strategy_name == self.name:
            skip_reasons["dup"] += 1
            return None

    skip_reasons["signal"] += 1
    # Call original to get the actual signal
    # Reset _last_bar_index since we already updated it
    self._last_bar_index = bar_idx
    return original_generate(self, ctx)

MeanReversionStrategy.generate_signal = patched_generate

from bayesbot.backtest.engine import BacktestEngine
from bayesbot.regime.hmm import HMMTrainer

bars = pd.read_csv("data/MES_dollar_bars.csv").tail(5000).reset_index(drop=True)
params = HMMTrainer.load_parameters("models/hmm_params.json")

engine = BacktestEngine()
result = engine.run(bars, pretrained_params=params)

mr_trades = [t for t in result.trades if t.strategy_name == "mean_reversion"]
mom_trades = [t for t in result.trades if t.strategy_name == "momentum"]

print(f"MR generate_signal called: {call_count} times")
print(f"Skip reasons: {skip_reasons}")
print(f"MR trades: {len(mr_trades)}, Mom trades: {len(mom_trades)}")
