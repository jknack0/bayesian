# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

BayesBot — an HMM-based regime-adaptive trading system for MES (Micro E-mini S&P 500) futures. Uses a 3-state Hidden Markov Model to classify market regimes in real-time and routes trades to regime-appropriate strategies.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run tests
pytest tests/
pytest tests/test_strategy.py          # single file
pytest tests/test_regime.py -k "test_forward_filter" -v  # single test

# Fast backtest (~1 year, pretrained HMM, no walk-forward)
python scripts/run_backtest_only.py --last 25000 --no-wf --use-pretrained

# Full 15-year backtest
python scripts/run_backtest_only.py

# CLI (after pip install)
bayesbot import --file data/MES_dollar_bars.csv --symbol MES --format databento --threshold 2000000
bayesbot train --file data/MES_dollar_bars.csv --states 3 --restarts 10 --output models/hmm_params.json
bayesbot backtest --file data/MES_dollar_bars.csv --capital 25000 --output results/
bayesbot live --symbol MES --capital 25000

# Lint
ruff check src/ tests/
```

## Architecture

### Data Flow

```
Raw ticks → Dollar bars → Features (24 indicators) → Normalization (rolling z-score)
    → HMM regime detection → Strategy signal selection → Risk sizing → Execution
```

Dollar bars sample by cumulative dollar-volume (not time), producing approximately IID-normal returns that satisfy HMM Gaussian assumptions. The primary data file is `data/MES_dollar_bars.csv` (51,742 bars, 15 years).

### HMM Regime Detection (`src/bayesbot/regime/`)

Two-layer system:
1. **HMM forward filter** — 3-state Gaussian model with PCA (11 components). States: `volatile` (0), `trending` (1), `mean_reverting` (2). Pretrained params in `models/hmm_params.json`.
2. **BOCPD** (Bayesian Online Changepoint Detection) — safety net for structural breaks. If changepoint probability > 0.4, boosts volatile state weight by 0.3.

Output is `RegimePrediction` with `regime_probabilities` dict (`{label → prob}`) used by strategies for soft gating.

### Strategy System (`src/bayesbot/strategy/`)

`StrategySelector` routes signals by regime:
- **MeanReversion** (primary) — mean_reverting regime. Fades VWAP overextensions. Entry: vwap_dev < -0.5, stop: 1.25 ATR, target: VWAP + ATR, time barrier: 30 bars. Bar-gap filter skips first 3 bars after regime transition.
- **Momentum** — trending regime. Multi-timeframe alignment (roc_10, roc_50, sma_ratio). Higher conviction threshold (trending_prob ≥ 0.55).
- **Defensive** — volatile regime only. Not yet evaluated.
- **ORB** — disabled (breakout doesn't work on dollar bars).

Signal selection blends by regime probability: filter by `applicable_regimes`, weight by regime probability, pick strongest. Reject if weighted strength < 0.1.

### Risk Stack (`src/bayesbot/risk/`)

Five-tier position sizing (multiplicative):
1. **CPPI** — floor at 80% capital, 3× multiplier on cushion
2. **Kelly** — quarter-Kelly (0.25×), per-strategy trade history filtering, min 30 trades
3. **Regime scaler** — trending: 1.0, mean_reverting: 0.8, volatile: 0.3. Confidence < 0.5 halves scale. BOCPD override → 0.2.
4. **Drawdown brake** — tiers at -5% (0.75×), -10% (0.50×), -15% (no entries), -20% (kill switch: flatten + cooldown). Daily loss limit: $750.
5. **Signal strength** — multiply by `signal.strength`

Hard cap: 4 contracts max.

### Backtest Engine (`src/bayesbot/backtest/engine.py`)

Event-driven, bar-by-bar loop matching live execution:
1. Periodic HMM retrain (every 500 bars, or use pretrained)
2. Regime prediction per bar
3. Manage open positions (stops, targets, time barriers, regime exits)
4. Generate and select new signals
5. Size via full risk stack, execute if qty > 0

Commission: $0.09/side/contract (NinjaTrader). Annualization assumes 100 bars/day.

### Key Data Models (`src/bayesbot/data/models.py`)

- `DollarBar` — OHLCV + buy/sell volume split + metadata
- `FeatureVector` — raw + normalized features per bar
- `RegimePrediction` — regime label, state probabilities, confidence, `regime_probabilities` dict
- `TradeSignal` — direction, quantity, stop, target, strategy_name
- `Position` / `CompletedTrade` — open/closed trade tracking

## Critical Conventions

- **Regime label order matters.** HMM states are `[volatile, trending, mean_reverting]` (indices 0, 1, 2). Always use `regime_probabilities` dict for lookups, never index into `state_probabilities` by hardcoded position.
- **Feature warm-up.** First ~50 bars produce mostly-zero features due to rolling window initialization. The engine strips zero-padding rows before HMM training.
- **LONG-only filtering.** Currently only long trades are taken (shorts filtered out). This is intentional for the current phase.
- **Strategies must implement** `generate_signal()` and `manage_position()` from `BaseStrategy`, plus declare `applicable_regimes`.
- **Config lives in `.env`** loaded via Pydantic `BaseSettings` in `src/bayesbot/config.py`. Key values: initial capital $25,000, point value $5 (MES).
