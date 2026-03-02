#!/usr/bin/env python
"""Run backtest + walk-forward on existing dollar bars (skip data download + dollar bar creation)."""

import argparse
import sys
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Run backtest on existing dollar bars")
    parser.add_argument("--last", type=int, default=0, help="Use only the last N bars (0 = all)")
    parser.add_argument("--no-wf", action="store_true", help="Skip walk-forward validation")
    parser.add_argument("--use-pretrained", action="store_true", help="Load HMM from models/hmm_params.json")
    args = parser.parse_args()

    csv_path = "data/MES_dollar_bars.csv"
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    if "buy_volume" not in df.columns:
        df["buy_volume"] = (df["volume"] * 0.55).astype(int)
        df["sell_volume"] = df["volume"] - df["buy_volume"]
    if "symbol" not in df.columns:
        df["symbol"] = "MES"
    if "timestamp" not in df.columns:
        df["timestamp"] = df["bar_start"]

    if args.last > 0:
        df = df.tail(args.last).reset_index(drop=True)
        print(f"Using last {args.last} bars")

    print(f"Loaded {len(df):,} dollar bars")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print()

    # --- Load pre-trained HMM if requested ---
    pretrained_params = None
    if args.use_pretrained:
        from bayesbot.regime.hmm import HMMTrainer
        hmm_path = "models/hmm_params.json"
        print(f"Loading pre-trained HMM from {hmm_path}...")
        pretrained_params = HMMTrainer.load_parameters(hmm_path)
        print(f"  States: {pretrained_params.n_states}, Features: {len(pretrained_params.feature_names)}")
        print()

    # --- Backtest ---
    from bayesbot.backtest.engine import BacktestEngine

    print("=" * 50)
    print("BACKTEST (long-only)")
    print("=" * 50)
    retrain = 500 if not args.use_pretrained else 999_999  # skip retraining when using pretrained HMM
    engine = BacktestEngine(initial_capital=25000, retrain_interval=retrain, min_train_bars=300)
    result = engine.run(df, initial_capital=25000, pretrained_params=pretrained_params)

    m = result.metrics
    print()
    print(f"  Total Return:  {m.total_return_pct:.2f}%")
    print(f"  Sharpe Ratio:  {m.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio: {m.sortino_ratio:.3f}")
    print(f"  Max Drawdown:  {m.max_drawdown_pct:.2f}%")
    print(f"  Total Trades:  {m.total_trades}")
    print(f"  Win Rate:      {m.win_rate:.1%}")
    print(f"  Profit Factor: {m.profit_factor:.2f}")

    # --- Regime Distribution ---
    if result.regime_history:
        print()
        print("-" * 50)
        print("REGIME DISTRIBUTION")
        print("-" * 50)
        regime_names = [r.regime_name for r in result.regime_history]
        total_regimes = len(regime_names)
        from collections import Counter
        regime_counts = Counter(regime_names)
        for regime in ["trending", "mean_reverting", "volatile"]:
            count = regime_counts.get(regime, 0)
            pct = count / total_regimes * 100 if total_regimes > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"  {regime:<16s} {pct:5.1f}%  {bar}")

    # --- Detailed Trade Breakdown ---
    if result.trades:
        trades_data = []
        for t in result.trades:
            trades_data.append({
                "pnl": t.pnl,
                "commission": t.commission,
                "slippage": t.slippage,
                "net_pnl": t.pnl - t.commission - t.slippage,
                "exit_reason": t.exit_reason,
                "strategy": t.strategy_name,
                "entry_regime": t.entry_regime,
                "exit_regime": t.exit_regime,
                "holding_bars": t.holding_bars,
                "direction": t.direction,
                "quantity": t.quantity,
            })
        tdf = pd.DataFrame(trades_data)
        tdf.to_csv("results/trades_longonly.csv", index=False)

        gross_pnl = tdf["pnl"].sum()
        total_comm = tdf["commission"].sum()
        total_slip = tdf["slippage"].sum()
        net_pnl = tdf["net_pnl"].sum()

        print()
        print("-" * 50)
        print("COST BREAKDOWN")
        print("-" * 50)
        print(f"  Gross PnL:     ${gross_pnl:>8.0f}")
        print(f"  Commission:    ${total_comm:>8.0f}")
        print(f"  Slippage:      ${total_slip:>8.0f}")
        print(f"  Net PnL:       ${net_pnl:>8.0f}")
        print(f"  Avg cost/trade: ${(total_comm + total_slip) / len(tdf):>7.2f}")

        print()
        print("-" * 50)
        print("BY STRATEGY")
        print("-" * 50)
        for strat, g in tdf.groupby("strategy"):
            wins = g[g["pnl"] > 0]
            losses = g[g["pnl"] <= 0]
            wr = len(wins) / len(g) * 100
            avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
            avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
            avg_hold = g["holding_bars"].mean()
            print(f"  {strat}")
            print(f"    Trades: {len(g):<4d}  WR: {wr:.0f}%  PnL: ${g['pnl'].sum():>+.0f}  Net: ${g['net_pnl'].sum():>+.0f}")
            print(f"    Avg win: ${avg_win:>+.0f}  Avg loss: ${avg_loss:>+.0f}  Avg hold: {avg_hold:.0f} bars")

        print()
        print("-" * 50)
        print("BY ENTRY REGIME")
        print("-" * 50)
        for reg, g in tdf.groupby("entry_regime"):
            wins = g[g["pnl"] > 0]
            wr = len(wins) / len(g) * 100
            print(f"  {reg:<16s}  {len(g)} trades  WR: {wr:.0f}%  PnL: ${g['pnl'].sum():>+.0f}  Net: ${g['net_pnl'].sum():>+.0f}")

        print()
        print("-" * 50)
        print("BY EXIT REASON")
        print("-" * 50)
        for reason, g in tdf.groupby("exit_reason"):
            print(f"  {reason:<16s}  {len(g)} trades  Avg PnL: ${g['pnl'].mean():>+.0f}  Total: ${g['pnl'].sum():>+.0f}")

        print(f"\n  Trades saved -> results/trades_longonly.csv")

    # --- Walk-forward ---
    if args.no_wf:
        print("\nSkipping walk-forward (--no-wf)")
        return

    print()
    print("=" * 50)
    print("WALK-FORWARD VALIDATION (long-only)")
    print("=" * 50)
    from bayesbot.backtest.walk_forward import WalkForwardValidator

    n_bars = len(df)
    train_bars = min(int(n_bars * 0.5), 9000)
    test_bars = min(int(n_bars * 0.2), 3000)

    validator = WalkForwardValidator(
        train_bars=train_bars,
        test_bars=test_bars,
        initial_capital=25000,
    )
    wf = validator.run(df)

    print()
    print(f"  Sharpe (mean +/- std): {wf.aggregate_sharpe_mean:.3f} +/- {wf.aggregate_sharpe_std:.3f}")
    print(f"  Max Drawdown:          {wf.aggregate_max_dd:.1f}%")
    print(f"  Avg Win Rate:          {wf.aggregate_win_rate:.1%}")
    print()
    for criterion, passed in wf.go_no_go.items():
        if criterion == "PASS":
            continue
        mark = "+" if passed else "x"
        print(f"  [{mark}] {criterion}")

    overall = wf.go_no_go.get("PASS", False)
    print(f"\n  VERDICT: {'GO' if overall else 'NO-GO'}")


if __name__ == "__main__":
    main()
