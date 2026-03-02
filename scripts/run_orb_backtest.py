#!/usr/bin/env python
"""Run VIX-adaptive ORB backtest on time bars.

Tests ORB strategy on different time bar intervals to find the best one.
Merges daily VIX data into bars for VIX-adaptive OR duration.
Allows both LONG and SHORT trades (unlike the main dollar-bar backtest).

Usage:
    python scripts/run_orb_backtest.py --interval 1m --orb-only
    python scripts/run_orb_backtest.py --interval 5m
    python scripts/run_orb_backtest.py --compare  # compare all intervals
"""

import argparse

import pandas as pd
import numpy as np


def load_vix() -> pd.DataFrame:
    """Load daily VIX data. Returns DataFrame with date and vix_close."""
    path = "data/VIX_daily.csv"
    vix = pd.read_csv(path)
    vix["date"] = pd.to_datetime(vix["date"]).dt.strftime("%Y-%m-%d")
    return vix


def merge_vix_into_bars(df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'vix' column to bars by matching on trading date (ET)."""
    from datetime import timezone
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    dates = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(et).dt.strftime("%Y-%m-%d")
    df = df.copy()
    df["_date"] = dates
    vix_map = dict(zip(vix_df["date"], vix_df["vix_close"]))
    df["vix"] = df["_date"].map(vix_map).fillna(20.0)  # default 20 if missing
    df.drop(columns=["_date"], inplace=True)
    return df


def load_time_bars(interval: str) -> pd.DataFrame:
    """Load time bar CSV for the given interval."""
    path = f"data/MES_time_{interval}.csv"
    print(f"Loading {path}...")
    df = pd.read_csv(path)

    if "buy_volume" not in df.columns:
        df["buy_volume"] = (df["volume"] * 0.55).astype(int)
        df["sell_volume"] = df["volume"] - df["buy_volume"]
    if "symbol" not in df.columns:
        df["symbol"] = "MES"
    if "timestamp" not in df.columns:
        df["timestamp"] = df["bar_end"] if "bar_end" in df.columns else df["bar_start"]

    return df


def run_backtest(df: pd.DataFrame, orb_only: bool = False, use_pretrained: bool = True):
    """Run backtest on the given bar DataFrame."""
    from bayesbot.backtest.engine import BacktestEngine

    pretrained_params = None
    if use_pretrained:
        from bayesbot.regime.hmm import HMMTrainer
        hmm_path = "models/hmm_params.json"
        pretrained_params = HMMTrainer.load_parameters(hmm_path)

    # Inject ORB into the strategy selector and allow shorts
    from bayesbot.strategy.selector import StrategySelector
    from bayesbot.strategy.orb import ORBStrategy
    from bayesbot.strategy.defensive import DefensiveStrategy
    from bayesbot.strategy.momentum import MomentumStrategy
    from bayesbot.strategy.mean_reversion import MeanReversionStrategy

    original_init = StrategySelector.__init__

    if orb_only:
        def patched_init(self):
            original_init(self)
            self.strategies = [ORBStrategy(), DefensiveStrategy()]
    else:
        def patched_init(self):
            original_init(self)
            self.strategies = [
                MomentumStrategy(),
                MeanReversionStrategy(),
                DefensiveStrategy(),
                ORBStrategy(),
            ]

    StrategySelector.__init__ = patched_init

    retrain = 999_999 if use_pretrained else 500
    engine = BacktestEngine(initial_capital=25000, retrain_interval=retrain, min_train_bars=300)
    result = engine.run(df, initial_capital=25000, pretrained_params=pretrained_params)

    StrategySelector.__init__ = original_init

    return result


def print_results(result, interval: str, orb_only: bool):
    """Print backtest results."""
    m = result.metrics
    mode = "ORB-only" if orb_only else "All strategies"

    print(f"\n{'='*50}")
    print(f"  {interval} bars — {mode}")
    print(f"{'='*50}")
    print(f"  Total Return:  {m.total_return_pct:.2f}%")
    print(f"  Sharpe Ratio:  {m.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio: {m.sortino_ratio:.3f}")
    print(f"  Max Drawdown:  {m.max_drawdown_pct:.2f}%")
    print(f"  Total Trades:  {m.total_trades}")
    print(f"  Win Rate:      {m.win_rate:.1%}")
    print(f"  Profit Factor: {m.profit_factor:.2f}")

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
                "direction": t.direction,
                "entry_regime": t.entry_regime,
                "holding_bars": t.holding_bars,
            })
        tdf = pd.DataFrame(trades_data)

        print(f"\n  BY STRATEGY")
        print(f"  {'-'*46}")
        for strat, g in tdf.groupby("strategy"):
            wins = g[g["pnl"] > 0]
            losses = g[g["pnl"] <= 0]
            wr = len(wins) / len(g) * 100 if len(g) > 0 else 0
            avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
            avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
            avg_hold = g["holding_bars"].mean()
            print(f"  {strat}")
            print(f"    Trades: {len(g):<4d}  WR: {wr:.0f}%  PnL: ${g['pnl'].sum():>+.0f}  Net: ${g['net_pnl'].sum():>+.0f}")
            print(f"    Avg win: ${avg_win:>+.0f}  Avg loss: ${avg_loss:>+.0f}  Avg hold: {avg_hold:.0f} bars")

        # ORB-specific: show long vs short breakdown
        orb_trades = tdf[tdf["strategy"] == "orb"]
        if len(orb_trades) > 0 and "direction" in orb_trades.columns:
            print(f"\n  BY DIRECTION (ORB)")
            print(f"  {'-'*46}")
            for d, g in orb_trades.groupby("direction"):
                wins = g[g["pnl"] > 0]
                wr = len(wins) / len(g) * 100 if len(g) > 0 else 0
                print(f"  {d:<6s}  {len(g)} trades  WR: {wr:.0f}%  PnL: ${g['pnl'].sum():>+.0f}  Net: ${g['net_pnl'].sum():>+.0f}")

        print(f"\n  BY EXIT REASON")
        print(f"  {'-'*46}")
        for reason, g in tdf.groupby("exit_reason"):
            print(f"  {reason:<16s}  {len(g)} trades  Avg PnL: ${g['pnl'].mean():>+.0f}  Total: ${g['pnl'].sum():>+.0f}")


def main():
    parser = argparse.ArgumentParser(description="Run VIX-adaptive ORB backtest on time bars")
    parser.add_argument("--interval", default="1m", help="Bar interval: 1m, 3m, 5m")
    parser.add_argument("--orb-only", action="store_true", help="Only run ORB + defensive strategies")
    parser.add_argument("--compare", action="store_true", help="Compare all intervals")
    parser.add_argument("--last", type=int, default=0, help="Use only the last N bars (0 = all)")
    parser.add_argument("--no-pretrained", action="store_true", help="Don't use pre-trained HMM")
    args = parser.parse_args()

    # Load VIX data
    try:
        vix_df = load_vix()
        print(f"Loaded VIX data: {len(vix_df)} days, range {vix_df['vix_close'].min():.1f}-{vix_df['vix_close'].max():.1f}")
    except FileNotFoundError:
        print("Warning: data/VIX_daily.csv not found. Using default VIX=20.")
        vix_df = None

    intervals = ["1m", "3m", "5m"] if args.compare else [args.interval]
    results_summary = []

    for interval in intervals:
        try:
            df = load_time_bars(interval)
        except FileNotFoundError:
            print(f"Error: data/MES_time_{interval}.csv not found. Run scripts/build_time_bars.py first.")
            continue

        # Merge VIX into bars
        if vix_df is not None:
            df = merge_vix_into_bars(df, vix_df)
            vix_mean = df["vix"].mean()
            print(f"  VIX coverage: {(df['vix'] != 20.0).mean():.0%} of bars, mean VIX={vix_mean:.1f}")

        if args.last > 0:
            df = df.tail(args.last).reset_index(drop=True)

        print(f"\nLoaded {len(df):,} {interval} bars")
        print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")

        result = run_backtest(
            df,
            orb_only=args.orb_only,
            use_pretrained=not args.no_pretrained,
        )
        print_results(result, interval, args.orb_only)

        results_summary.append({
            "interval": interval,
            "bars": len(df),
            "trades": result.metrics.total_trades,
            "sharpe": result.metrics.sharpe_ratio,
            "return_pct": result.metrics.total_return_pct,
            "max_dd": result.metrics.max_drawdown_pct,
            "win_rate": result.metrics.win_rate,
        })

    if args.compare and len(results_summary) > 1:
        print(f"\n{'='*60}")
        print(f"  COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Interval':<10s} {'Bars':>8s} {'Trades':>7s} {'Sharpe':>8s} {'Return':>8s} {'MaxDD':>8s} {'WR':>6s}")
        print(f"  {'-'*60}")
        for r in results_summary:
            print(f"  {r['interval']:<10s} {r['bars']:>8,d} {r['trades']:>7d} {r['sharpe']:>8.3f} {r['return_pct']:>7.1f}% {r['max_dd']:>7.1f}% {r['win_rate']:>5.1%}")


if __name__ == "__main__":
    main()
