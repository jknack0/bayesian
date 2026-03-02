"""Analyze correlation between MR and ORB strategies."""
import pandas as pd
import numpy as np

trades = pd.read_csv("results/trades_longonly.csv")
col = "strategy"
print(f"Total trades: {len(trades)}")
print(f"Strategies: {trades[col].value_counts().to_dict()}")

mr = trades[trades[col] == "mean_reversion"]
orb = trades[trades[col] == "orb"]
mom = trades[trades[col] == "momentum"]

# 1. Regime overlap — do they compete for the same regime?
print("\n--- REGIME OVERLAP ---")
print(f"\nMR entries by regime:")
print(mr["entry_regime"].value_counts().to_string())
print(f"\nORB entries by regime:")
print(orb["entry_regime"].value_counts().to_string())
print(f"\nMomentum entries by regime:")
if len(mom) > 0:
    print(mom["entry_regime"].value_counts().to_string())

# 2. Direction overlap
print("\n--- DIRECTION ---")
print(f"MR:  {mr['direction'].value_counts().to_dict()}")
print(f"ORB: {orb['direction'].value_counts().to_dict()}")

# 3. PnL distribution comparison
print("\n--- PnL COMPARISON ---")
for name, df in [("MR", mr), ("ORB", orb)]:
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    wr = len(wins) / len(df) * 100 if len(df) > 0 else 0
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf")
    print(f"\n{name}: {len(df)} trades, {wr:.0f}% WR, PF={pf:.2f}")
    print(f"  Gross: ${df['pnl'].sum():+.0f}  Net: ${df['net_pnl'].sum():+.0f}")
    print(f"  Avg win: ${wins['pnl'].mean():+.0f}  Avg loss: ${losses['pnl'].mean():+.0f}")
    print(f"  Avg hold: {df['holding_bars'].mean():.1f} bars")

# 4. Exit reason breakdown
print("\n--- EXIT REASONS ---")
for name, df in [("MR", mr), ("ORB", orb)]:
    print(f"\n{name}:")
    for reason, g in df.groupby("exit_reason"):
        print(f"  {reason:<16s}  {len(g):>4d} trades  ${g['pnl'].sum():>+8.0f}")

# 5. Sequential PnL correlation (window-based)
# Group trades into sequential windows of 20 and compare PnL
print("\n--- SEQUENTIAL PnL CORRELATION ---")
# Use trade index position as time proxy
trades_indexed = trades.reset_index(drop=True)
window = 50  # trades per window
n_windows = len(trades_indexed) // window

mr_window_pnl = []
orb_window_pnl = []
for w in range(n_windows):
    chunk = trades_indexed.iloc[w * window : (w + 1) * window]
    mr_chunk = chunk[chunk[col] == "mean_reversion"]["pnl"].sum()
    orb_chunk = chunk[chunk[col] == "orb"]["pnl"].sum()
    mr_window_pnl.append(mr_chunk)
    orb_window_pnl.append(orb_chunk)

mr_arr = np.array(mr_window_pnl)
orb_arr = np.array(orb_window_pnl)
if len(mr_arr) > 5:
    corr = np.corrcoef(mr_arr, orb_arr)[0, 1]
    print(f"Windowed PnL correlation (50-trade windows): {corr:.3f}")

    # Concordance
    both_pos = ((mr_arr > 0) & (orb_arr > 0)).sum()
    both_neg = ((mr_arr <= 0) & (orb_arr <= 0)).sum()
    disagree = n_windows - both_pos - both_neg
    print(f"Windows: {n_windows}")
    print(f"  Both profitable:   {both_pos} ({both_pos/n_windows*100:.0f}%)")
    print(f"  Both unprofitable: {both_neg} ({both_neg/n_windows*100:.0f}%)")
    print(f"  Disagree:          {disagree} ({disagree/n_windows*100:.0f}%)")

# 6. What would dropping momentum do?
print("\n--- MOMENTUM IMPACT ---")
if len(mom) > 0:
    print(f"Momentum: {len(mom)} trades, ${mom['pnl'].sum():+.0f} gross, ${mom['net_pnl'].sum():+.0f} net")
    print(f"Removing it saves {len(mom)} trades of cost overhead")
    print(f"Net impact of dropping: ${-mom['net_pnl'].sum():+.0f}")
