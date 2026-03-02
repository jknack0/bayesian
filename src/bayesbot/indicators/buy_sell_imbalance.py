"""Buy/sell imbalance indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_buy_sell_imbalance(df: pd.DataFrame) -> pd.Series:
    """(buy_vol - sell_vol) / total_vol.  Range [-1, 1]."""
    buy = df.get("buy_volume", pd.Series(0, index=df.index)).astype(float)
    sell = df.get("sell_volume", pd.Series(0, index=df.index)).astype(float)
    total = (buy + sell).replace(0, np.nan)
    return (buy - sell) / total
