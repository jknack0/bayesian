"""Order flow imbalance indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_order_flow_imbalance(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Cumulative signed volume over N bars, normalized by total volume."""
    buy = df.get("buy_volume", pd.Series(0, index=df.index)).astype(float)
    sell = df.get("sell_volume", pd.Series(0, index=df.index)).astype(float)
    signed = buy - sell
    total = df["volume"].astype(float)
    return signed.rolling(window).sum() / total.rolling(window).sum().replace(0, np.nan)
