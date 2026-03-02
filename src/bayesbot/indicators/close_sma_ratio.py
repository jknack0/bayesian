"""Close / SMA ratio indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_close_sma_ratio(df: pd.DataFrame, window: int) -> pd.Series:
    """close / SMA(close).  >1 = uptrend, <1 = downtrend."""
    sma = df["close"].rolling(window).mean().replace(0, np.nan)
    return df["close"] / sma
