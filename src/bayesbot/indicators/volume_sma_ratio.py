"""Volume / SMA ratio indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_volume_sma_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume / SMA(volume).  >1 = above-average activity."""
    sma = df["volume"].rolling(window).mean().replace(0, np.nan)
    return df["volume"] / sma
