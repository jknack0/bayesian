"""Tick count ratio indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_tick_count_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """tick_count / SMA(tick_count).  Measures activity density."""
    tc = df.get("tick_count", pd.Series(1, index=df.index)).astype(float)
    avg = tc.rolling(window).mean().replace(0, np.nan)
    return tc / avg
