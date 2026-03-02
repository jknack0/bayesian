"""Bar duration ratio indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_bar_duration_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Bar duration / SMA(duration).

    >1 = slow bar (quiet, mean-reverting).  <1 = fast bar (trending/volatile).
    Unique to dollar bars — time bars all have the same duration by definition.
    """
    if "bar_start" in df.columns:
        duration = df["timestamp"].astype(float) - df["bar_start"].astype(float)
    else:
        duration = pd.Series(1.0, index=df.index)
    avg = duration.rolling(window).mean().replace(0, np.nan)
    return duration / avg
