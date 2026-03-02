"""Log returns indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(df: pd.DataFrame, period: int = 1) -> pd.Series:
    """Log returns over N bars: ln(close_t / close_{t-N})."""
    return np.log(df["close"] / df["close"].shift(period))
