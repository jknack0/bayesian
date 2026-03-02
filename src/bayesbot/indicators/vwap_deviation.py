"""VWAP deviation indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vwap_deviation(df: pd.DataFrame) -> pd.Series:
    """(close - vwap) / vwap.  Positive = buying pressure."""
    vwap = df["vwap"].replace(0, np.nan)
    return (df["close"] - vwap) / vwap
