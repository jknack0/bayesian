"""Rate of change indicator."""

from __future__ import annotations

import pandas as pd


def compute_rate_of_change(df: pd.DataFrame, period: int) -> pd.Series:
    """(close - close_N_ago) / close_N_ago"""
    return (df["close"] - df["close"].shift(period)) / df["close"].shift(period)
