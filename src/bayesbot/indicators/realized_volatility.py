"""Realized volatility indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_realized_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling std of 1-bar log returns (not annualized — will be z-scored)."""
    ret = np.log(df["close"] / df["close"].shift(1))
    return ret.rolling(window).std()
