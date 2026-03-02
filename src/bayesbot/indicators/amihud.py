"""Amihud illiquidity ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Amihud (2002) illiquidity ratio: avg |return| / dollar_volume.

    High Amihud = illiquid -> regime likely volatile or transitioning.
    """
    returns = np.log(df["close"] / df["close"].shift(1)).abs()
    dv = df.get("dollar_volume", df["close"] * df["volume"]).replace(0, np.nan)
    ratio = returns / dv
    return ratio.rolling(window).mean()
