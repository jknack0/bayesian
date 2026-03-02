"""Kyle's Lambda price impact indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_kyle_lambda(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Kyle's Lambda: price impact per unit of order flow.

    Rolling OLS:  delta_price = lambda * signed_volume + epsilon
    HIGH lambda = illiquid (dangerous).  LOW lambda = liquid (tradeable).
    Lambda spikes BEFORE volatility — early warning of regime shift.
    """
    delta_price = df["close"].diff()
    signed_vol = df.get("buy_volume", pd.Series(0, index=df.index)).astype(float) - df.get(
        "sell_volume", pd.Series(0, index=df.index)
    ).astype(float)
    cov = delta_price.rolling(window).cov(signed_vol)
    var = signed_vol.rolling(window).var().replace(0, np.nan)
    return cov / var
