"""Garman-Klass OHLC volatility estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_garman_klass_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Garman-Klass (1980) OHLC volatility — ~8x efficient vs close-to-close."""
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return np.sqrt(gk.rolling(window).mean().clip(lower=0))
