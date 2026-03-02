"""Parkinson high-low volatility estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Parkinson (1980) high-low volatility estimator.

    ~5x more efficient than close-to-close for continuous diffusions.
    Formula: sqrt(1/(4N ln2) * sum ln(H/L)^2)
    """
    log_hl = np.log(df["high"] / df["low"])
    return np.sqrt((log_hl**2).rolling(window).mean() / (4 * np.log(2)))
