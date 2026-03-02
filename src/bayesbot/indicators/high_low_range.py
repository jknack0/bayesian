"""High-low range indicator."""

from __future__ import annotations

import pandas as pd


def compute_high_low_range(df: pd.DataFrame) -> pd.Series:
    """Normalized intrabar range: (H-L)/C."""
    return (df["high"] - df["low"]) / df["close"]
