"""Average True Range indicator."""

from __future__ import annotations

import pandas as pd


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (Wilder EMA).

    TR = max(H-L, |H-prevC|, |L-prevC|)
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=window, adjust=False).mean()
