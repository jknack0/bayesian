"""Rolling Z-score normalizer with clamping.

Uses a rolling window so normalization adapts to non-stationary markets.
Clamps to [-4, 4] to prevent extreme outliers from distorting the HMM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class NormalizerState:
    """Serializable state for crash recovery."""
    buffers: dict[str, list[float]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


class RollingZScoreNormalizer:
    """Per-feature rolling z-score with clamping.

    Parameters
    ----------
    window : 252
        Roughly 2-3 trading days of dollar bars.
    min_samples : 50
        Need enough history for meaningful z-score.  Before this, return 0.
    clamp_range : (-4, 4)
        Prevents a single flash-crash bar from dominating the feature space.
    """

    def __init__(
        self,
        window: int = 252,
        min_samples: int = 50,
        clamp_range: tuple[float, float] = (-4.0, 4.0),
    ):
        self.window = window
        self.min_samples = min_samples
        self.clamp_range = clamp_range
        self._buffers: dict[str, deque[float]] = {}

    def normalize(self, features: dict[str, float]) -> dict[str, float]:
        """Normalize a single observation (live trading path)."""
        result: dict[str, float] = {}
        for name, value in features.items():
            if name not in self._buffers:
                self._buffers[name] = deque(maxlen=self.window)
            buf = self._buffers[name]
            buf.append(value)

            if len(buf) < self.min_samples:
                result[name] = 0.0
            else:
                arr = np.array(buf)
                mean = arr.mean()
                std = arr.std()
                if std < 1e-10:
                    result[name] = 0.0
                else:
                    z = (value - mean) / std
                    result[name] = float(np.clip(z, *self.clamp_range))
        return result

    def normalize_dataframe(
        self, df: pd.DataFrame, feature_columns: list[str]
    ) -> pd.DataFrame:
        """Batch-normalize a DataFrame while maintaining sequential (no look-ahead) guarantee.

        Processes rows in order so each row's z-score uses only past data.
        """
        out = pd.DataFrame(0.0, index=df.index, columns=feature_columns)
        for col in feature_columns:
            vals = df[col].values
            buf: deque[float] = deque(maxlen=self.window)
            normed = np.zeros(len(vals))
            for i, v in enumerate(vals):
                buf.append(v)
                if len(buf) < self.min_samples:
                    normed[i] = 0.0
                else:
                    arr = np.array(buf)
                    mean = arr.mean()
                    std = arr.std()
                    if std < 1e-10:
                        normed[i] = 0.0
                    else:
                        normed[i] = np.clip((v - mean) / std, *self.clamp_range)
            out[col] = normed
        return out

    def get_state(self) -> NormalizerState:
        return NormalizerState(
            buffers={k: list(v) for k, v in self._buffers.items()},
            counts={k: len(v) for k, v in self._buffers.items()},
        )

    def load_state(self, state: NormalizerState) -> None:
        self._buffers = {
            k: deque(v, maxlen=self.window) for k, v in state.buffers.items()
        }

    def reset(self) -> None:
        self._buffers.clear()
