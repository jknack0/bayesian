"""Combined regime detector: HMM forward filter + BOCPD overlay."""

from __future__ import annotations

import numpy as np
from loguru import logger

from bayesbot.data.models import FeatureVector, RegimePrediction
from bayesbot.regime.bocpd import BOCPD
from bayesbot.regime.forward_filter import ForwardFilter
from bayesbot.regime.hmm import HMMParameters


class RegimeDetector:
    """Two-layer regime detection.

    The HMM provides smooth, probabilistic regime tracking.
    BOCPD fires on abrupt structural breaks.

    When BOCPD alerts, probability mass is shifted toward the "volatile" state
    regardless of what the HMM thinks — a safety net for unprecedented events.
    """

    def __init__(
        self,
        hmm_params: HMMParameters,
        bocpd_hazard_rate: float = 1 / 250,
        bocpd_weight: float = 0.3,
        volatile_boost_threshold: float = 0.4,
    ):
        self.forward_filter = ForwardFilter(hmm_params)
        self.bocpd = BOCPD(hazard_rate=bocpd_hazard_rate)
        self.bocpd_weight = bocpd_weight
        self.volatile_boost_threshold = volatile_boost_threshold
        self.params = hmm_params
        self._volatile_idx = (
            hmm_params.state_labels.index("volatile")
            if "volatile" in hmm_params.state_labels
            else hmm_params.n_states - 1
        )

    def predict(self, feature_vector: FeatureVector) -> RegimePrediction:
        """Run HMM + BOCPD and return the blended regime prediction."""
        # Build observation from original feature names, then apply PCA if needed
        source_names = self.params.original_feature_names or self.params.feature_names
        obs = np.array(
            [
                feature_vector.normalized_features.get(name, 0.0)
                for name in source_names
            ]
        )
        # Apply PCA transform if model was trained with it
        if self.params.pca_components is not None and self.params.pca_mean is not None:
            obs = (obs - self.params.pca_mean) @ self.params.pca_components.T

        # HMM update
        hmm_pred = self.forward_filter.update(obs)
        hmm_probs = np.array(hmm_pred.state_probabilities)

        # BOCPD update on 1-bar return
        ret = feature_vector.features.get("returns_1", 0.0)
        bocpd_result = self.bocpd.update(ret)

        # Blend if BOCPD fires
        if bocpd_result.change_point_probability > self.volatile_boost_threshold:
            w = self.bocpd_weight
            volatile_hot = np.zeros(self.params.n_states)
            volatile_hot[self._volatile_idx] = 1.0
            final_probs = (1 - w) * hmm_probs + w * volatile_hot
            final_probs /= final_probs.sum()
            logger.warning(
                "BOCPD alert (cp={:.2f}) — boosting volatile state",
                bocpd_result.change_point_probability,
            )
        else:
            final_probs = hmm_probs

        best = int(np.argmax(final_probs))
        regime_probs = {
            self.params.state_labels[i]: float(final_probs[i])
            for i in range(self.params.n_states)
        }
        return RegimePrediction(
            timestamp=feature_vector.timestamp,
            bar_index=feature_vector.bar_index,
            most_likely_regime=best,
            regime_name=self.params.state_labels[best],
            state_probabilities=final_probs.tolist(),
            confidence=float(final_probs[best]),
            regime_probabilities=regime_probs,
        )

    def get_summary(self) -> dict:
        probs = np.exp(self.forward_filter.log_alpha)
        probs = probs / probs.sum()
        best = int(np.argmax(probs))
        return {
            "current_regime": self.params.state_labels[best],
            "confidence": float(probs[best]),
            "probabilities": {
                self.params.state_labels[i]: float(probs[i])
                for i in range(self.params.n_states)
            },
            "bars_processed": self.forward_filter._bar_count,
        }

    def get_state(self) -> dict:
        return {
            "forward_filter": self.forward_filter.get_state(),
            "bocpd": self.bocpd.get_state(),
        }

    def load_state(self, state: dict) -> None:
        self.forward_filter.load_state(state["forward_filter"])
        self.bocpd.load_state(state["bocpd"])

    def reload_model(self, new_params: HMMParameters) -> None:
        """Hot-reload new HMM parameters.  Preserves BOCPD state."""
        bocpd_state = self.bocpd.get_state()
        self.forward_filter = ForwardFilter(new_params)
        self.params = new_params
        self._volatile_idx = (
            new_params.state_labels.index("volatile")
            if "volatile" in new_params.state_labels
            else new_params.n_states - 1
        )
        self.bocpd.load_state(bocpd_state)
