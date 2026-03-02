"""Online HMM inference via the forward algorithm.

Processes one observation at a time, maintaining filtered state distribution
P(state_t | obs_{1:t}).  Uses log-space throughout to prevent underflow.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from bayesbot.data.models import RegimePrediction
from bayesbot.regime.hmm import HMMParameters


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


class ForwardFilter:
    """Online forward algorithm for real-time HMM inference.

    Precomputes Cholesky factors of emission covariances at init.
    Each ``update()`` call is O(K² + K*D²) — negligible latency.
    """

    def __init__(self, params: HMMParameters):
        self.params = params
        self.n_states = params.n_states
        self.n_features = len(params.feature_names)

        # Filtered distribution in log space
        self.log_alpha: np.ndarray = np.log(params.initial_distribution + 1e-300)

        # Detect covariance type from shape: (K, D) = diagonal, (K, D, D) = full
        self._is_diag = len(params.emission_covariances.shape) == 2

        self._chol_factors: list = []
        self._log_dets: list[float] = []
        for k in range(self.n_states):
            if self._is_diag:
                # Diagonal covariance: store variance vector directly
                var = params.emission_covariances[k] + 1e-6
                log_det = float(np.sum(np.log(var)))
                self._chol_factors.append(var)
                self._log_dets.append(log_det)
            else:
                cov = params.emission_covariances[k]
                cov = cov + np.eye(cov.shape[0]) * 1e-6
                L, lower = cho_factor(cov)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                self._chol_factors.append((L, lower))
                self._log_dets.append(log_det)

        self.log_transition = np.log(params.transition_matrix + 1e-300)
        self._bar_count = 0

    def update(self, observation: np.ndarray) -> RegimePrediction:
        """Process one observation vector and return the regime prediction."""
        self._bar_count += 1

        # --- Prediction step ---
        log_alpha_pred = np.zeros(self.n_states)
        for j in range(self.n_states):
            log_alpha_pred[j] = _logsumexp(self.log_alpha + self.log_transition[:, j])

        # --- Observation step ---
        log_obs = np.zeros(self.n_states)
        for k in range(self.n_states):
            log_obs[k] = self._log_gaussian_pdf(
                observation,
                self.params.emission_means[k],
                self._chol_factors[k],
                self._log_dets[k],
            )

        self.log_alpha = log_alpha_pred + log_obs
        log_norm = _logsumexp(self.log_alpha)
        self.log_alpha -= log_norm

        probs = np.exp(self.log_alpha)
        probs = probs / probs.sum()

        best = int(np.argmax(probs))
        regime_probs = {
            self.params.state_labels[i]: float(probs[i])
            for i in range(self.params.n_states)
        }
        return RegimePrediction(
            timestamp=0.0,
            bar_index=self._bar_count,
            most_likely_regime=best,
            regime_name=self.params.state_labels[best],
            state_probabilities=probs.tolist(),
            confidence=float(probs[best]),
            regime_probabilities=regime_probs,
        )

    def _log_gaussian_pdf(
        self,
        x: np.ndarray,
        mean: np.ndarray,
        chol_factor_or_var,
        log_det: float,
    ) -> float:
        """log N(x | μ, Σ) using precomputed Cholesky or diagonal variance."""
        diff = x - mean
        k = len(x)
        if self._is_diag:
            quad = np.sum(diff ** 2 / chol_factor_or_var)
        else:
            z = cho_solve(chol_factor_or_var, diff)
            quad = np.dot(diff, z)
        return -0.5 * (k * np.log(2 * np.pi) + log_det + quad)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        return {"log_alpha": self.log_alpha.tolist(), "bar_count": self._bar_count}

    def load_state(self, state: dict) -> None:
        self.log_alpha = np.array(state["log_alpha"])
        self._bar_count = state["bar_count"]
