"""Bayesian Online Change-Point Detection (Adams & MacKay, 2007).

Runs in parallel with the HMM as a fast-reacting alarm system.
Detects abrupt structural breaks — flash crashes, circuit breakers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


@dataclass
class BOCPDResult:
    change_point_probability: float  # P(run_length = 0)
    expected_run_length: float
    is_change_point: bool


class BOCPD:
    """Normal-Inverse-Gamma conjugate BOCPD on scalar observations (1-bar return).

    Parameters
    ----------
    hazard_rate : 1/250
        Prior prob of a change at each step (~1/day of dollar bars).
    threshold : 0.5
        Change-point probability above this triggers an alert.
    max_run_length : 500
        Truncate the run-length distribution for memory/speed.
    """

    def __init__(
        self,
        hazard_rate: float = 1 / 250,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 0.01,
        threshold: float = 0.5,
        max_run_length: int = 500,
    ):
        self.hazard = hazard_rate
        self.threshold = threshold
        self.max_rl = max_run_length

        self._mu0 = mu0
        self._kappa0 = kappa0
        self._alpha0 = alpha0
        self._beta0 = beta0

        # Run-length distribution (log space)
        self._log_R = np.array([0.0])

        # Sufficient statistics per run length
        self._muT = np.array([mu0])
        self._kappaT = np.array([kappa0])
        self._alphaT = np.array([alpha0])
        self._betaT = np.array([beta0])

    def update(self, observation: float) -> BOCPDResult:
        """Process one scalar observation and return change-point info."""
        x = observation

        # 1. Predictive log-probability under each run length
        log_pred = self._predictive_log_prob(
            x, self._muT, self._kappaT, self._alphaT, self._betaT
        )

        # 2. Growth probabilities (run length increases by 1)
        log_growth = self._log_R + log_pred + np.log(1 - self.hazard)

        # 3. Change-point probability (run length resets to 0)
        log_cp = _logsumexp(self._log_R + log_pred + np.log(self.hazard))

        # 4. New run-length distribution
        new_log_R = np.empty(len(log_growth) + 1)
        new_log_R[0] = log_cp
        new_log_R[1:] = log_growth

        # Normalize
        log_norm = _logsumexp(new_log_R)
        new_log_R -= log_norm

        # Truncate if needed
        if len(new_log_R) > self.max_rl:
            new_log_R = new_log_R[: self.max_rl]
            log_norm2 = _logsumexp(new_log_R)
            new_log_R -= log_norm2

        self._log_R = new_log_R

        # 5. Update sufficient statistics
        new_mu = np.empty(len(new_log_R))
        new_kappa = np.empty(len(new_log_R))
        new_alpha = np.empty(len(new_log_R))
        new_beta = np.empty(len(new_log_R))

        # Run length 0 → reset to prior
        new_mu[0] = self._mu0
        new_kappa[0] = self._kappa0
        new_alpha[0] = self._alpha0
        new_beta[0] = self._beta0

        # Run lengths 1..N → Bayesian update of NIG params
        old_mu = self._muT[: len(new_log_R) - 1]
        old_kappa = self._kappaT[: len(new_log_R) - 1]
        old_alpha = self._alphaT[: len(new_log_R) - 1]
        old_beta = self._betaT[: len(new_log_R) - 1]

        new_kappa[1:] = old_kappa + 1
        new_mu[1:] = (old_kappa * old_mu + x) / new_kappa[1:]
        new_alpha[1:] = old_alpha + 0.5
        new_beta[1:] = old_beta + 0.5 * old_kappa * (x - old_mu) ** 2 / new_kappa[1:]

        self._muT = new_mu
        self._kappaT = new_kappa
        self._alphaT = new_alpha
        self._betaT = new_beta

        # Output
        cp_prob = float(np.exp(new_log_R[0]))
        probs = np.exp(new_log_R)
        expected_rl = float(np.sum(np.arange(len(probs)) * probs))

        return BOCPDResult(
            change_point_probability=cp_prob,
            expected_run_length=expected_rl,
            is_change_point=cp_prob > self.threshold,
        )

    def _predictive_log_prob(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """Log probability of x under Student-t predictive from NIG conjugate."""
        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        scale = np.maximum(scale, 1e-300)
        z = (x - mu) / scale
        # Student-t log PDF
        log_p = (
            gammaln((df + 1) / 2)
            - gammaln(df / 2)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
            - (df + 1) / 2 * np.log(1 + z**2 / df)
        )
        return log_p

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "log_R": self._log_R.tolist(),
            "muT": self._muT.tolist(),
            "kappaT": self._kappaT.tolist(),
            "alphaT": self._alphaT.tolist(),
            "betaT": self._betaT.tolist(),
        }

    def load_state(self, state: dict) -> None:
        self._log_R = np.array(state["log_R"])
        self._muT = np.array(state["muT"])
        self._kappaT = np.array(state["kappaT"])
        self._alphaT = np.array(state["alphaT"])
        self._betaT = np.array(state["betaT"])


def _logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))
