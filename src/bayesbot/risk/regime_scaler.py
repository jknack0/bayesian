"""Regime-conditional risk scaling — THE most important risk feature.

The 0.3 scale in volatile regimes is the single most impactful parameter.
The Bayesian model's primary value is KNOWING WHEN TO STAY SMALL.
"""

from __future__ import annotations

from bayesbot.data.models import RegimePrediction


# Scale factors per regime
REGIME_SCALES = {
    "trending": 1.0,        # full size — highest Sharpe regime
    "mean_reverting": 0.8,  # slightly reduced — lower payoff per trade
    "volatile": 0.3,        # dramatically reduced — survival mode
}

# When the model is confused (max prob < 0.5), apply this penalty
UNCERTAINTY_PENALTY = 0.5

# When BOCPD fires, override everything
BOCPD_OVERRIDE_SCALE = 0.2


class RegimeRiskScaler:
    def compute_scale(
        self,
        regime: RegimePrediction,
        bocpd_alert: bool = False,
    ) -> float:
        if bocpd_alert:
            return BOCPD_OVERRIDE_SCALE

        scale = REGIME_SCALES.get(regime.regime_name, 0.5)

        if regime.confidence < 0.5:
            scale *= UNCERTAINTY_PENALTY

        return scale
