"""Offline HMM training pipeline.

Uses hmmlearn's GaussianHMM with multiple random restarts,
then exports lightweight parameters for the online forward filter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM
from loguru import logger
from sklearn.decomposition import PCA


@dataclass
class HMMParameters:
    """Everything needed for online inference — no hmmlearn dependency required."""
    n_states: int
    feature_names: list[str]
    transition_matrix: np.ndarray       # (K, K)
    emission_means: np.ndarray          # (K, D)
    emission_covariances: np.ndarray    # (K, D, D) full or (K, D) diagonal
    initial_distribution: np.ndarray    # (K,)
    state_labels: list[str]             # e.g. ['mean_reverting', 'trending', 'volatile']
    trained_at: str = ""
    training_date_range: tuple[str, str] = ("", "")
    metrics: dict = field(default_factory=dict)
    # PCA dimensionality reduction (if used during training)
    pca_components: np.ndarray | None = None   # (n_components, D_original)
    pca_mean: np.ndarray | None = None         # (D_original,)
    original_feature_names: list[str] = field(default_factory=list)


@dataclass
class TrainingReport:
    best_params: HMMParameters
    model_comparison: dict       # {n_states: {bic, aic, ll}}
    state_statistics: dict       # per-state stats
    regime_history: np.ndarray   # (T, K) posterior


class HMMTrainer:
    """Trains Gaussian HMMs for regime detection on dollar bar features."""

    def __init__(
        self,
        n_restarts: int = 20,
        max_iter: int = 300,
        tol: float = 1e-4,
        covariance_type: str = "diag",
        use_pca: bool = True,
        pca_variance_threshold: float = 0.95,
    ):
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.use_pca = use_pca
        self.pca_variance_threshold = pca_variance_threshold
        self._pca: PCA | None = None

    def train(
        self,
        feature_matrix: np.ndarray,
        feature_names: list[str],
        n_states: int = 3,
    ) -> tuple[HMMParameters, TrainingReport]:
        """Train with multiple random restarts, pick the best by log-likelihood."""
        original_feature_names = list(feature_names)

        # Optional PCA dimensionality reduction
        if self.use_pca and feature_matrix.shape[1] > 5:
            self._pca = PCA(n_components=self.pca_variance_threshold, svd_solver="full")
            train_data = self._pca.fit_transform(feature_matrix)
            n_components = train_data.shape[1]
            pca_feature_names = [f"pc_{i}" for i in range(n_components)]
            logger.info(
                "PCA reduced {} features to {} components ({:.1f}% variance)",
                len(feature_names), n_components,
                sum(self._pca.explained_variance_ratio_) * 100,
            )
        else:
            self._pca = None
            train_data = feature_matrix
            pca_feature_names = feature_names

        logger.info(
            "Training {}-state HMM on {} samples × {} features ({} restarts, cov={})",
            n_states, train_data.shape[0], train_data.shape[1],
            self.n_restarts, self.covariance_type,
        )

        best_model: GaussianHMM | None = None
        best_score = -np.inf

        for i in range(self.n_restarts):
            try:
                model, score = self._fit_single(train_data, n_states, random_state=i * 42)
                if score > best_score:
                    best_score = score
                    best_model = model
                    logger.debug("Restart {}: score={:.2f} (new best)", i, score)
            except Exception as e:
                logger.warning("Restart {} failed: {}", i, e)

        if best_model is None:
            raise RuntimeError("All HMM training restarts failed")

        # Label states by volatility (use PCA feature names for labeling)
        labels = self._label_states(best_model, pca_feature_names)
        metrics = self._compute_metrics(best_model, train_data, n_states)

        # RCM degenerate state warning
        rcm = metrics.get("rcm", 100.0)
        if rcm < 1.0:
            logger.warning(
                "RCM={:.1f} indicates degenerate state collapse", rcm,
            )

        # Posterior probabilities
        posteriors = best_model.predict_proba(train_data)

        # Per-state statistics (use returns from original feature matrix)
        decoded = best_model.predict(train_data)
        state_stats = {}
        ret_idx = (
            original_feature_names.index("returns_1")
            if "returns_1" in original_feature_names
            else 0
        )
        for k in range(n_states):
            mask = decoded == k
            state_returns = feature_matrix[mask, ret_idx]
            state_stats[labels[k]] = {
                "n_bars": int(mask.sum()),
                "fraction": float(mask.mean()),
                "mean_return": float(state_returns.mean()) if len(state_returns) else 0.0,
                "std_return": float(state_returns.std()) if len(state_returns) else 0.0,
            }

        # hmmlearn 0.3+ always returns (K,D,D) from .covars_;
        # extract diagonal when covariance_type="diag" for compact storage
        covars = best_model.covars_
        if self.covariance_type == "diag" and covars.ndim == 3:
            covars = np.array([np.diag(covars[k]) for k in range(n_states)])

        params = HMMParameters(
            n_states=n_states,
            feature_names=pca_feature_names,
            transition_matrix=best_model.transmat_,
            emission_means=best_model.means_,
            emission_covariances=covars,
            initial_distribution=best_model.startprob_,
            state_labels=labels,
            trained_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            pca_components=self._pca.components_ if self._pca else None,
            pca_mean=self._pca.mean_ if self._pca else None,
            original_feature_names=original_feature_names,
        )

        # Model comparison (2, 3, 4 states)
        comparison = self._compare_state_counts(train_data)

        report = TrainingReport(
            best_params=params,
            model_comparison=comparison,
            state_statistics=state_stats,
            regime_history=posteriors,
        )

        logger.info(
            "Training complete — labels={}, BIC={:.1f}, RCM={:.1f}",
            labels, metrics.get("bic", 0), metrics.get("rcm", 0),
        )
        return params, report

    def _fit_single(
        self, X: np.ndarray, n_states: int, random_state: int
    ) -> tuple[GaussianHMM, float]:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=self.covariance_type,
            n_iter=self.max_iter,
            tol=self.tol,
            random_state=random_state,
            verbose=False,
        )
        model.fit(X)
        return model, model.score(X)

    def _label_states(
        self, model: GaussianHMM, feature_names: list[str]
    ) -> list[str]:
        """Sort states by emission mean of realized_vol_20 (ascending)."""
        vol_idx = (
            feature_names.index("realized_vol_20")
            if "realized_vol_20" in feature_names
            else 0
        )
        vol_means = model.means_[:, vol_idx]
        order = np.argsort(vol_means)
        label_map = {0: "mean_reverting", 1: "trending", 2: "volatile"}
        labels = [""] * model.n_components
        for rank, state_idx in enumerate(order):
            labels[state_idx] = label_map.get(rank, f"state_{rank}")
        return labels

    def _compute_metrics(
        self, model: GaussianHMM, X: np.ndarray, n_states: int
    ) -> dict:
        T, D = X.shape
        ll = model.score(X) * T  # hmmlearn score returns per-sample
        # Free parameters depend on covariance type
        if model.covariance_type == "full":
            n_cov_params = n_states * D * (D + 1) // 2
        elif model.covariance_type == "diag":
            n_cov_params = n_states * D
        elif model.covariance_type == "spherical":
            n_cov_params = n_states
        elif model.covariance_type == "tied":
            n_cov_params = D * (D + 1) // 2
        else:
            n_cov_params = n_states * D
        k = (n_states - 1) + n_states * (n_states - 1) + n_states * D + n_cov_params
        bic = -2 * ll + k * np.log(T)
        aic = -2 * ll + 2 * k

        # Regime Classification Measure: 0 = crisp, 100 = max uncertainty
        posteriors = model.predict_proba(X)
        rcm = 400.0 / T * np.sum(
            np.prod(np.sqrt(posteriors), axis=1)
        ) if T > 0 else 100.0

        return {
            "log_likelihood": float(ll),
            "bic": float(bic),
            "aic": float(aic),
            "rcm": float(rcm),
            "n_samples": T,
            "n_params": k,
        }

    def _compare_state_counts(self, X: np.ndarray) -> dict:
        comparison = {}
        for ns in [2, 3, 4]:
            try:
                model, score = self._fit_single(X, ns, random_state=0)
                metrics = self._compute_metrics(model, X, ns)
                comparison[ns] = metrics
            except Exception as e:
                comparison[ns] = {"error": str(e)}
        return comparison

    @staticmethod
    def save_parameters(params: HMMParameters, path: str) -> None:
        data = {
            "n_states": params.n_states,
            "feature_names": params.feature_names,
            "transition_matrix": params.transition_matrix.tolist(),
            "emission_means": params.emission_means.tolist(),
            "emission_covariances": params.emission_covariances.tolist(),
            "initial_distribution": params.initial_distribution.tolist(),
            "state_labels": params.state_labels,
            "trained_at": params.trained_at,
            "training_date_range": list(params.training_date_range),
            "metrics": params.metrics,
            "pca_components": params.pca_components.tolist() if params.pca_components is not None else None,
            "pca_mean": params.pca_mean.tolist() if params.pca_mean is not None else None,
            "original_feature_names": params.original_feature_names,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info("Saved HMM parameters to {}", path)

    @staticmethod
    def load_parameters(path: str) -> HMMParameters:
        data = json.loads(Path(path).read_text())
        return HMMParameters(
            n_states=data["n_states"],
            feature_names=data["feature_names"],
            transition_matrix=np.array(data["transition_matrix"]),
            emission_means=np.array(data["emission_means"]),
            emission_covariances=np.array(data["emission_covariances"]),
            initial_distribution=np.array(data["initial_distribution"]),
            state_labels=data["state_labels"],
            trained_at=data.get("trained_at", ""),
            training_date_range=tuple(data.get("training_date_range", ["", ""])),
            metrics=data.get("metrics", {}),
            pca_components=np.array(data["pca_components"]) if data.get("pca_components") is not None else None,
            pca_mean=np.array(data["pca_mean"]) if data.get("pca_mean") is not None else None,
            original_feature_names=data.get("original_feature_names", []),
        )
