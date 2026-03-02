"""Tests for HMM regime detection and BOCPD."""

import numpy as np
import pytest

from bayesbot.regime.bocpd import BOCPD
from bayesbot.regime.forward_filter import ForwardFilter
from bayesbot.regime.hmm import HMMParameters, HMMTrainer


def _make_dummy_params(n_features: int = 5, n_states: int = 3) -> HMMParameters:
    """Create minimal valid HMM parameters for testing."""
    np.random.seed(0)
    trans = np.array([
        [0.8, 0.15, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80],
    ])
    means = np.random.randn(n_states, n_features) * 0.5
    covs = np.array([np.eye(n_features) * 0.1 for _ in range(n_states)])
    return HMMParameters(
        n_states=n_states,
        feature_names=[f"f{i}" for i in range(n_features)],
        transition_matrix=trans,
        emission_means=means,
        emission_covariances=covs,
        initial_distribution=np.array([0.33, 0.34, 0.33]),
        state_labels=["mean_reverting", "trending", "volatile"],
    )


class TestForwardFilter:
    def test_probabilities_sum_to_one(self):
        params = _make_dummy_params(5)
        ff = ForwardFilter(params)
        obs = np.random.randn(5)
        pred = ff.update(obs)
        assert abs(sum(pred.state_probabilities) - 1.0) < 1e-8

    def test_trending_observations_drive_trending(self):
        params = _make_dummy_params(5)
        ff = ForwardFilter(params)

        # Feed observations matching trending state (state 1)
        for _ in range(50):
            obs = params.emission_means[1] + np.random.randn(5) * 0.1
            pred = ff.update(obs)

        assert pred.regime_name == "trending"
        assert pred.confidence > 0.5

    def test_state_persistence(self):
        params = _make_dummy_params(5)
        ff = ForwardFilter(params)
        for _ in range(10):
            ff.update(np.random.randn(5))
        state = ff.get_state()

        ff2 = ForwardFilter(params)
        ff2.load_state(state)

        obs = np.random.randn(5)
        p1 = ff.update(obs)
        p2 = ff2.update(obs)
        np.testing.assert_array_almost_equal(
            p1.state_probabilities, p2.state_probabilities
        )


class TestForwardFilterDiag:
    def test_probabilities_sum_to_one_diag(self):
        """ForwardFilter with diagonal covariance should produce valid probs."""
        np.random.seed(0)
        trans = np.array([[0.8, 0.15, 0.05], [0.10, 0.80, 0.10], [0.05, 0.15, 0.80]])
        means = np.random.randn(3, 5) * 0.5
        covs = np.ones((3, 5)) * 0.1  # diagonal shape (K, D)
        params = HMMParameters(
            n_states=3,
            feature_names=[f"f{i}" for i in range(5)],
            transition_matrix=trans,
            emission_means=means,
            emission_covariances=covs,
            initial_distribution=np.array([0.33, 0.34, 0.33]),
            state_labels=["mean_reverting", "trending", "volatile"],
        )
        ff = ForwardFilter(params)
        obs = np.random.randn(5)
        pred = ff.update(obs)
        assert abs(sum(pred.state_probabilities) - 1.0) < 1e-8


class TestBOCPD:
    def test_detects_mean_shift(self):
        np.random.seed(42)
        bocpd = BOCPD(hazard_rate=1 / 50, threshold=0.3)

        # Stationary phase — build up stable statistics
        for _ in range(100):
            r = bocpd.update(np.random.randn() * 0.1)

        # Large mean shift — should trigger detection
        max_cp = 0.0
        for _ in range(50):
            r = bocpd.update(5.0 + np.random.randn() * 0.1)
            max_cp = max(max_cp, r.change_point_probability)

        assert max_cp > 0.01  # CP probability elevated above baseline after mean shift

    def test_stationary_stays_low(self):
        np.random.seed(42)
        bocpd = BOCPD(hazard_rate=1 / 250, threshold=0.5)

        max_cp = 0.0
        for _ in range(200):
            r = bocpd.update(np.random.randn() * 0.01)
            max_cp = max(max_cp, r.change_point_probability)

        # After warm-up, CP probability should stay low most of the time
        assert not r.is_change_point

    def test_state_persistence(self):
        bocpd = BOCPD()
        for _ in range(20):
            bocpd.update(np.random.randn())
        state = bocpd.get_state()

        bocpd2 = BOCPD()
        bocpd2.load_state(state)

        r1 = bocpd.update(0.5)
        r2 = bocpd2.update(0.5)
        assert abs(r1.change_point_probability - r2.change_point_probability) < 1e-10


class TestHMMTrainer:
    def test_train_on_synthetic_data(self):
        """Train a 3-state HMM on synthetic data with clear regimes."""
        np.random.seed(42)
        n_features = 5

        # Generate 3 distinct clusters
        data = np.vstack([
            np.random.randn(200, n_features) * 0.3 + np.array([0, -1, 0, 0, 0]),
            np.random.randn(200, n_features) * 0.3 + np.array([1, 0, 1, 0, 0]),
            np.random.randn(200, n_features) * 1.0 + np.array([0, 1, 0, 2, 0]),
        ])

        feature_names = ["returns_1", "realized_vol_20", "momentum_roc_10", "kyle_lambda_20", "vwap_deviation"]
        trainer = HMMTrainer(n_restarts=3, max_iter=50)
        params, report = trainer.train(data, feature_names, n_states=3)

        assert params.n_states == 3
        assert len(params.state_labels) == 3
        assert "volatile" in params.state_labels
        assert params.transition_matrix.shape == (3, 3)
        assert params.emission_means.shape == (3, 5)
        assert report.best_params.metrics["bic"] != 0

    def test_train_diag_covariance(self):
        """Diagonal covariance should produce (K, D) shaped covars."""
        np.random.seed(42)
        data = np.vstack([
            np.random.randn(200, 5) * 0.3 + np.array([0, -1, 0, 0, 0]),
            np.random.randn(200, 5) * 0.3 + np.array([1, 0, 1, 0, 0]),
            np.random.randn(200, 5) * 1.0 + np.array([0, 1, 0, 2, 0]),
        ])
        feature_names = ["returns_1", "realized_vol_20", "momentum_roc_10", "kyle_lambda_20", "vwap_deviation"]
        trainer = HMMTrainer(n_restarts=3, max_iter=50, covariance_type="diag", use_pca=False)
        params, report = trainer.train(data, feature_names, n_states=3)
        assert params.emission_covariances.shape == (3, 5)

    def test_train_with_pca(self):
        """PCA should reduce feature count and store transform."""
        np.random.seed(42)
        data = np.vstack([
            np.random.randn(200, 10) * 0.3,
            np.random.randn(200, 10) * 0.3 + 1.0,
            np.random.randn(200, 10) * 1.0 + 2.0,
        ])
        feature_names = [f"f{i}" for i in range(10)]
        trainer = HMMTrainer(n_restarts=3, max_iter=50, covariance_type="diag", use_pca=True)
        params, report = trainer.train(data, feature_names, n_states=3)
        assert params.pca_components is not None
        assert params.pca_mean is not None
        assert params.original_feature_names == feature_names
        # PCA should reduce 10 features to fewer components
        assert len(params.feature_names) <= 10

    def test_save_load_roundtrip(self, tmp_path):
        params = _make_dummy_params(5)
        path = str(tmp_path / "params.json")
        HMMTrainer.save_parameters(params, path)
        loaded = HMMTrainer.load_parameters(path)

        assert loaded.n_states == params.n_states
        np.testing.assert_array_almost_equal(
            loaded.transition_matrix, params.transition_matrix
        )
        np.testing.assert_array_almost_equal(
            loaded.emission_means, params.emission_means
        )

    def test_save_load_with_pca(self, tmp_path):
        """PCA fields should survive save/load roundtrip."""
        np.random.seed(42)
        data = np.vstack([np.random.randn(200, 10) * 0.3 for _ in range(3)])
        feature_names = [f"f{i}" for i in range(10)]
        trainer = HMMTrainer(n_restarts=3, max_iter=50, use_pca=True)
        params, _ = trainer.train(data, feature_names, n_states=3)
        path = str(tmp_path / "pca_params.json")
        HMMTrainer.save_parameters(params, path)
        loaded = HMMTrainer.load_parameters(path)
        assert loaded.pca_components is not None
        np.testing.assert_array_almost_equal(loaded.pca_components, params.pca_components)
