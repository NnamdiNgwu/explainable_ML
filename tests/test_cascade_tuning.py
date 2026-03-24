"""Tests for cascade threshold tuning (τ and τ₂ optimization)."""
import os
import sys
import importlib
import numpy as np
import pytest

# Direct-import cascade module (it imports from models.cybersecurity_transformer
# which needs torch; torch is available in the system env)
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo)

from models.cascade import tune_threshold, tune_tau2, rf_predict_proba


class MockRF:
    """Minimal RF mock that returns configurable probabilities."""

    def __init__(self, proba_map):
        """proba_map: list of (B, 3) arrays — one per call to predict_proba."""
        self._proba = proba_map
        self._idx = 0

    def predict_proba(self, X):
        """Return next pre-configured probability row."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        result = self._proba[self._idx: self._idx + len(X)]
        self._idx += len(X)
        return np.array(result)


class MockTransformer:
    """Minimal transformer mock for tune_tau2."""
    pass


class TestTuneThreshold:
    """Test τ (RF gate) threshold optimization."""

    def test_returns_float(self):
        """τ should be a float."""
        # 10 samples: 8 benign (class 0), 2 critical (class 2)
        # RF gives high confidence on critical, low on benign
        n = 10
        X_val = np.random.randn(n, 5)
        y_val = np.array([0, 0, 0, 0, 0, 0, 0, 0, 2, 2])

        # RF probabilities: benign samples have low max, critical have high max
        proba = []
        for y in y_val:
            if y == 0:
                proba.append([0.85, 0.10, 0.05])  # high P(benign), max=0.85
            else:
                proba.append([0.05, 0.10, 0.85])  # high P(critical), max=0.85
        rf = MockRF(proba)

        tau = tune_threshold(rf, None, X_val, None, y_val, 'cpu', 5, 1, 1)
        assert isinstance(tau, (float, np.floating))

    def test_tau_in_valid_range(self):
        """τ should be in (0, 1)."""
        n = 20
        X_val = np.random.randn(n, 5)
        y_val = np.array([0]*15 + [2]*5)

        proba = []
        for y in y_val:
            if y == 0:
                proba.append([0.7, 0.2, 0.1])
            else:
                proba.append([0.1, 0.1, 0.8])
        rf = MockRF(proba)

        tau = tune_threshold(rf, None, X_val, None, y_val, 'cpu', 5, 1, 1)
        assert 0.0 < tau < 1.0

    def test_separable_data_finds_good_threshold(self):
        """When classes are perfectly separable by RF confidence, τ should
        correctly separate benign from critical."""
        n = 20
        X_val = np.random.randn(n, 5)
        y_val = np.array([0]*15 + [2]*5)

        # Benign: max prob = 0.4, Critical: max prob = 0.9
        proba = []
        for y in y_val:
            if y == 0:
                proba.append([0.4, 0.35, 0.25])  # max = 0.4
            else:
                proba.append([0.05, 0.05, 0.9])  # max = 0.9
        rf = MockRF(proba)

        tau = tune_threshold(rf, None, X_val, None, y_val, 'cpu', 5, 1, 1)
        # τ should land between 0.4 and 0.9 to separate the groups
        assert 0.3 < tau < 0.95


class TestTuneTau2:
    """Test τ₂ (Transformer gate) threshold optimization."""

    def test_fallback_when_few_escalated(self):
        """Returns 0.5 when fewer than 5 samples are escalated."""
        y_val = np.array([0, 0, 0, 2])
        escalated_mask = np.array([False, False, False, True])  # only 1

        tau2 = tune_tau2(None, None, y_val, escalated_mask, 'cpu', 5, 1, 1)
        assert tau2 == 0.5

    def test_fallback_when_no_escalated(self):
        """Returns 0.5 when nothing is escalated."""
        y_val = np.array([0, 0, 0])
        escalated_mask = np.array([False, False, False])

        tau2 = tune_tau2(None, None, y_val, escalated_mask, 'cpu', 5, 1, 1)
        assert tau2 == 0.5


class TestRFPredictProba:
    """Test the RF prediction helper."""

    def test_single_row(self):
        """Should return a 1D probability vector for a single row."""
        proba = [[0.3, 0.2, 0.5]]
        rf = MockRF(proba)
        x = np.array([1.0, 2.0, 3.0])
        result = rf_predict_proba(rf, x)
        np.testing.assert_array_almost_equal(result, [0.3, 0.2, 0.5])

    def test_max_prob_gives_predicted_class(self):
        """The argmax of the probabilities should be the predicted class."""
        proba = [[0.1, 0.2, 0.7]]
        rf = MockRF(proba)
        x = np.array([1.0, 2.0])
        result = rf_predict_proba(rf, x)
        assert np.argmax(result) == 2
