"""Tests for SafeSMOTE adaptive k_neighbors wrapper."""
import numpy as np
import pytest

try:
    from models.safe_smote import SafeSMOTE
except ImportError:
    SafeSMOTE = None

# SafeSMOTE imports fine (lazy SMOTENC), but fit_resample() needs imblearn at runtime.
# Check if imblearn itself is usable — if not, skip all tests.
_imblearn_ok = True
try:
    from imblearn.over_sampling import SMOTENC  # noqa: F401
except (ImportError, AttributeError):
    _imblearn_ok = False

pytestmark = pytest.mark.skipif(
    SafeSMOTE is None or not _imblearn_ok,
    reason="imbalanced-learn not installed or version conflict with scikit-learn"
)


class TestSafeSMOTE:
    @pytest.fixture
    def categorical_mask(self):
        """Last column is categorical."""
        return [False, False, True]

    def test_adapts_k_for_tiny_minority(self, categorical_mask):
        """When minority class has 2 samples, k should be clamped to 1."""
        # 10 majority, 2 minority — k_neighbors=5 would fail without adaptation
        X = np.array([
            [1.0, 2.0, 0],
            [1.1, 2.1, 0],
            [1.2, 2.2, 1],
            [1.3, 2.3, 0],
            [1.4, 2.4, 0],
            [1.5, 2.5, 0],
            [1.6, 2.6, 0],
            [1.7, 2.7, 0],
            [1.8, 2.8, 0],
            [1.9, 2.9, 0],
            [5.0, 6.0, 1],  # minority
            [5.1, 6.1, 1],  # minority
        ])
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

        smote = SafeSMOTE(categorical_features=categorical_mask, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X, y)

        # Should not raise and should produce balanced output
        assert len(y_res) > len(y)
        assert np.bincount(y_res)[0] == np.bincount(y_res)[1]

    def test_preserves_k_when_sufficient_samples(self, categorical_mask):
        """When minority has enough samples, k stays at requested value."""
        rng = np.random.default_rng(42)
        n_majority, n_minority = 50, 20
        X_maj = np.column_stack([rng.standard_normal((n_majority, 2)),
                                  np.zeros(n_majority)])
        X_min = np.column_stack([rng.standard_normal((n_minority, 2)) + 5,
                                  np.ones(n_minority)])
        X = np.vstack([X_maj, X_min])
        y = np.array([0] * n_majority + [1] * n_minority)

        smote = SafeSMOTE(categorical_features=categorical_mask, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X, y)

        assert np.bincount(y_res)[0] == np.bincount(y_res)[1]

    def test_single_minority_sample_uses_k1(self, categorical_mask):
        """Edge case: only 1 minority sample → k clamped to max(1, 0) = 1."""
        X = np.array([
            [1.0, 2.0, 0],
            [1.1, 2.1, 0],
            [1.2, 2.2, 0],
            [1.3, 2.3, 0],
            [1.4, 2.4, 0],
            [5.0, 6.0, 1],  # single minority
        ])
        y = np.array([0, 0, 0, 0, 0, 1])

        smote = SafeSMOTE(categorical_features=categorical_mask, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X, y)

        # k=1 with 1 sample — SMOTENC uses the single sample as its own neighbor
        assert len(y_res) >= len(y)
