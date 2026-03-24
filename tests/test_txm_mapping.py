"""Tests for the TXM (Transformer eXplanation Mapper) core mapping logic.

Tests the probability-ratio scaling, class-wise SHAP selection,
fidelity computation, and mapping confidence — without requiring
Flask, trained models, or GPU.
"""
import os
import importlib
import numpy as np
import pytest

# ---- direct-import the mapper module to bypass Flask-dependent __init__ ----
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_fm_path = os.path.join(_repo, "src", "serving", "utils", "feature_mapping.py")
# We can't load feature_mapping.py directly because it imports from
# ..models.encoders (Flask-dependent).  Instead we test the core math
# by extracting the logic into small, focused helpers below.

_met_path = os.path.join(_repo, "src", "serving", "utils",
                         "cross_model_attribution_fidelity_metrics.py")
spec = importlib.util.spec_from_file_location("fidelity_metrics", _met_path)
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)

sign_fidelity = fm.sign_fidelity
rank_fidelity = fm.rank_fidelity
prob_monotonicity = fm.prob_monotonicity


# ---------------------------------------------------------------------------
# Replicate the core TXM math from TransformerExplanationMapper.explain_transformer_via_rf
# so we can test it in isolation (no Flask, no models, no GPU).
# ---------------------------------------------------------------------------

def txm_map(rf_shap: np.ndarray, p_rf_hat: np.ndarray, p_trans: np.ndarray,
            alpha_max: float = 10.0, eps: float = 1e-8):
    """Core TXM probability-ratio scaling.

    Args:
        rf_shap:   (B, F) RF SHAP values per instance.
        p_rf_hat:  (B,) reconstructed RF probability (margin + baseline).
        p_trans:   (B,) transformer probability for predicted class.

    Returns:
        mapped:  (B, F)  scaled SHAP values.
        alpha:   (B,)    scaling factors.
    """
    alpha = np.clip(p_trans / (p_rf_hat + eps), 0.0, alpha_max)
    mapped = rf_shap * alpha[:, None]
    return mapped, alpha


def select_classwise_shap_3d(shap_3d: np.ndarray, y_pred: np.ndarray):
    """Pick the SHAP slice for the predicted class.  shape: (B,F,C) -> (B,F)."""
    B, F, C = shap_3d.shape
    out = np.zeros((B, F), dtype=shap_3d.dtype)
    for i, c in enumerate(y_pred):
        out[i] = shap_3d[i, :, c]
    return out


def select_classwise_shap_list(shap_list, y_pred: np.ndarray):
    """Pick the SHAP slice for the predicted class.  list of C arrays each (B,F)."""
    parts = [shap_list[c][i] for i, c in enumerate(y_pred)]
    return np.stack(parts, axis=0)


def mapping_confidence(p_rf_hat: np.ndarray, p_trans: np.ndarray) -> float:
    """Correlation-based confidence, matching TransformerExplanationMapper._mapping_confidence."""
    if p_rf_hat.size == 1 or p_trans.size == 1:
        return float(max(0.0, 1.0 - abs(float(p_rf_hat[0]) - float(p_trans[0]))))
    corr = np.corrcoef(p_rf_hat.flatten(), p_trans.flatten())[0, 1]
    return float(max(0.0, corr)) if not np.isnan(corr) else 0.5


# ===========================================================================
# Tests
# ===========================================================================


class TestTXMProbabilityRatioScaling:
    """Test the core α = clip(P_trans / P̂_rf) mapping."""

    def test_identity_when_probs_equal(self):
        """α = 1 when P_trans == P̂_rf → mapped == rf_shap."""
        rf_shap = np.array([[0.5, -0.3, 0.2]])
        p_rf = np.array([0.7])
        p_trans = np.array([0.7])
        mapped, alpha = txm_map(rf_shap, p_rf, p_trans)
        np.testing.assert_allclose(alpha, [1.0], atol=1e-6)
        np.testing.assert_allclose(mapped, rf_shap, atol=1e-6)

    def test_upscale_when_transformer_more_confident(self):
        """α > 1 when P_trans > P̂_rf → magnitudes increase."""
        rf_shap = np.array([[0.4, -0.2, 0.1]])
        p_rf = np.array([0.3])
        p_trans = np.array([0.9])  # 3x more confident
        mapped, alpha = txm_map(rf_shap, p_rf, p_trans)
        assert alpha[0] == pytest.approx(3.0, abs=0.1)
        assert np.abs(mapped).sum() > np.abs(rf_shap).sum()

    def test_downscale_when_transformer_less_confident(self):
        """α < 1 when P_trans < P̂_rf → magnitudes decrease."""
        rf_shap = np.array([[0.4, -0.2, 0.1]])
        p_rf = np.array([0.9])
        p_trans = np.array([0.3])
        mapped, alpha = txm_map(rf_shap, p_rf, p_trans)
        assert alpha[0] < 1.0
        assert np.abs(mapped).sum() < np.abs(rf_shap).sum()

    def test_alpha_clipped_at_max(self):
        """α never exceeds alpha_max (default 10)."""
        rf_shap = np.array([[0.5, -0.3]])
        p_rf = np.array([0.01])  # very low
        p_trans = np.array([0.99])  # very high → ratio ~99
        mapped, alpha = txm_map(rf_shap, p_rf, p_trans)
        assert alpha[0] == 10.0

    def test_alpha_floored_at_zero(self):
        """α >= 0 even with edge-case inputs."""
        rf_shap = np.array([[0.5, -0.3]])
        p_rf = np.array([0.5])
        p_trans = np.array([0.0])  # zero transformer confidence
        mapped, alpha = txm_map(rf_shap, p_rf, p_trans)
        assert alpha[0] == 0.0
        np.testing.assert_array_equal(mapped, 0.0)

    def test_preserves_sign(self):
        """Mapping by positive α preserves feature attribution signs."""
        rf_shap = np.array([[0.5, -0.3, 0.0, 0.2, -0.1]])
        p_rf = np.array([0.4])
        p_trans = np.array([0.8])
        mapped, _ = txm_map(rf_shap, p_rf, p_trans)
        assert sign_fidelity(rf_shap[0], mapped[0]) == 1.0

    def test_preserves_rank(self):
        """Mapping by scalar α preserves feature importance ranking."""
        rf_shap = np.array([[0.6, -0.2, 0.1, 0.0, 0.3]])
        p_rf = np.array([0.4])
        p_trans = np.array([0.8])
        mapped, _ = txm_map(rf_shap, p_rf, p_trans)
        assert rank_fidelity(rf_shap[0], mapped[0], k=3) == 1.0

    def test_batch_independent(self):
        """Each row in a batch gets its own α."""
        rf_shap = np.array([[0.5, -0.3], [0.1, 0.2]])
        p_rf = np.array([0.3, 0.8])
        p_trans = np.array([0.9, 0.4])
        mapped, alpha = txm_map(rf_shap, p_rf, p_trans)
        assert alpha[0] != alpha[1]
        # First row upscaled, second downscaled
        assert np.abs(mapped[0]).sum() > np.abs(rf_shap[0]).sum()
        assert np.abs(mapped[1]).sum() < np.abs(rf_shap[1]).sum()

    def test_zero_rf_prob_handled(self):
        """P̂_rf = 0 doesn't cause division by zero (eps protects)."""
        rf_shap = np.array([[0.5, -0.3]])
        p_rf = np.array([0.0])
        p_trans = np.array([0.5])
        mapped, alpha = txm_map(rf_shap, p_rf, p_trans)
        assert np.isfinite(alpha[0])
        assert np.all(np.isfinite(mapped))


class TestClasswiseSHAPSelection:
    """Test picking the correct SHAP slice per predicted class."""

    def test_3d_array_selection(self):
        """(B, F, C) → (B, F) by selecting predicted class."""
        # 2 instances, 3 features, 3 classes
        shap_3d = np.array([
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ])
        y_pred = np.array([2, 0])  # instance 0 → class 2, instance 1 → class 0
        result = select_classwise_shap_3d(shap_3d, y_pred)
        np.testing.assert_array_equal(result[0], [0.3, 0.6, 0.9])  # class 2
        np.testing.assert_array_equal(result[1], [1.0, 4.0, 7.0])  # class 0

    def test_list_selection(self):
        """list of C arrays each (B, F) → (B, F)."""
        shap_list = [
            np.array([[0.1, 0.2], [1.0, 2.0]]),  # class 0
            np.array([[0.3, 0.4], [3.0, 4.0]]),  # class 1
            np.array([[0.5, 0.6], [5.0, 6.0]]),  # class 2
        ]
        y_pred = np.array([1, 2])
        result = select_classwise_shap_list(shap_list, y_pred)
        np.testing.assert_array_equal(result[0], [0.3, 0.4])  # class 1
        np.testing.assert_array_equal(result[1], [5.0, 6.0])  # class 2


class TestMappingConfidence:
    """Test the correlation-based confidence metric."""

    def test_identical_probs_high_confidence(self):
        p = np.array([0.3, 0.5, 0.7, 0.9])
        assert mapping_confidence(p, p) == pytest.approx(1.0, abs=1e-6)

    def test_single_sample_uses_distance(self):
        p_rf = np.array([0.8])
        p_trans = np.array([0.6])
        # 1 - |0.8 - 0.6| = 0.8
        assert mapping_confidence(p_rf, p_trans) == pytest.approx(0.8, abs=1e-6)

    def test_anticorrelated_clipped_to_zero(self):
        p_rf = np.array([0.1, 0.9, 0.1, 0.9])
        p_trans = np.array([0.9, 0.1, 0.9, 0.1])  # perfectly anticorrelated
        assert mapping_confidence(p_rf, p_trans) == 0.0


class TestTXMFidelityIntegration:
    """End-to-end: mapping → fidelity metrics."""

    def test_perfect_scaling_yields_perfect_sign_fidelity(self):
        rf_shap = np.array([[0.5, -0.3, 0.2, -0.1, 0.4]])
        p_rf = np.array([0.4])
        p_trans = np.array([0.8])
        mapped, _ = txm_map(rf_shap, p_rf, p_trans)
        # Positive α preserves all signs
        assert sign_fidelity(rf_shap[0], mapped[0]) == 1.0

    def test_perfect_scaling_yields_perfect_rank_fidelity(self):
        rf_shap = np.array([[0.5, -0.3, 0.2, -0.1, 0.4]])
        p_rf = np.array([0.4])
        p_trans = np.array([0.8])
        mapped, _ = txm_map(rf_shap, p_rf, p_trans)
        assert rank_fidelity(rf_shap[0], mapped[0], k=3) == 1.0

    def test_upscale_yields_positive_monotonicity(self):
        rf_shap = np.array([[0.3, 0.1, 0.2]])
        p_rf = np.array([0.3])
        p_trans = np.array([0.6])
        mapped, _ = txm_map(rf_shap, p_rf, p_trans)
        # Both attribution L1 and prob increased
        mono = prob_monotonicity(rf_shap[0], mapped[0], p_rf[0], p_trans[0])
        assert mono > 0
