"""Tests for TXM cross-model attribution fidelity metrics."""
import sys
import os
import importlib
import numpy as np
import pytest

# Import the module directly to avoid src.serving.__init__ pulling in Flask
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_mod_path = os.path.join(_repo, "src", "serving", "utils", "cross_model_attribution_fidelity_metrics.py")
spec = importlib.util.spec_from_file_location("fidelity_metrics", _mod_path)
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)

sign_fidelity = fm.sign_fidelity
rank_fidelity = fm.rank_fidelity
prob_monotonicity = fm.prob_monotonicity


class TestSignFidelity:
    def test_perfect_agreement(self):
        rf = np.array([0.6, -0.2, 0.1, 0.3])
        txm = 2.0 * rf
        assert sign_fidelity(rf, txm) == 1.0

    def test_complete_disagreement(self):
        rf = np.array([0.5, -0.3, 0.2])
        txm = -rf
        assert sign_fidelity(rf, txm) == 0.0

    def test_partial_agreement(self):
        rf = np.array([0.5, -0.3, 0.2, -0.1])
        txm = np.array([0.5, 0.3, 0.2, -0.1])  # 3/4 agree
        assert sign_fidelity(rf, txm) == 0.75

    def test_zeros_agree(self):
        rf = np.array([0.0, 0.5, -0.3])
        txm = np.array([0.0, 0.1, -0.9])
        assert sign_fidelity(rf, txm) == 1.0

    def test_output_range(self):
        rng = np.random.default_rng(42)
        rf = rng.standard_normal(50)
        txm = rng.standard_normal(50)
        result = sign_fidelity(rf, txm)
        assert 0.0 <= result <= 1.0


class TestRankFidelity:
    def test_identical_vectors(self):
        rf = np.array([0.6, -0.2, 0.1, 0.0, 0.3])
        assert rank_fidelity(rf, rf, k=3) == 1.0

    def test_scaled_vector_preserves_rank(self):
        rf = np.array([0.6, -0.2, 0.1, 0.0, 0.3])
        txm = 2.0 * rf
        assert rank_fidelity(rf, txm, k=3) > 0.9

    def test_reversed_ranks(self):
        rf = np.array([0.9, 0.1, 0.5])
        txm = np.array([0.1, 0.9, 0.5])
        result = rank_fidelity(rf, txm, k=3)
        assert result < 1.0

    def test_k_clipped_to_size(self):
        rf = np.array([0.5, 0.3])
        txm = np.array([0.5, 0.3])
        assert rank_fidelity(rf, txm, k=10) == 1.0

    def test_output_range(self):
        rng = np.random.default_rng(42)
        rf = rng.standard_normal(20)
        txm = rng.standard_normal(20)
        result = rank_fidelity(rf, txm, k=5)
        assert -1.0 <= result <= 1.0


class TestProbMonotonicity:
    def test_consistent_increase(self):
        """When both attribution magnitude and probability increase, score > 0."""
        rf = np.array([0.3, 0.1, 0.2])
        txm = np.array([0.6, 0.2, 0.4])
        result = prob_monotonicity(rf, txm, p_rf=0.3, p_trans=0.6)
        assert result > 0

    def test_consistent_decrease(self):
        """When both decrease, product of negatives is positive."""
        rf = np.array([0.6, 0.2, 0.4])
        txm = np.array([0.3, 0.1, 0.2])
        result = prob_monotonicity(rf, txm, p_rf=0.6, p_trans=0.3)
        assert result > 0

    def test_inconsistent_change(self):
        """Attribution increases but probability decreases → negative."""
        rf = np.array([0.3, 0.1, 0.2])
        txm = np.array([0.6, 0.2, 0.4])
        result = prob_monotonicity(rf, txm, p_rf=0.6, p_trans=0.3)
        assert result < 0

    def test_output_in_tanh_range(self):
        rng = np.random.default_rng(42)
        rf = np.abs(rng.standard_normal(20))
        txm = np.abs(rng.standard_normal(20))
        result = prob_monotonicity(rf, txm, p_rf=0.5, p_trans=0.7)
        assert -1.0 <= result <= 1.0

    def test_zero_rf_prob_no_division_error(self):
        rf = np.array([0.3, 0.1])
        txm = np.array([0.6, 0.2])
        result = prob_monotonicity(rf, txm, p_rf=0.0, p_trans=0.5)
        assert np.isfinite(result)
