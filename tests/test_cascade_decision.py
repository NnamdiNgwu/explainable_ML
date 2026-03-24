"""Tests for the two-stage cascade decision logic."""
import os
import json
import importlib
import pytest

# Import directly to avoid src.serving.__init__ pulling in Flask
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_mod_path = os.path.join(_repo, "src", "serving", "utils", "tau_serving_decision_helper.py")
spec = importlib.util.spec_from_file_location("tau_helper", _mod_path)
th = importlib.util.module_from_spec(spec)
spec.loader.exec_module(th)

cascade_decision = th.cascade_decision
load_cascade_config = th.load_cascade_config


class TestCascadeDecision:
    """Test the two-threshold cascade rule."""

    def test_low_rf_confidence_returns_benign(self):
        """Stage 1: max(P_RF) < τ → Benign regardless of transformer."""
        assert cascade_decision(p_rf=0.3, p_trans=0.99, tau=0.5, tau2=0.5) == 0

    def test_high_rf_and_high_trans_returns_malicious(self):
        """Stage 2: P_RF >= τ AND P_Trans >= τ₂ → Malicious."""
        assert cascade_decision(p_rf=0.8, p_trans=0.95, tau=0.5, tau2=0.9) == 1

    def test_high_rf_and_low_trans_returns_benign(self):
        """Stage 2: P_RF >= τ BUT P_Trans < τ₂ → Benign."""
        assert cascade_decision(p_rf=0.8, p_trans=0.5, tau=0.5, tau2=0.9) == 0

    def test_exact_tau_boundary_escalates(self):
        """P_RF == τ exactly should escalate (>= not >)."""
        assert cascade_decision(p_rf=0.5, p_trans=0.95, tau=0.5, tau2=0.9) == 1

    def test_exact_tau2_boundary_is_malicious(self):
        """P_Trans == τ₂ exactly should be Malicious (>= not >)."""
        assert cascade_decision(p_rf=0.8, p_trans=0.9, tau=0.5, tau2=0.9) == 1

    def test_fallback_when_transformer_none_high_rf(self):
        """No transformer available, very high RF → Malicious."""
        assert cascade_decision(p_rf=0.95, p_trans=None, tau=0.5, tau2=0.9) == 1

    def test_fallback_when_transformer_none_moderate_rf(self):
        """No transformer available, moderate RF → Benign (threshold 0.9)."""
        assert cascade_decision(p_rf=0.85, p_trans=None, tau=0.5, tau2=0.9) == 0

    def test_below_tau_ignores_none_transformer(self):
        """P_RF < τ → Benign even if transformer is None."""
        assert cascade_decision(p_rf=0.1, p_trans=None, tau=0.5, tau2=0.9) == 0


class TestLoadCascadeConfig:
    def test_loads_valid_config(self, tmp_path):
        cfg = {"tau": 0.6, "tau2": 0.85, "cont_dim": 36}
        config_path = tmp_path / "cascade_config.json"
        config_path.write_text(json.dumps(cfg))

        tau, tau2, loaded = load_cascade_config(config_path)
        assert tau == 0.6
        assert tau2 == 0.85
        assert loaded["cont_dim"] == 36

    def test_defaults_when_keys_missing(self, tmp_path):
        config_path = tmp_path / "cascade_config.json"
        config_path.write_text("{}")

        tau, tau2, _ = load_cascade_config(config_path)
        assert tau == 0.2  # default
        assert tau2 == 0.5  # default
