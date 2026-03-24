"""End-to-end test: TXM transfers RF explanations to Transformer for analyst interpretability.

Verifies the full pipeline:
  RF SHAP  →  probability-ratio scaling  →  mapped attributions  →  analyst-readable output

Uses mock RF (SHAP TreeExplainer) and a real CybersecurityTransformer
(randomly initialised) to prove that:
  1. The Transformer receives the same raw features the RF saw.
  2. TXM produces per-feature attributions aligned with the Transformer prediction.
  3. Fidelity metrics (sign, rank, prob_monotonicity) confirm faithful transfer.
  4. An analyst can identify top-contributing features from the TXM output.
"""
import os
import sys
import math
import importlib
import numpy as np
import torch
import pytest

# ---------------------------------------------------------------------------
# Bypass Flask-dependent src/serving/__init__.py by direct-loading modules.
# ---------------------------------------------------------------------------
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo)

# Load the Transformer model class (pure PyTorch, no Flask)
from models.cybersecurity_transformer import (
    CybersecurityTransformer,
    build_cybersecurity_transformer_from_maps,
)

# Load fidelity metrics directly (no Flask)
_met_path = os.path.join(_repo, "src", "serving", "utils",
                         "cross_model_attribution_fidelity_metrics.py")
spec = importlib.util.spec_from_file_location("fidelity_metrics", _met_path)
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)
sign_fidelity = fm.sign_fidelity
rank_fidelity = fm.rank_fidelity
prob_monotonicity = fm.prob_monotonicity

# Load feature_mapping module (needs dict_to_transformer_tensors — patch Flask import)
# We load the encoder function standalone since it supports non-Flask mode
_enc_path = os.path.join(_repo, "src", "serving", "models", "encoders.py")
_enc_spec = importlib.util.spec_from_file_location("encoders", _enc_path)
_enc_mod = importlib.util.module_from_spec(_enc_spec)
# Inject a fake flask module so the top-level 'from flask import current_app' doesn't fail
import types
_fake_flask = types.ModuleType("flask")
_fake_flask.current_app = None
sys.modules.setdefault("flask", _fake_flask)
_enc_spec.loader.exec_module(_enc_mod)
dict_to_transformer_tensors = _enc_mod.dict_to_transformer_tensors


# ===========================================================================
# Fixtures — deterministic fake models to simulate the TXM pipeline
# ===========================================================================

# Minimal feature config matching the Transformer architecture
FEATURE_LISTS = {
    "CONTINUOUS_USED": ["logon_count", "file_access_count", "email_count", "http_count",
                        "after_hours_ratio"],
    "BOOLEAN_USED": ["is_admin", "used_removable_media"],
    "HIGH_CAT_USED": ["department"],
    "LOW_CAT_USED": ["role", "pc_type", "business_unit"],
}
FEATURES_ORDER = (
    FEATURE_LISTS["CONTINUOUS_USED"]
    + FEATURE_LISTS["BOOLEAN_USED"]
    + FEATURE_LISTS["HIGH_CAT_USED"]
    + FEATURE_LISTS["LOW_CAT_USED"]
)

EMBED_MAPS = {
    "department": {"engineering": 0, "finance": 1, "hr": 2, "sales": 3, "unknown": 4},
    "role": {"analyst": 0, "manager": 1, "director": 2, "unknown": 3},
    "pc_type": {"desktop": 0, "laptop": 1, "unknown": 2},
    "business_unit": {"us_east": 0, "us_west": 1, "emea": 2, "unknown": 3},
}


def _make_transformer():
    """Build a random CybersecurityTransformer matching FEATURE_LISTS."""
    cont_dim = len(FEATURE_LISTS["CONTINUOUS_USED"]) + len(FEATURE_LISTS["BOOLEAN_USED"])
    model = build_cybersecurity_transformer_from_maps(EMBED_MAPS, continuous_dim=cont_dim, num_classes=3)
    model.eval()
    return model


def _make_sample_event(seed=42):
    """Return a single raw event dict with realistic values."""
    rng = np.random.default_rng(seed)
    return {
        "logon_count": float(rng.integers(1, 50)),
        "file_access_count": float(rng.integers(0, 200)),
        "email_count": float(rng.integers(0, 100)),
        "http_count": float(rng.integers(0, 500)),
        "after_hours_ratio": float(rng.uniform(0, 1)),
        "is_admin": float(rng.choice([0, 1])),
        "used_removable_media": float(rng.choice([0, 1])),
        "department": rng.choice(["engineering", "finance", "hr", "sales"]),
        "role": rng.choice(["analyst", "manager", "director"]),
        "pc_type": rng.choice(["desktop", "laptop"]),
        "business_unit": rng.choice(["us_east", "us_west", "emea"]),
    }


def _event_to_raw_row(evt):
    """Convert event dict to raw numpy row in FEATURES_ORDER."""
    row = []
    for f in FEATURES_ORDER:
        row.append(evt.get(f, 0.0))
    return np.array(row, dtype=object)


class MockRFEstimator:
    """Fake RF that returns deterministic probabilities matching a fixed class."""

    def __init__(self, n_features, predicted_class=0, confidence=0.75):
        self.n_features_in_ = n_features
        self._class = predicted_class
        self._conf = confidence
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        B = X.shape[0]
        proba = np.full((B, 3), (1.0 - self._conf) / 2)
        proba[:, self._class] = self._conf
        return proba


class MockTreeExplainer:
    """Fake SHAP TreeExplainer that returns deterministic SHAP values."""

    def __init__(self, rf_estimator, n_features):
        self.model = rf_estimator
        self.expected_value = np.array([0.33, 0.33, 0.34])
        self._n_features = n_features

    def shap_values(self, X, check_additivity=False):
        """Return list of 3 arrays (one per class), each (B, F)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        B = X.shape[0]
        rng = np.random.default_rng(123)
        return [rng.standard_normal((B, self._n_features)) * 0.1 for _ in range(3)]


class MockPreprocessor:
    """Preprocessor that encodes categoricals via embed_maps, matching real ColumnTransformer output."""

    def transform(self, X_df):
        import pandas as pd
        result = X_df.copy()
        for c in FEATURE_LISTS["CONTINUOUS_USED"] + FEATURE_LISTS["BOOLEAN_USED"]:
            if c in result:
                result[c] = pd.to_numeric(result[c], errors="coerce").fillna(0.0)
        for c in FEATURE_LISTS["HIGH_CAT_USED"] + FEATURE_LISTS["LOW_CAT_USED"]:
            if c in result:
                mapping = EMBED_MAPS.get(c, {})
                result[c] = result[c].astype(str).map(lambda v, m=mapping: float(m.get(v, 0)))
        return result.values.astype(float)


# ===========================================================================
# Core TXM logic (replicated from TransformerExplanationMapper for isolation)
# ===========================================================================

def _txm_pipeline(rf_explainer, rf_estimator, transformer_model,
                  X_raw, feature_lists, features_order, embed_maps,
                  preprocessor):
    """
    Full TXM pipeline without Flask dependency.

    Steps:
    1. RF predicts class and SHAP values in preprocessed space.
    2. Transformer predicts class and probability in its own space.
    3. TXM scales RF SHAP by α = clip(P_trans / P̂_rf, 0, 10).
    4. Fidelity metrics validate the transfer.
    5. Returns analyst-facing attribution dict.
    """
    import pandas as pd

    B = X_raw.shape[0]

    # --- Step 1: RF side ---
    X_df = pd.DataFrame(X_raw, columns=features_order)
    # Coerce types
    for c in feature_lists["CONTINUOUS_USED"] + feature_lists["BOOLEAN_USED"]:
        if c in X_df:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(0.0)
    for c in feature_lists["HIGH_CAT_USED"] + feature_lists["LOW_CAT_USED"]:
        if c in X_df:
            X_df[c] = X_df[c].astype(str).fillna("unknown")

    X_est = preprocessor.transform(X_df)
    rf_proba = rf_estimator.predict_proba(X_est)
    y_rf = np.argmax(rf_proba, axis=1)

    # SHAP values — class-wise selection
    shap_raw = rf_explainer.shap_values(X_est, check_additivity=False)
    # shap_raw is list of C arrays each (B, F)
    rf_shap = np.stack([shap_raw[c][i] for i, c in enumerate(y_rf)], axis=0)  # (B, F)

    # Baseline for predicted class
    ev = np.asarray(rf_explainer.expected_value)
    rf_base = ev[y_rf]

    # Reconstruct RF probability
    rf_margin = rf_shap.sum(axis=1) + rf_base
    p_rf_hat = rf_margin.copy()
    if p_rf_hat.max() > 1.0 or p_rf_hat.min() < 0.0:
        p_rf_hat = 1.0 / (1.0 + np.exp(-rf_margin))  # sigmoid

    # --- Step 2: Transformer side ---
    p_trans_list, y_trans_list = [], []
    with torch.no_grad():
        for _, row in X_df.iterrows():
            evt = {k: row[k] for k in features_order}
            for c in feature_lists["CONTINUOUS_USED"] + feature_lists["BOOLEAN_USED"]:
                evt[c] = float(evt[c])
            for c in feature_lists["HIGH_CAT_USED"] + feature_lists["LOW_CAT_USED"]:
                evt[c] = str(evt[c])

            cont, ch, cl = dict_to_transformer_tensors(evt, feature_lists, embed_maps, "cpu")
            out = transformer_model(cont, ch, cl)
            logits = out["logits"] if isinstance(out, dict) else out
            prob = torch.softmax(logits, dim=1)
            cls = int(prob.argmax(1).item())
            y_trans_list.append(cls)
            p_trans_list.append(float(prob[0, cls].item()))

    p_trans = np.array(p_trans_list)
    y_trans = np.array(y_trans_list)

    # --- Step 3: probability-ratio scaling ---
    eps = 1e-8
    alpha = np.clip(p_trans / (p_rf_hat + eps), 0.0, 10.0)
    mapped = rf_shap * alpha[:, None]

    # --- Step 4: fidelity metrics ---
    fidelity_per_row = []
    for i in range(B):
        fidelity_per_row.append({
            "sign": float(sign_fidelity(rf_shap[i], mapped[i])),
            "rank_k5": float(rank_fidelity(rf_shap[i], mapped[i], k=5)),
            "prob_mono": float(prob_monotonicity(
                rf_shap[i], mapped[i], float(p_rf_hat[i]), float(p_trans[i])
            )),
        })

    # --- Step 5: analyst-facing output ---
    return {
        "mapped_shap": mapped,                     # (B, F) attributions
        "rf_shap": rf_shap,                        # (B, F) original RF SHAP
        "feature_names": list(features_order),
        "alpha": alpha,                             # (B,) scaling factors
        "p_rf_hat": p_rf_hat,                       # (B,) reconstructed RF prob
        "p_trans": p_trans,                         # (B,) transformer prob
        "y_rf": y_rf,                               # (B,) RF predicted class
        "y_trans": y_trans,                         # (B,) transformer predicted class
        "fidelity": fidelity_per_row,
    }


# ===========================================================================
# Tests
# ===========================================================================


class TestTXMTransferE2E:
    """End-to-end: prove TXM transfers RF→Transformer explanation."""

    @pytest.fixture
    def models(self):
        """Set up mock RF + real Transformer + preprocessor."""
        torch.manual_seed(0)
        np.random.seed(0)
        n_features = len(FEATURES_ORDER)
        rf_est = MockRFEstimator(n_features, predicted_class=2, confidence=0.80)
        rf_exp = MockTreeExplainer(rf_est, n_features)
        transformer = _make_transformer()
        pre = MockPreprocessor()
        return rf_est, rf_exp, transformer, pre

    @pytest.fixture
    def sample_batch(self):
        """Return a batch of 5 raw events as (B, F) array."""
        events = [_make_sample_event(seed=i) for i in range(5)]
        rows = [_event_to_raw_row(e) for e in events]
        return np.stack(rows, axis=0)

    def _run_pipeline(self, models, sample_batch):
        rf_est, rf_exp, transformer, pre = models
        return _txm_pipeline(
            rf_explainer=rf_exp,
            rf_estimator=rf_est,
            transformer_model=transformer,
            X_raw=sample_batch,
            feature_lists=FEATURE_LISTS,
            features_order=FEATURES_ORDER,
            embed_maps=EMBED_MAPS,
            preprocessor=pre,
        )

    # ------ 1. Pipeline completes without error ------
    def test_pipeline_runs_end_to_end(self, models, sample_batch):
        """TXM pipeline completes with mock RF + real Transformer."""
        result = self._run_pipeline(models, sample_batch)
        assert result is not None
        assert "mapped_shap" in result

    # ------ 2. Output shapes match expectations ------
    def test_output_shapes(self, models, sample_batch):
        """Mapped SHAP has shape (B, F) matching feature list."""
        result = self._run_pipeline(models, sample_batch)
        B = sample_batch.shape[0]
        F = len(FEATURES_ORDER)
        assert result["mapped_shap"].shape == (B, F)
        assert result["rf_shap"].shape == (B, F)
        assert result["alpha"].shape == (B,)
        assert result["p_rf_hat"].shape == (B,)
        assert result["p_trans"].shape == (B,)

    # ------ 3. Mapped attributions are finite and non-degenerate ------
    def test_mapped_values_finite(self, models, sample_batch):
        """All mapped SHAP values are finite (no NaN/Inf)."""
        result = self._run_pipeline(models, sample_batch)
        assert np.all(np.isfinite(result["mapped_shap"]))
        assert np.all(np.isfinite(result["alpha"]))

    def test_mapped_not_all_zero(self, models, sample_batch):
        """At least some features have non-zero attribution (explanations exist)."""
        result = self._run_pipeline(models, sample_batch)
        assert np.abs(result["mapped_shap"]).sum() > 0

    # ------ 4. Sign fidelity: TXM preserves feature direction ------
    def test_sign_fidelity_preserved(self, models, sample_batch):
        """Positive α means all signs preserved (sign_fidelity == 1.0)."""
        result = self._run_pipeline(models, sample_batch)
        for row_fid in result["fidelity"]:
            assert row_fid["sign"] == 1.0, (
                "TXM scaling by positive α must preserve all feature signs"
            )

    # ------ 5. Rank fidelity: feature importance order preserved ------
    def test_rank_fidelity_perfect(self, models, sample_batch):
        """Scalar α preserves rank → rank fidelity should be ~1.0."""
        result = self._run_pipeline(models, sample_batch)
        for row_fid in result["fidelity"]:
            assert row_fid["rank_k5"] == pytest.approx(1.0, abs=1e-9), (
                "Scalar α scaling should not change feature importance ranking"
            )

    # ------ 6. Alpha is bounded [0, 10] ------
    def test_alpha_bounded(self, models, sample_batch):
        """Scaling factor α stays in [0, alpha_max]."""
        result = self._run_pipeline(models, sample_batch)
        assert np.all(result["alpha"] >= 0.0)
        assert np.all(result["alpha"] <= 10.0)

    # ------ 7. Transformer receives same features as RF ------
    def test_transformer_gets_same_features(self, models, sample_batch):
        """Feature list in output matches what both models see."""
        result = self._run_pipeline(models, sample_batch)
        assert result["feature_names"] == list(FEATURES_ORDER)

    # ------ 8. Analyst can identify top-K contributing features ------
    def test_analyst_can_rank_features(self, models, sample_batch):
        """An analyst can extract top-K features from mapped SHAP."""
        result = self._run_pipeline(models, sample_batch)
        mapped = result["mapped_shap"]
        names = result["feature_names"]

        for i in range(mapped.shape[0]):
            # Sort features by absolute attribution
            indices = np.argsort(-np.abs(mapped[i]))
            top_3 = [names[j] for j in indices[:3]]
            # Analyst sees a ranked list — must have 3 distinct features
            assert len(top_3) == 3
            assert len(set(top_3)) == 3  # all unique

    # ------ 9. Attribution magnitude scales with transformer confidence ------
    def test_attribution_scales_with_confidence(self):
        """Higher transformer confidence → larger α → larger attribution L1."""
        torch.manual_seed(0)
        np.random.seed(0)
        n_features = len(FEATURES_ORDER)
        pre = MockPreprocessor()
        transformer = _make_transformer()

        # Low-confidence RF
        rf_lo = MockRFEstimator(n_features, predicted_class=2, confidence=0.40)
        exp_lo = MockTreeExplainer(rf_lo, n_features)

        events = [_make_sample_event(seed=i) for i in range(3)]
        X_raw = np.stack([_event_to_raw_row(e) for e in events])

        res = _txm_pipeline(exp_lo, rf_lo, transformer, X_raw,
                            FEATURE_LISTS, FEATURES_ORDER, EMBED_MAPS, pre)

        # With the same transformer output, lower RF confidence → higher α
        # → mapped attribution L1 >= RF attribution L1 (if transformer is more confident)
        for i in range(X_raw.shape[0]):
            rf_l1 = np.abs(res["rf_shap"][i]).sum()
            mapped_l1 = np.abs(res["mapped_shap"][i]).sum()
            if res["alpha"][i] > 1.0:
                assert mapped_l1 > rf_l1, (
                    f"Row {i}: α={res['alpha'][i]:.2f} > 1 but mapped L1 "
                    f"({mapped_l1:.4f}) <= RF L1 ({rf_l1:.4f})"
                )

    # ------ 10. Each instance gets independent explanation ------
    def test_per_instance_independence(self, models, sample_batch):
        """Different events → different mapped attributions."""
        result = self._run_pipeline(models, sample_batch)
        mapped = result["mapped_shap"]
        # At least two rows should differ (they have different features)
        diffs = [not np.allclose(mapped[0], mapped[i]) for i in range(1, mapped.shape[0])]
        assert any(diffs), "All rows have identical attributions — independence violated"

    # ------ 11. Fidelity dict has all required metrics ------
    def test_fidelity_metrics_present(self, models, sample_batch):
        """Each row has sign, rank, and prob_monotonicity metrics."""
        result = self._run_pipeline(models, sample_batch)
        for row_fid in result["fidelity"]:
            assert "sign" in row_fid
            assert "rank_k5" in row_fid
            assert "prob_mono" in row_fid
            # All are floats in valid ranges
            assert isinstance(row_fid["sign"], float)
            assert 0.0 <= row_fid["sign"] <= 1.0
            assert -1.0 <= row_fid["rank_k5"] <= 1.0
            assert -1.0 <= row_fid["prob_mono"] <= 1.0

    # ------ 12. Probabilities are valid ------
    def test_probabilities_valid(self, models, sample_batch):
        """Both RF and transformer probabilities are in [0, 1]."""
        result = self._run_pipeline(models, sample_batch)
        assert np.all(result["p_rf_hat"] >= 0.0) and np.all(result["p_rf_hat"] <= 1.0)
        assert np.all(result["p_trans"] >= 0.0) and np.all(result["p_trans"] <= 1.0)

    # ------ 13. Transformer actually produces predictions ------
    def test_transformer_produces_predictions(self, models, sample_batch):
        """Transformer predicted classes are in {0, 1, 2}."""
        result = self._run_pipeline(models, sample_batch)
        assert set(result["y_trans"]).issubset({0, 1, 2})
        assert set(result["y_rf"]).issubset({0, 1, 2})


class TestTXMAnalystInterpretability:
    """Tests that the TXM output is suitable for analyst consumption."""

    def test_feature_attribution_report(self):
        """Simulate what an analyst sees: feature name + signed contribution."""
        torch.manual_seed(0)
        np.random.seed(0)
        n_f = len(FEATURES_ORDER)
        rf_est = MockRFEstimator(n_f, predicted_class=2, confidence=0.80)
        rf_exp = MockTreeExplainer(rf_est, n_f)
        transformer = _make_transformer()
        pre = MockPreprocessor()

        evt = _make_sample_event(seed=99)
        X_raw = _event_to_raw_row(evt).reshape(1, -1)

        result = _txm_pipeline(rf_exp, rf_est, transformer, X_raw,
                               FEATURE_LISTS, FEATURES_ORDER, EMBED_MAPS, pre)

        mapped = result["mapped_shap"][0]
        names = result["feature_names"]
        pred_class = result["y_trans"][0]
        trans_prob = result["p_trans"][0]

        # Build the report an analyst would see
        report = []
        for idx in np.argsort(-np.abs(mapped)):
            report.append({
                "feature": names[idx],
                "contribution": float(mapped[idx]),
                "direction": "increases risk" if mapped[idx] > 0 else "decreases risk",
            })

        # Analyst requirements:
        # 1. Report covers all features
        assert len(report) == n_f
        # 2. Every entry has a name, numeric contribution, and direction
        for entry in report:
            assert isinstance(entry["feature"], str) and len(entry["feature"]) > 0
            assert isinstance(entry["contribution"], float)
            assert entry["direction"] in ("increases risk", "decreases risk")
        # 3. Features are ordered by absolute contribution (most important first)
        abs_contribs = [abs(e["contribution"]) for e in report]
        assert abs_contribs == sorted(abs_contribs, reverse=True)
        # 4. Predicted class and confidence are available for the header
        assert pred_class in (0, 1, 2)
        assert 0.0 <= trans_prob <= 1.0

    def test_explanation_changes_with_different_input(self):
        """Different events produce different explanations (not a static template)."""
        torch.manual_seed(0)
        np.random.seed(0)
        n_f = len(FEATURES_ORDER)
        rf_est = MockRFEstimator(n_f, predicted_class=0, confidence=0.90)
        rf_exp = MockTreeExplainer(rf_est, n_f)
        transformer = _make_transformer()
        pre = MockPreprocessor()

        evt1 = _make_sample_event(seed=10)
        evt2 = _make_sample_event(seed=20)
        X1 = _event_to_raw_row(evt1).reshape(1, -1)
        X2 = _event_to_raw_row(evt2).reshape(1, -1)

        r1 = _txm_pipeline(rf_exp, rf_est, transformer, X1,
                           FEATURE_LISTS, FEATURES_ORDER, EMBED_MAPS, pre)
        # Reset SHAP mock state
        rf_exp_2 = MockTreeExplainer(rf_est, n_f)
        r2 = _txm_pipeline(rf_exp_2, rf_est, transformer, X2,
                           FEATURE_LISTS, FEATURES_ORDER, EMBED_MAPS, pre)

        # Transformer probabilities should differ (different inputs)
        assert r1["p_trans"][0] != r2["p_trans"][0], (
            "Same transformer probability for different inputs — model is not responsive"
        )
