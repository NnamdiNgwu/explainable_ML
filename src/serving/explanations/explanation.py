"""Explanation Blueprint for ML Model Serving - Detailed and Comparative Explanations
"""
from flask import Blueprint, request, jsonify, current_app
import logging
import numpy as np
from ..utils.shap_utils import _safe_get_shap_values
from ..utils.explanation_utils import (
    get_rf_explanation_data,
    get_transformer_explanation_data,
    # _get_comprehensive_rf_analysis,
    _get_comprehensive_transformer_analysis,
    _analyze_feature_alignment,
    _get_safe_baseline
)
from ..models.encoders import encode_tabular
from ..models.cascade import rf_predict_proba, transformer_predict_proba
from ..utils.tau_serving_decision_helper import load_cascade_config, cascade_decision
from pathlib import Path


explanation_bp = Blueprint('explanation', __name__)


@explanation_bp.route('/explain', methods=['POST'])
def detailed_explanation():
    """Comprehensive explanation endpoint."""
    evt = request.get_json()
    if not evt:
        return jsonify({"error": "No JSON payload"}), 400
    
    # evt = handle_missing_features(evt)
    method = request.args.get('method', 'shap')  # 'shap', 'captum', 'attention', 'auto'
    
    try:
        models = current_app.ml_models
        # Ensure τ and τ2 are available (load from config if needed)
        if 'tau2' not in models:
            cfg_path = current_app.config.get('CASCADE_CONFIG_PATH', None)
            cfg_path = Path(cfg_path) if cfg_path else Path('config/cascade_config.json')
            try:
                tau_loaded, tau2_loaded, _cfg = load_cascade_config(cfg_path)
                models.setdefault('tau', float(tau_loaded))
                models['tau2'] = float(tau2_loaded)
            except Exception:
                # Fallback defaults
                models.setdefault('tau', 0.2)
                models.setdefault('tau2', 0.5)

        tau = float(models.get('tau', 0.2))
        tau2 = float(models.get('tau2', 0.5))

        # Determine which model would be used
        X_tab = encode_tabular(evt)
        rf_proba = rf_predict_proba(models['rf'], X_tab)
        rf_max = float(rf_proba.max())

        # Apply cascade with τ and τ2
        if rf_max < tau:
            # Early exit: RF-only (benign)
            rf_payload = get_rf_explanation_data(evt, models)
            rf_payload['cascade'] = {
                "tau": tau,
                "tau2": tau2,
                "rf_max": rf_max,
                "escalated": False,
                "decision": 0  # Benign
            }
            return jsonify(rf_payload)
        else:
            # Escalate to Transformer, decide with τ2
            trans_payload = get_transformer_explanation_data(evt, models, method=method)
            probs = trans_payload.get("probabilities", [])
            p_trans = float(probs[1] if isinstance(probs, list) and len(probs) > 1 else (max(probs) if probs else 0.0))
            decision = cascade_decision(rf_max, p_trans, tau, tau2)


            # Optional fidelity gate (advisory): if low, request IG as corroboration
            txm_fid = trans_payload.get("txm_fidelity", {}) or {}
            if txm_fid.get("sign_mean", 1.0) < 0.7 or txm_fid.get("rank_k10_mean", 1.0) < 0.2:
                try:
                    ig_payload = get_transformer_explanation_data(evt, models, method="captum")
                    trans_payload["ig_backup"] = {
                        "attributions": ig_payload.get("explanation", {}).get("captum_analysis", {}),
                        "used": True
                    }
                except Exception:
                    trans_payload["ig_backup"] = {"used": False}

            trans_payload['cascade'] = {
                "tau": tau,
                "tau2": tau2,
                "rf_max": rf_max,
                "p_trans": p_trans,
                "escalated": True,
                "decision": int(decision)  # 1=Malicious, 0=Benign
            }
            return jsonify(trans_payload)
            
    except Exception as e:
        logging.error(f"Explanation error: {e}")
        return jsonify({"error": str(e)}), 500



@explanation_bp.route('/explain/compare', methods=['POST'])
def compare_explanations():
    """Compare explanations from both models."""
    evt = request.get_json()
    if not evt:
        return jsonify({"error": "No JSON payload"}), 400
    
    # evt = handle_missing_features(evt)
    
    try:
        models = current_app.ml_models
        
        # Get predictions from both models
        rf_analysis = get_rf_explanation_data(evt, models)
        transformer_analysis = get_transformer_explanation_data(evt, models)
        # rf_analysis = _get_rf_analysis(evt, models)
        # transformer_analysis = _get_transformer_analysis(evt, models)
        
        # Cascade decision logic
        cascade_decision = {
            "would_use_rf": rf_analysis["confidence"] >= models['tau'],
            "threshold": models['tau'],
            "rf_confidence": rf_analysis["confidence"],
            "transformer_confidence": transformer_analysis["confidence"],
            "agreement": rf_analysis["prediction"] == transformer_analysis["prediction"]
            # "confidence_gap": abs(rf_analysis["confidence"] - transformer_analysis["confidence"])
        }
        
        return jsonify({
            "input_event": evt,
            "rf_analysis": rf_analysis,
            "transformer_analysis": transformer_analysis,
            "cascade_decision": cascade_decision,
            # "recommendation": _get_decision_recommendation(cascade_decision)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500



