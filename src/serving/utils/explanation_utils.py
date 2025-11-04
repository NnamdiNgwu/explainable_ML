"""utils for model explanations."""
import logging
import numpy as np
import torch
from flask import current_app
from ..models.encoders import encode_tabular, encode_sequence_semantic
from ..models.cascade import rf_predict_proba, transformer_predict_proba
from .shap_utils import _safe_get_shap_values
from ..utils.feature_mapping import create_feature_vector_from_event

def _select_class_vector(phi_rf_any, cls_idx: int) -> np.ndarray:
    """Return (F,) vector for the chosen class from SHAP outputs that may be list/(F,C)/(1,F)."""
    if isinstance(phi_rf_any, list):
        v = np.asarray(phi_rf_any[cls_idx])
        return v[0] if v.ndim == 2 else v
    arr = np.asarray(phi_rf_any)
    if arr.ndim == 2:      # (1,F)
        return arr[0]
    if arr.ndim == 3:      # (1,F,C)
        return arr[0, :, cls_idx]
    if arr.ndim == 1:      # (F,)
        return arr
    return arr.reshape(-1)

def _expected_value_for_class(ev_any, cls_idx: int) -> float:
    if isinstance(ev_any, list):
        return float(np.asarray(ev_any)[cls_idx])
    if isinstance(ev_any, np.ndarray):
        if ev_any.ndim == 1 and ev_any.size > cls_idx:
            return float(ev_any[cls_idx])
        return float(ev_any.reshape(-1)[0])
    return float(ev_any)


def get_rf_explanation_data(evt, models):
    """Get RF explanation data."""
    X_tab = encode_tabular(evt)
    rf_proba = rf_predict_proba(models['rf'], X_tab)
    cls = int(rf_proba.argmax())
    conf = float(rf_proba.max())

    # Compute baseline
    try:
        baseline = _get_safe_baseline(models['rf_explainer'], cls)
    except Exception as be:
        logging.warning(f"RF baseline fetch failed, defaulting to 0.0: {be}")
        baseline = 0.0

    try:
        shap_values = _safe_get_shap_values(models['rf_explainer'], X_tab.reshape(1, -1), cls)
        if shap_values is None or (hasattr(shap_values, "size") and shap_values.size == 0):
            raise ValueError("Empty SHAP values")

        # ===== LOCAL IMPORTANCE (this event) =====
        local_importance = _format_feature_importance(shap_values, models['feature_names'])

        return {
            "model_used": "RandomForest",
            "prediction": cls,
            "confidence": conf,
            "probabilities": rf_proba.tolist(),
            "explanation": {
                "method": "SHAP_TreeExplainer",
                "local_importance": local_importance,
                "local_importance_type": "instance-level SHAP values (specific to this event)",
                "baseline": float(baseline),
                "total_effect": float(shap_values.sum())
            }
        }
    except Exception as e:
        logging.error(f"RF SHAP failed: {e}", exc_info=True)
        return {
            "model_used": "RandomForest",
            "prediction": cls,
            "confidence": conf,
            "probabilities": rf_proba.tolist(),
            "explanation": {
                "method": "basic",
                "note": f"SHAP failed: {e}"
            }
        }

def get_transformer_explanation_data(evt, models, method='basic'):
    """Get transformer explanation data ."""
    logging.info(f"=== TRANSFORMER EXPLANATION START ===")
    logging.info(f"Method: {method}")
    
    cont, cat_high, cat_low = encode_sequence_semantic(
        evt,
        models['feature_lists'],
        models['embed_maps'],
        models.get('device', 'cpu'))  
    transformer_proba = transformer_predict_proba(models['transformer'], cont, cat_high, cat_low, models['device'])
    cls = int(transformer_proba.argmax())
    conf = float(transformer_proba.max())
    
    explanation = {"method": method}

    # check for transformer explainer correctly
    transformer_explainer = models.get('transformer_explainer')
    has_explainer = transformer_explainer is not None

    logging.info(f"RF-to-Transformer mapper available: {has_explainer}")

    # if method == 'shap' and has_shap:
    if method == 'shap' and has_explainer:
        try:
            logging.info("=== CALLING RF-TO-TRANSFORMER MAPPING ===")
            explanation.update(_get_transformer_shap_explanation(evt, models, cls))
            logging.info("=== RF-TO-TRANSFORMER MAPPING SUCCESS ===")
        except Exception as e:
            logging.error(f"=== RF-TO-TRANSFORMER MAPPING FAILED ===")
            logging.error(f"Error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            explanation = {"method": "basic", "note": f"Explanation mapping failed: {e}"}
    else:
        explanation = {
            "method": "attention_analysis",
            "confidence_level": "approximate",
            "note": "Deep sequential analysis for uncertain cases"
        }
    
    return {
        "model_used": "CybersecurityTransformer",
        "prediction": cls,
        "confidence": conf,
        "probabilities": transformer_proba.tolist(),
        "explanation": explanation
    }


def _explain_transformer_prediction(evt, models, method):
    """Get comprehensive Transformer explanation with SHAP, Captum, and Attention."""
    cont, cat_high, cat_low = encode_sequence_semantic(
        evt,
        models['feature_lists'],
        models['embed_maps'],
        models.get('device', 'cpu')
    )
    transformer_proba = transformer_predict_proba(
        models['transformer'], 
        cont, cat_high, cat_low, 
        models['device']
    )
    
    cls = int(transformer_proba.argmax())
    conf = float(transformer_proba.max())
    
    explanation = {"method": method}
    
    # 1. SHAP Explanation
    if method in ['shap', 'auto'] and models.get('transformer_explainer'):
        try:
            explanation.update(_get_transformer_shap_explanation(evt, models, cls))
        except Exception as e:
            logging.warning(f"Transformer SHAP failed: {e}")
    
    # 2. Captum Explanation  
    if method in ['captum', 'auto'] and models.get('transformer_captum'):
        try:
            explanation.update(_get_transformer_captum_explanation(evt, models, cont, cat_high, cat_low, cls))
        except Exception as e:
            logging.warning(f"Transformer Captum explanation failed: {e}")
            explanation['captum_error'] = str(e)
    # 3. Feature Mapping to RF
    try:
        explanation['rf_mapping'] = _map_transformer_to_rf_features(evt, models, explanation)
    except Exception as e:
        logging.warning(f"RF mapping failed: {e}")

    return jsonify({
        "model_used": "CybersecurityTransformer",
        "prediction": cls,
        "confidence": conf,
        "probabilities": transformer_proba.tolist(),
        "explanation": explanation
    })


def _get_transformer_shap_explanation(evt, models, cls):
    """Get SHAP explanation for transformer via RF-to-Transformer mapping."""
    
    transformer_explainer = models.get('transformer_explainer', {})

    if not transformer_explainer:
        return {"shap_note": "SHAP explainer not available"}
    
    try:
        logging.info("=== USING RF-TO-TRANSFORMER MAPPING ===")

        feature_vector = create_feature_vector_from_event(
            evt,
            models['feature_names'],
            models['feature_lists'],
            models['embed_maps']
        )
        X = feature_vector.reshape(1, -1)  # Ensure X is 2D

        logging.info(f"Input X shape: {X.shape}, dtype: {X.dtype}")
        logging.info(f"Input X first 5 values: {X[0][:5]}")

        explanation_result = transformer_explainer.explain_transformer_via_rf(X, method='waterfall')
        
        if explanation_result is None:
            raise ValueError("Explanation mapping failed")
        
        # Extract mapped SHAP values
        mapped_shap_values = explanation_result['shap_values']
        
        # Handle the output format
        class_shap_values = mapped_shap_values[0] if mapped_shap_values.ndim > 1 else mapped_shap_values
        
        logging.info(f"class_shap_values type: {type(class_shap_values)}")
        logging.info(f"class_shap_values has detach: {hasattr(class_shap_values, 'detach')}")
        
        # Ensure it's a numpy array
        if not isinstance(class_shap_values, np.ndarray):
            class_shap_values = np.array(class_shap_values)
            
        # Flatten if needed
        if class_shap_values.ndim > 1:
            class_shap_values = class_shap_values.flatten()

        logging.info(f"Mapped SHAP values shape: {class_shap_values.shape}")
        logging.info(f"Mapping confidence: {explanation_result.get('mapping_confidence', 'N/A')}")
        logging.info("=== RF-TO-TRANSFORMER MAPPING SUCCESS ===")

        # ===== LOCAL IMPORTANCE (this event) =====
        local_importance = _format_feature_importance(
            class_shap_values[:len(models['feature_names'])], 
            models['feature_names'],
            top_k=5
        )

        return {
            "shap_analysis": {
                "local_importance": local_importance,
                "total_effect": float(class_shap_values.sum()),
                "method": explanation_result.get('method', 'rf_mapped_to_transformer'),
                "mapping_confidence": explanation_result.get('mapping_confidence', 0.5),
                "transformer_probability": explanation_result.get('transformer_probability', [0.5])[0],
            }
        }
        
    except Exception as e:
        logging.warning(f"RF-to-Transformer mapping failed: {e}")
        return {"shap_note": f"SHAP computation failed: {e}"}


def _get_transformer_captum_explanation(evt, models, cont, cat_high, cat_low, cls):
    """Get Captum-based explanations for transformer."""
    explanations = {}
    
    # Baseline tensors (all zeros)
    baseline_cont = torch.zeros_like(cont)
    baseline_cat_high = torch.zeros_like(cat_high) 
    baseline_cat_low = torch.zeros_like(cat_low)
    
    try:
        # Integrated Gradients
        if models.get('transformer_captum'):
            attributions = models['transformer_captum'].attribute(
                (cont, cat_high, cat_low),
                baselines=(baseline_cont, baseline_cat_high, baseline_cat_low),
                target=cls,
                n_steps=25
            )
            
            explanations['integrated_gradients'] = _process_captum_attributions(
                attributions, models['feature_lists']
            )
    except Exception as e:
        logging.warning(f"Integrated Gradients failed: {e}")

    return {"captum_analysis": explanations}

def _process_captum_attributions(attributions, feature_lists):
    """Process Captum attributions into readable format."""
    cont_attr, cat_high_attr, cat_low_attr = attributions
    
    # Extract last timestep attributions (actual event, not padding)
    cont_values = cont_attr[0, -1, :].cpu().numpy()
    
    # Map to feature names
    feature_importance = []
    cont_features = feature_lists["CONTINUOUS_USED"] + feature_lists["BOOLEAN_USED"]
    
    for i, feature_name in enumerate(cont_features):
        if i < len(cont_values):
            feature_importance.append({
                "feature": feature_name,
                "attribution": float(cont_values[i]),
                "abs_attribution": float(abs(cont_values[i]))
            })
    
    # Sort by absolute attribution
    feature_importance.sort(key=lambda x: x['abs_attribution'], reverse=True)
    
    return {
        "top_features": feature_importance[:2],
        "total_attribution": float(cont_values.sum()),
        "note": "Captum attributions for continuous/boolean features"
    }


def _format_feature_importance(values, feature_names, top_k=5):
    """
    Format SHAP feature importance (ranked by absolute impact).
    
    Args:
        values: SHAP values array (local importance)
        feature_names: Feature names list
        top_k: Number of top features to return
    
    Returns:
        list: Ranked features by absolute importance
    """
    if values is None or len(values) == 0:
        return []
    
    # Ensure numpy array
    if not isinstance(values, np.ndarray):
        values = np.asarray(values)
    
    # Flatten if needed
    if values.ndim > 1:
        values = values.flatten()
    
    # Handle length mismatch
    if len(values) != len(feature_names):
        min_len = min(len(values), len(feature_names))
        logging.warning(f"Length mismatch: {len(values)} values vs {len(feature_names)} names. Using first {min_len}")
        values = values[:min_len]
        feature_names = feature_names[:min_len]
    
    # Pair features with values and sort by absolute importance
    feature_pairs = list(zip(feature_names, values))
    sorted_pairs = sorted(feature_pairs, key=lambda x: abs(x[1]), reverse=True)
    
    # Return top-k ranked by impact
    return [
        {
            "feature": feat,
            "importance": round(float(imp), 4),
            "rank": idx + 1,
            # "direction": "↑ +risk" if imp > 0 else "↓ -risk" if imp < 0 else "→ neutral"
        }
        for idx, (feat, imp) in enumerate(sorted_pairs[:top_k])
    ]


def _analyze_feature_alignment(rf_analysis, transformer_analysis):
    """Analyze how well RF and Transformer feature importance align."""
    rf_features = {f['feature']: f['importance'] for f in rf_analysis['shap_analysis']['local_importance']}
    
    # Get transformer features from best available explanation
    transformer_features = {}
    if 'shap' in transformer_analysis.get('explanations', {}):
        transformer_features = {
            f['feature']: f['importance'] 
            for f in transformer_analysis['explanations']['shap'].get('shap_analysis', {}).get('local_importance', [])
        }
    
    # Calculate alignment metrics
    common_features = set(rf_features.keys()) & set(transformer_features.keys())
    
    if not common_features:
        return {"alignment_score": 0.0, "note": "No common features found"}
    
    # Calculate correlation of feature importance
    rf_values = [rf_features[f] for f in common_features]
    transformer_values = [transformer_features[f] for f in common_features]
    
    correlation = np.corrcoef(rf_values, transformer_values)[0, 1] if len(rf_values) > 1 else 0.0
    
    # Top feature agreement
    rf_top_5 = set([f['feature'] for f in rf_analysis['shap_analysis']['local_importance'][:2]])
    transformer_top_5 = set([f['feature'] for f in list(transformer_features.items())[:2]])
    top_feature_overlap = len(rf_top_5 & transformer_top_5) / 2.0
    
    return {
        "alignment_score": float(correlation if not np.isnan(correlation) else 0.0),
        "top_feature_overlap": top_feature_overlap,
        "common_features_count": len(common_features),
        "feature_comparison": [
            {
                "feature": feature,
                "rf_importance": rf_features[feature],
                "transformer_importance": transformer_features[feature],
                "agreement_direction": (rf_features[feature] > 0) == (transformer_features[feature] > 0)
            }
            for feature in sorted(common_features, key=lambda f: abs(rf_features[f]), reverse=True)[:10]
        ]
    }


def _get_comprehensive_transformer_analysis(evt, models):
    """Get comprehensive transformer analysis with all explanation methods."""
    cont, cat_high, cat_low = encode_sequence_semantic(
        evt,
        models['feature_lists'],
        models['embed_maps'],
        models.get('device', 'cpu')
    )
    transformer_proba = transformer_predict_proba(
        models['transformer'], cont, cat_high, cat_low, models['device']
    )
    
    cls = int(transformer_proba.argmax())
    
    analysis = {
        "model": "CybersecurityTransformer", 
        "prediction": cls,
        "confidence": float(transformer_proba.max()),
        "probabilities": transformer_proba.tolist(),
        "explanations": {}
    }
    
    # Add all available explanations
    if models.get('transformer_explainer'):
        try:
            logging.info("Using RF-to-Transformer explanation mapping")
            analysis["explanations"]["shap"] = _get_transformer_shap_explanation(evt, models, cls)
            logging.info("RF-to-Transformer mapping completed successfully")
        except Exception as e:
            analysis["explanations"]["shap"] = {"error": f"Explanation mapping failed: {str(e)}"}
            logging.error(f"RF-to-Transformer mapping failed: {e}")
    
   
    if models.get('transformer_captum'):
        try:
            analysis["explanations"]["captum"] = _get_transformer_captum_explanation(evt, models, cont, cat_high, cat_low, cls)
        except Exception as e:
            analysis["explanations"]["captum"] = {"error": f"Captum explanation failed: {str(e)}"}
    return analysis


def _get_safe_baseline(explainer, cls):
    """Safely get baseline value from SHAP explainer."""
    expected_value = explainer.expected_value
    
    if isinstance(expected_value, np.ndarray):
        if len(expected_value) > cls:
            baseline_val = expected_value[cls]
        else:
            baseline_val = expected_value[0]  # Fallback
    else:
        baseline_val = expected_value
    
    # Ensure it's a scalar
    if isinstance(baseline_val, np.ndarray):
        baseline_val = baseline_val.item()  # Convert 0-d array to scalar
    
    return float(baseline_val)


def _map_transformer_to_rf_features(evt, models, transformer_explanation):
    """Map transformer explanations to RF feature space for comparison."""
    try:
        # Check if RF explainer is available
        if not models.get('rf_explainer'):
            logging.warning("RF explainer not available for mapping")
            return {"note": "RF explainer not available"}
        
        # Get RF explanation for the same event
        X_tab = encode_tabular(evt)
        rf_proba = rf_predict_proba(models['rf'], X_tab)
        rf_cls = int(rf_proba.argmax())

        # Get RF SHAP values
        rf_shap_values = _safe_get_shap_values(models['rf_explainer'], X_tab.reshape(1, -1), rf_cls)
        
        if rf_shap_values is None or rf_shap_values.size == 0:
            logging.warning("RF SHAP values are empty")
            return {"note": "RF SHAP values unavailable"}
        
        # Create mapping between transformer and RF features
        rf_local_importance = _format_feature_importance(rf_shap_values, models['feature_names'])

        # Get transformer local importance
        transformer_local_importance = transformer_explanation.get('shap_analysis', {}).get('local_importance', [])
        
        if not transformer_local_importance:
            logging.warning("Transformer local importance is empty")
            return {"note": "Transformer explanation unavailable"}
        
        # create comparison dict
        transformer_dict = {f['feature']: f['importance'] for f in transformer_local_importance}

        # Compare feature rankings
        feature_comparison = []
        for rf_feature in rf_local_importance[:5]:  # Top 5 RF features
            rf_name = rf_feature['feature']
            rf_importance = rf_feature['importance']
            
            # Find matching transformer feature
            transformer_importance = transformer_dict.get(rf_name, 0.0)
            
            feature_comparison.append({
                "feature": rf_name,
                "rf_importance": rf_importance,
                "transformer_importance": transformer_importance,
                "agreement_direction": (rf_importance > 0) == (transformer_importance > 0),
                "magnitude_ratio": round(abs(transformer_importance) / (abs(rf_importance) + 1e-8), 4)
            })
        
        # Calculate overall agreement
        agreement_count = sum(1 for f in feature_comparison if f['agreement_direction'])
        overall_agreement = round(agreement_count / len(feature_comparison), 4) if feature_comparison else 0.0
        
        return {
            "feature_comparison": feature_comparison,
            "overall_agreement": overall_agreement,
            "note": "Comparison of feature importance between RF and Transformer models"
        }
        
    except Exception as e:
        logging.error(f"Feature mapping failed: {e}", exc_info=True)
        return {"error": f"Feature mapping failed: {str(e)}", "note": "Comparison unavailable"}
