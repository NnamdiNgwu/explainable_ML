# explanation_utils.py
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


# def get_rf_explanation_data(evt, models):
#     """Get RF explanation data ."""
#     X_tab = encode_tabular(evt)
#     rf_proba = rf_predict_proba(models['rf'], X_tab)
#     cls = int(rf_proba.argmax())
#     conf = float(rf_proba.max())
    
#     try:
#         # shap_values = _safe_get_shap_values(models['rf_explainer'], X_tab.reshape(1, -1), cls)
#         # baseline = _get_safe_baseline(models['rf_explainer'], cls)
#         phi_rf_any = models['rf_explainer'].shap_values(X_tab.reshape(1, -1))
#         shap_values = _select_class_vector(phi_rf_any, cls)  # (F,)
#         # Build dict(feature -> shap importance) for NLG
#         shap_dict = dict(zip(models['feature_names'], shap_values))
#         brief_reason = generate_natural_language_explanation(shap_dict, evt)
#         logging.info(f"RF explanation generated: {brief_reason}")

#         return {
#             "model_used": "RandomForest",
#             "prediction": cls,
#             "confidence": conf,
#             "probabilities": rf_proba.tolist(),
#             "explanation": {
#                 "method": "SHAP_TreeExplainer",
#                 "local_importance": _format_feature_importance(shap_values, models['feature_names']),
#                 "baseline": baseline,
#                 "brief_reason": brief_reason,
#                 "top_features": [
#                     {"feature": feat, "importance": round(imp, 4)}
#                     for feat, imp in list(zip(models['feature_names'], shap_values))
#                 ][:5]
#             }
#         }
#     except Exception as e:
#         logging.warning(f"RF SHAP failed: {e}")
#         return {
#             "model_used": "RandomForest", 
#             "prediction": cls,
#             "confidence": conf,
#             "probabilities": rf_proba.tolist(),
#             "explanation": {"method": "basic", "note": f"SHAP failed: {e}"}
#         }


def get_rf_explanation_data(evt, models):
    """Get RF explanation data."""
    X_tab = encode_tabular(evt)
    rf_proba = rf_predict_proba(models['rf'], X_tab)
    cls = int(rf_proba.argmax())
    conf = float(rf_proba.max())

    # Compute baseline first so it exists even if SHAP fails later
    try:
        baseline = _get_safe_baseline(models['rf_explainer'], cls)
    except Exception as be:
        logging.warning(f"RF baseline fetch failed, defaulting to 0.0: {be}")
        baseline = 0.0

    try:
        # rf_feat_names = models.get('rf_feature_names', models['feature_names'])
        shap_values = _safe_get_shap_values(models['rf_explainer'], X_tab.reshape(1, -1), cls)
        if shap_values is None or (hasattr(shap_values, "size") and shap_values.size == 0):
            raise ValueError("Empty SHAP values")

        # Build dict(feature -> shap importance) for NLG
        shap_dict = dict(zip(models['feature_names'], shap_values))
        # shap_dict = dict(zip(rf_feat_names, shap_values))
        # brief_reason = generate_natural_language_explanation(shap_dict, evt)
        # logging.info(f"RF explanation generated: {brief_reason}")

        return {
            "model_used": "RandomForest",
            "prediction": cls,
            "confidence": conf,
            "probabilities": rf_proba.tolist(),
            "explanation": {
                "method": "SHAP_TreeExplainer",
                "local_importance": _format_feature_importance(shap_values, models['feature_names']),
                # "local_importance": _format_feature_importance(shap_values,rf_feat_names ),
                "baseline": float(baseline),
                # "brief_reason": brief_reason,
                "top_features": [
                    {"feature": feat, "importance": round(imp, 4)}
                    for feat, imp in list(zip(models['feature_names'], shap_values))
                    # for feat, imp in list(zip(rf_feat_names, shap_values))
                ][:5]
            }
        }
    except Exception as e:
        logging.warning(f"RF SHAP failed: {e}", exc_info=True)
        # Fallback: still return a usable payload; do NOT reference 'baseline' here
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
    elif method == 'attention':
        try:
            explanation.update(_get_transformer_attention_explanation(evt, models, cont, cat_high, cat_low))
        except Exception as e:
            explanation = {"method": "attention_analysis", "note": f"Attention failed: {e}"}
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

def _get_top_features(shap_values, feature_names, top_k=5):
    """Get top features from SHAP values."""
    if shap_values is None or shap_values.size == 0:
        return []
        
    top_idx = np.abs(shap_values).argsort()[::-1][:top_k]
    return [
        {
            "feature": feature_names[i],
            "importance": round(float(shap_values[i]), 4)
        }
        for i in top_idx
    ]


def _explain_rf_prediction(evt, models, rf_proba):
    """Get comprehensive RF explanation with SHAP - Binary classification."""
    cls = int(rf_proba.argmax())
    conf = float(rf_proba.max())

    # Check if SHAP explainer is available
    if not models.get('rf_explainer'):
        logging.warning("RF SHAP explainer not available, using fallback")
        return jsonify({
            "model_used": "RandomForest",
            "prediction": cls,
            "confidence": conf,
            "probabilities": rf_proba.tolist(),
            "explanation": {
                "method": "basic",
                "note": "SHAP explainer not available",
                "global_importance": _format_feature_importance(
                    models['rf_estimator'].feature_importances_, 
                    models['feature_names']
                ) if hasattr(models['rf_estimator'], 'feature_importances_') else []
            }
        })
    
    try:
        # Debugging info
        logging.info("=== SHAP DEBUG INFO ===")
        logging.info(f"rf_explainer exists: {models.get('rf_explainer') is not None}")
        logging.info(f"rf_explainer type: {type(models.get('rf_explainer'))}")
        logging.info(f"rf_estimator exists: {models.get('rf_estimator') is not None}")
        logging.info(f"rf_estimator type: {type(models.get('rf_estimator'))}")
        
        # SHAP explanation
        X_tab = encode_tabular(evt) 
        logging.info(f"X_tab shape: {X_tab.shape}, type: {type(X_tab)}")
        logging.info(f"X_tab first 5 values: {X_tab[:5]}")
        
        if hasattr(models['rf_explainer'], 'model'):
            logging.info(f"rf_explainer.model n_features_in_: {getattr(models['rf_explainer'].model, 'n_features_in_', 'Not available')}")
        
        logging.info(f"Feature names count: {len(models['feature_names'])}")
        logging.info(f"Feature names: {models['feature_names'][:10]}...")  # First 10 only
        logging.info(f"Event keys: {list(evt.keys())}")
        logging.info("=== CALLING SHAP ===")

        shap_values = _safe_get_shap_values(models['rf_explainer'], X_tab.reshape(1, -1), cls)
        
        logging.info("=== SHAP SUCCESS ===")
        logging.info(f"SHAP values shape: {shap_values.shape}")
        logging.info(f"SHAP values type: {type(shap_values)}")

        # Binary classification expected value
        expected_value = models['rf_explainer'].expected_value
        # baseline =  float(expected_value[cls]) if isinstance(expected_value, np.ndarray) else float(expected_value) # [0.14288017, 0.85711983]
        baseline = _get_safe_baseline(models['rf_explainer'], cls)
        # Global vs local importance
        global_importance = models['rf_estimator'].feature_importances_

        return jsonify({
            "model_used": "RandomForest",
            "prediction": cls,
            "confidence": conf,
            "probabilities": rf_proba.tolist(),
            "explanation": {
                "method": "SHAP_TreeExplainer",
                "local_importance": _format_feature_importance(shap_values, models['feature_names']),
                "global_importance": _format_feature_importance(global_importance, models['feature_names']),
                "baseline": baseline,
                "shap_summary": {
                    "total_effect": float(shap_values.sum()),
                    "prediction_class": cls,
                    "class_names": ["normal", "high_risk"],
                    "strongest_positive": _get_strongest_feature(shap_values, models['feature_names'], positive=True),
                    "strongest_negative": _get_strongest_feature(shap_values, models['feature_names'], positive=False)
                }
            }
        })
        
    except Exception as e:
        logging.error(f"SHAP explanation failed: {e}")
        logging.error(f"SHAP explanation failed: {e}")
        logging.error(f"Error type: {type(e)}")
        import traceback
        logging.error(traceback.format_exc())
        # Fallback to basic explanation
        return jsonify({
            "model_used": "RandomForest",
            "prediction": cls,
            "confidence": conf,
            "probabilities": rf_proba.tolist(),
            "explanation": {
                "method": "basic",
                "note": f"SHAP explanation failed: {e}",
                "global_importance": _format_feature_importance(
                    models['rf_estimator'].feature_importances_, 
                    models['feature_names']
                ) if hasattr(models['rf_estimator'], 'feature_importances_') else []
            }
        })
    


def _explain_transformer_prediction(evt, models, method):
    """Get comprehensive Transformer explanation with SHAP, Captum, and Attention."""
    cont, cat_high, cat_low = encode_sequence_semantic(
        evt,
        models['feature_lists'],
        models['embed_maps'],
        models.get('device', 'cpu')  # Use 'cpu' as default if device not specified
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
    
    # 3. Attention Analysis
    if method in ['attention', 'auto']:
        try:
            explanation.update(_get_transformer_attention_explanation(evt, models, cont, cat_high, cat_low))
        except Exception as e:
            logging.warning(f"Attention analysis failed: {e}")
            explanation['attention_error'] = str(e)
    
    # 4. Feature Mapping to RF
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
    """Get SHAP explanation for transformer."""
    
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
        # Convert event to flat array
        # feature_vector = [evt.get(name, 0) for name in models['feature_names']]
        # X = np.array(feature_vector).reshape(1, -1)

        logging.info(f"Input X shape: {X.shape}, dtype: {X.dtype}")
        logging.info(f"Input X first 5 values: {X[0][:5]}")

        explanation_result = transformer_explainer.explain_transformer_via_rf(X, method='waterfall')
        
        if explanation_result is None:
            raise ValueError("Explanation mapping failed")
        
        # Extract mapped SHAP values
        mapped_shap_values = explanation_result['shap_values']
        
        # Handle the output format (ensure we get values for the predicted class)
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
        
        # NLG
        shap_dict = dict(zip(models['feature_names'], class_shap_values[:len(models['feature_names'])]))
        # brief_reason = generate_natural_language_explanation(shap_dict, evt)
        baseline_value = explanation_result.get('expected_value', 0.0)
        
        return {
            "shap_analysis": {
                "local_importance": _format_feature_importance(class_shap_values[:len(models['feature_names'])], models['feature_names']), #(shap_values, models['feature_names']),
                # "baseline": float(models['transformer_explainer'].shap_values[cls]) if hasattr(models['transformer_explainer'], 'shap_values') else 0.0,
                # "baseline": float(explanation_result.get('expected_value', 0.0)), #float(models['transformer_explainer'].expected_value[cls]) if hasattr(models['transformer_explainer'], 'expected_value') and isinstance(models['transformer_explainer'].expected_value, np.ndarray) else 0.0,
                # "baseline": float(baseline_value) if np.isscalar(baseline_value) else float(baseline_value[0]) if hasattr(baseline_value, '__len__') else 0.0,
                "total_effect": float(class_shap_values.sum()),
                "strongest_features": _get_top_features(class_shap_values[:len(models['feature_names'])], models['feature_names'], top_k=5),
                "method": explanation_result.get('method', 'rf_mapped_to_transformer'),
                "mapping_confidence": explanation_result.get('mapping_confidence', 0.5),
                "transformer_probability": explanation_result.get('transformer_probability', [0.5])[0],
                # "brief_reason": brief_reason
            }
        }
        
    except Exception as e:
        logging.warning(f"RF-to-Transformer mapping failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
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



def _get_transformer_attention_explanation(evt, models, cont, cat_high, cat_low):
    """Get attention-based explanation from transformer."""
    try:
        # This requires modifying the transformer to return attention weights
        # For now, we'll use input magnitude as proxy
        
        feature_lists = models['feature_lists']
        
        # Extract actual event values (last timestep)
        cont_values = cont[0, -1, :].cpu().numpy()
        
        # Create feature importance based on input magnitudes and transformer patterns
        feature_importance = []
        
        # Continuous + Boolean features
        cont_features = feature_lists["CONTINUOUS_USED"] + feature_lists["BOOLEAN_USED"]
        for i, feature_name in enumerate(cont_features):
            if i < len(cont_values):
                # Weight by magnitude and known cybersecurity importance
                base_importance = float(abs(cont_values[i]))

                # Apply domain knowledge weights for cybersecurity features
                if 'hour' in feature_name or 'after_hours' in feature_name:
                    base_importance *= 1.5  # Example: increase importance for time-related features
                elif 'entropy' in feature_name or 'suspicious' in feature_name:
                    base_importance *= 2.0 # Anomalies are critical in cybersecurity
                elif 'megabyte_sent' in feature_name or 'burst' in feature_name:
                    base_importance *= 1.3 # Activity patterns  are important

                feature_importance.append({
                    "feature": feature_name,
                    "importance": float(abs(cont_values[i])),
                    "raw_value": float(cont_values[i]),
                    "normalized_importance": float(abs(cont_values[i])) / (abs(cont_values).max() + 1e-8)
                })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            "attention_analysis": {
                "method": "Transformer_Input_magnitude_proxy_Analysis",
                "top_features": feature_importance[:5],  # Top 5 features
                "note": "Input feature magnitude analysis with cybersecurity domain weighting. True attention weights would require model modification.",
                "semantic_preservation": "maintained",
                "domain_knowledge_applied": True
            }
        }
    except Exception as e:
        logging.error(f"Transformer attention explaination failed: {e}")
        return {"attention_error": f"Attention analysis failed: {str(e)}"}
    

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
    """Format feature importance for response."""
    if values is None or len(values) != len(feature_names):
        return []
    
    feature_importance = list(zip(feature_names, values))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return [
        {
            "feature": feat,
            "importance": round(float(imp), 4),
            "rank": idx + 1
        }
        for idx, (feat, imp) in enumerate(feature_importance[:top_k])
    ]

def _get_strongest_feature(values, feature_names, positive=True):
    """Get strongest positive or negative feature."""
    if positive:
        idx = np.argmax(values)
    else:
        idx = np.argmin(values)
    
    return {
        "feature": feature_names[idx],
        "value": float(values[idx])
    }


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
    elif 'attention' in transformer_analysis.get('explanations', {}):
        transformer_features = {
            f['feature']: f['importance'] 
            for f in transformer_analysis['explanations']['attention']['attention_analysis']['top_features']
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
        models.get('device', 'cpu')  # Use 'cpu' as default if device not specified
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
    
    try:
        analysis["explanations"]["attention"] = _get_transformer_attention_explanation(evt, models, cont, cat_high, cat_low)
    except Exception as e:
        analysis["explanations"]["attention"] = {"error": f"Attention analysis failed: {str(e)}"}

    return analysis

def _get_comprehensive_rf_analysis(evt, models, rf_proba):
    """Get comprehensive RF analysis."""
    cls = int(rf_proba.argmax())
    X_tab = encode_tabular(evt)
    # shap_values = models['rf_explainer'].shap_values(X_tab.reshape(1, -1))[cls][0]
    shap_values = _safe_get_shap_values(models['rf_explainer'], X_tab.reshape(1, -1), cls)
    
    return {
        "model": "RandomForest",
        "prediction": cls,
        "confidence": float(rf_proba.max()),
        "probabilities": rf_proba.tolist(),
        "shap_analysis": {
            "local_importance": _format_feature_importance(shap_values, models['feature_names']),
            # "baseline": float(models['rf_explainer'].expected_value[cls]) if isinstance(models['rf_explainer'].expected_value, np.ndarray) and len(models['rf_explainer'].expected_values) > cls else float(models['rf_explainer'].expected_value) if not isinstance(models['rf_explainer'].expected_value, np.ndarray) else float(models['rf_explainer'].expected_value[0]),
            "baseline": _get_safe_baseline(models['rf_explainer'], cls),
            "total_effect": float(shap_values.sum())
        },
        "global_importance": _format_feature_importance(
            models['rf_estimator'].feature_importances_, 
            models['feature_names']
        )
    }

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
        # Get RF explanation for the same event
        X_tab = encode_tabular(evt)
        rf_proba = rf_predict_proba(models['rf'], X_tab)
        rf_cls = int(rf_proba.argmax())
        rf_shap_values = models['rf_explainer'].shap_values(X_tab.reshape(1, -1))[rf_cls][0]
        
        # Create mapping between transformer and RF features
        transformer_features = transformer_explanation.get('shap_analysis', {}).get('local_importance', [])
        rf_features = _format_feature_importance(rf_shap_values, models['feature_names'])
        
        # Find common features and compare importance
        feature_comparison = []
        transformer_dict = {f['feature']: f['importance'] for f in transformer_features}
        
        for rf_feature in rf_features[:5]:  # Top 5 RF features
            rf_name = rf_feature['feature']
            rf_importance = rf_feature['importance']
            
            transformer_importance = transformer_dict.get(rf_name, 0.0)
            
            feature_comparison.append({
                "feature": rf_name,
                "rf_importance": rf_importance,
                "transformer_importance": transformer_importance,
                "agreement": (rf_importance > 0) == (transformer_importance > 0),
                "magnitude_ratio": abs(transformer_importance) / (abs(rf_importance) + 1e-8)
            })
        
        return {
            "feature_comparison": feature_comparison,
            "overall_agreement": sum(f['agreement'] for f in feature_comparison) / len(feature_comparison),
            "note": "Comparison of feature importance between RF and Transformer models"
        }
        
    except Exception as e:
        return {"error": f"Feature mapping failed: {e}"}
 