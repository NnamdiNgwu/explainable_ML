"""prediction endpoints."""
from flask import Blueprint, render_template, request, jsonify, current_app
import logging
import time
from ..utils.validation import validate_event,get_sample_event, handle_missing_features
from ..utils.shap_utils import _safe_get_shap_values
from ..utils.explanation_utils import get_rf_explanation_data, get_transformer_explanation_data
from ..models.cascade import cascade_predict
from ..models.encoders import encode_tabular


def _get_explanation(evt, result, current_app_context):
    """Helper to get explanation with graceful fallback.
    
    Args:
        evt: Event data
        result: Cascade prediction result
        current_app_context: Current app context with ML models
    
    Returns:
        dict: Explanation with SHAP analysis or fallback
    """
    try:
        start_time = time.time()
        method = request.args.get('method', 'shap')
        # # Always initialize so the variable exists even if downstream fails
        explanation_data = {}
        explanation = {"method": "basic", "note": "explanation not available"}

        model_used = result.get("model_used", "")
        if model_used in ("RandomForest", "RF", "RF-gate"):
            explanation_data = get_rf_explanation_data(evt, current_app.ml_models)
            explanation = explanation_data.get("explanation", explanation_data)
        else:
            explanation_data = get_transformer_explanation_data(evt, current_app.ml_models, method)
            explanation = explanation_data.get("explanation", explanation_data)

        return explanation

    except Exception as e:
        logging.warning(f"Explanation failed: {e}")
        return {
            "method": "shap",
            "note": f"Explanation unavailable: {str(e)}",
            "local_importance": []
        }



prediction_bp = Blueprint('prediction', __name__, template_folder='templates')

@prediction_bp.route("/")
def index():
    return render_template("main/main.html")

RISK_DESCRIPTIONS = {
    0: "Low - No insider threat detected",
    1: "Medium - Possible insider threat detected",
    2: "High - Potential insider threat detected"
}


@prediction_bp.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    evt = request.get_json()
    if not evt:
        return jsonify({"error": "No JSON payload provided"}), 400

    evt = handle_missing_features(evt)
    _ = validate_event(evt)  

    try:
        result  = cascade_predict(evt)
        # method  = request.args.get('method', 'shap')
        explanation = _get_explanation(evt, result, current_app.ml_models)
        print("Explanation obtained:", explanation)

        return jsonify({
            "risk": result["prediction"],
            "confidence": round(result["confidence"], 3),
            "model_used": result["model_used"],
            "risk_description": RISK_DESCRIPTIONS.get(result["prediction"], "Unknown"),
            "probabilities": {
                "normal": round(result["probabilities"][0], 3),
                "medium": round(result["probabilities"][1], 3),
                "high_risk": round(result["probabilities"][2], 3),
            },
            "explanation": explanation,
            "metadata": {
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "cascade_decision": result["model_used"] in ("RandomForest", "RF", "RF-gate"),
                "threshold": {"tau": current_app.ml_models['tau'], "tau2": current_app.ml_models['tau2']},
                "Brief_Reason": "This is a brief reason for the prediction."
            }
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e), "processing_time_ms": round((time.time() - start_time) * 1000, 2)}), 500



@prediction_bp.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint."""
    events = request.get_json()
    if not events or not isinstance(events, list):
        return jsonify({"error": "Expected list of events"}), 400
    
    if len(events) > 10000:  # Limit batch size
        return jsonify({"error": "Batch size limited to 10000 events"}), 400

    results = []
    start_time = time.time()
    
    for i, evt in enumerate(events):
        try:
            evt = handle_missing_features(evt)
            result = cascade_predict(evt)
            
            # get explanation for this event
            explanation = _get_explanation(evt, result, current_app.ml_models)
            results.append({
                "index": i,
                "risk": result["prediction"],
                "risk_label": ["LOW", "MEDIUM", "HIGH"][result["prediction"]],
                "risk_description": RISK_DESCRIPTIONS.get(result["prediction"], "Unknown"),
                "confidence": round(result["confidence"], 3),
                "model_used": result["model_used"],
                "probabilities": {
                    "normal": round(result["probabilities"][0], 3),
                    "medium": round(result["probabilities"][1], 3),
                    "high_risk": round(result["probabilities"][2], 3),
                },
                "explanation": explanation,
                "metadata": {
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                    "cascade_decision": result["model_used"] in ("RandomForest", "RF", "RF-gate"),
                    "threshold": {"tau": current_app.ml_models['tau'], "tau2": current_app.ml_models['tau2']},
                    "Brief_Reason": "This is a brief reason for the prediction."
                }
            })
        except Exception as e:
            logging.error(f"Batch prediction error at index {i}: {e}")
            results.append({"index": i, "error": str(e),"risk": None, "explanation": None})
    
    return jsonify({
        "results": results,
        "metadata": {
            "total_events": len(events),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "cascade_decisions": len([r for r in results if r.get("model_used") in ("RandomForest", "RF", "RF-gate")]),
            "threshold": {"tau": current_app.ml_models['tau'], "tau2": current_app.ml_models['tau2']},
            "successful_predictions": len([r for r in results if "error" not in r])
        }
    })
