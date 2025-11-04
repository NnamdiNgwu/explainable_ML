"""Health check blueprint for the Flask application."""
from flask import Blueprint, jsonify, current_app
from datetime import datetime

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Docker."""
    try:
        # Check if models are loaded
        if not hasattr(current_app, 'ml_models') or not current_app.ml_models:
            return jsonify({"status": "unhealthy", "reason": "Models not loaded"}), 503
        
        required_models = ['rf', 'transformer', 'feature_names']
        missing = [m for m in required_models if m not in current_app.ml_models]
        
        if missing:
            return jsonify({"status": "unhealthy", "reason": f"Missing models: {missing}"}), 503
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "models_loaded": list(current_app.ml_models.keys())[:5]  # Sample
        }), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 503