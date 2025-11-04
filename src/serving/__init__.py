import os
from flask import Flask, jsonify
from flask_cors import CORS
import logging
import sys
from pathlib import Path
from models.safe_smote import SafeSMOTE

sys.modules[__name__].SafeSMOTE = SafeSMOTE
# Ensure SafeSMOTE is available in the app context

from .config.settings import config as Config
from .models.model_loader import ModelLoader
from .prediction.predictions import prediction_bp
from .explanations.explanation import explanation_bp
from .health_check.health_checks import health_bp

def create_app(config_name='None'):
    """Application factory pattern."""
    
    app = Flask(__name__)

    # Load configuration
    if config_name == 'None':
        config_name = os.getenv('FLASK_CONFIG', 'default')

    app.config.from_object(Config[config_name])
    
    # Initialize extensions
    CORS(app)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load models once at startup
    try:
        with app.app_context():
            app.ml_models = ModelLoader.load_all_models(app.config.get('MODEL_DIR'))
            logging.info("All models loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        raise
    
    # Register blueprints
    app.register_blueprint(health_bp, url_prefix='/api/v1/health')
    app.register_blueprint(prediction_bp, url_prefix='/api/v1')
    app.register_blueprint(explanation_bp, url_prefix='/api/v1')
    

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Endpoint not found",
            "path": error.description.split("'")[1] if "'" in error.description else "unknown",
            "method": "POST or GET"
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logging.error(f"Internal server error: {error}")
        return jsonify({
            "error": "Internal server error",
            "message": str(error) if app.debug else "An unexpected error occurred"
        }), 500
    
    return app