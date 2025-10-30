import os
import sys
from dotenv import load_dotenv, find_dotenv
import pathlib

load_dotenv(find_dotenv())
class Config:
    """Base configuration."""
    BASE_DIR = pathlib.Path(os.getenv("BASE_DIR", ".")).resolve()
    print(f"BASE_DIR set to: {BASE_DIR}")

    APP_NAME = os.getenv("APP_NAME")
    SECRET_KEY = os.getenv('SECRET_KEY')
    # Model configuration
    _default_model_dir = os.getenv("MODEL_DIR")
    MODEL_DIR = os.getenv("MODEL_DIR", _default_model_dir)
    print(f"MODEL_DIR set to: {MODEL_DIR}")

    DEVICE = os.getenv(
        "DEVICE", 
        "cuda:0" if __import__('torch').cuda.is_available() else "cpu"
    )

    DEBUG_MODE = bool(str(os.getenv("DEBUG")))

    # CORS 
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    # Monitoring
    ENABLE_METRICS = bool(str(os.getenv("ENABLE_METRICS")))
    LOG_LEVEL = os.getenv('LOG_LEVEL')
class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = 'INFO'

config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}