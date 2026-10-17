"""
Configuration settings for the Boston Housing Price Prediction Application.
"""

import os

class Config:
    """Base application configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "boston-housing-ann-secret-key-2026")
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
    PORT = int(os.environ.get("PORT", 5050))
    HOST = os.environ.get("HOST", "0.0.0.0")
    
    # Upload configurations
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    
    # Model Artifacts
    MODEL_PATH = os.environ.get("MODEL_PATH", "ANN_regression.h5")
    SCALER_PATH = os.environ.get("SCALER_PATH", "scaler.joblib")
    METADATA_PATH = os.environ.get("METADATA_PATH", "model_metadata.json")

    # Rate Limiting
    RATELIMIT_DEFAULT = "120/minute"
    RATELIMIT_STORAGE_URI = "memory://"

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": ProductionConfig
}
