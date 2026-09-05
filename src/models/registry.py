import os
import joblib
import hopsworks
from src.config import HOPSWORKS_API_KEY
from src.utils.logger import logger

def upload_model(model, model_name: str, metrics: dict, description: str):
    """Saves model locally and uploads it to Hopsworks Model Registry."""
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set.")

    # 1. Save model locally
    local_dir = "saved_models"
    os.makedirs(local_dir, exist_ok=True)
    model_path = os.path.join(local_dir, "model.pkl")
    joblib.dump(model, model_path)
    
    logger.info(f"Model saved locally to {model_path}")

    # 2. Upload to Hopsworks
    logger.info("Authenticating with Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    mr = project.get_model_registry()

    logger.info(f"Uploading model '{model_name}' to registry...")
    hs_model = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        description=description
    )
    
    hs_model.save(local_dir)
    logger.info(f"Model '{model_name}' successfully registered!")

def download_model(model_name: str, version: int = None):
    """Downloads a model from the registry."""
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set.")

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    mr = project.get_model_registry()
    
    if version:
        model = mr.get_model(model_name, version=version)
    else:
        model = mr.get_model(model_name)
        
    local_dir = os.path.join("saved_models", "inference")
    os.makedirs(local_dir, exist_ok=True)
    
    model_path = model.download(local_dir)
    pkl_path = os.path.join(model_path, "model.pkl")
    
    return joblib.load(pkl_path)
