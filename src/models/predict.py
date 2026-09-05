import os
import json
from datetime import datetime
import pandas as pd
import joblib
import hopsworks

from src.config import CONFIG, HOPSWORKS_API_KEY


def get_latest_feature_vector():
    """
    Connect to Hopsworks Feature Store and fetch the latest feature vector.
    Returns:
        feature_vector (pd.DataFrame): Single-row dataframe with features.
        current_aqi (float): The actual current AQI value.
    """
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set in the environment.")

    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project=CONFIG["feature_store"]["project_name"])
    fs = project.get_feature_store()

    fg_name = CONFIG["feature_store"]["feature_group_name"]
    fg_version = CONFIG["feature_store"]["feature_group_version"]
    
    print(f"Fetching feature group '{fg_name}' (version {fg_version})...")
    fg = fs.get_feature_group(name=fg_name, version=fg_version)
    
    # Read the dataset and sort to get the most recent row
    df = fg.read()
    if df.empty:
        raise ValueError(f"Feature group {fg_name} is empty.")
        
    df = df.sort_values(by="time", ascending=True)
    latest_row = df.tail(1).copy()
    
    current_aqi = latest_row["us_aqi"].iloc[0]
    
    # Isolate feature columns by dropping metadata and target columns
    metadata_cols = ["time", "location", "event_timestamp", "record_type"]
    target_cols = [col for col in latest_row.columns if col.startswith("target_")]
    cols_to_drop = metadata_cols + target_cols
    
    feature_vector = latest_row.drop(columns=cols_to_drop, errors="ignore")
    
    return feature_vector, current_aqi


def load_model_from_registry(model_name="islamabad_aqi_model_24h"):
    """
    Connect to the Hopsworks Model Registry, download, and load the specified model.
    """
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set in the environment.")

    print(f"Connecting to Model Registry to fetch '{model_name}'...")
    project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project=CONFIG["feature_store"]["project_name"])
    mr = project.get_model_registry()
    
    model = mr.get_model(model_name)
    
    local_dir = os.path.join("saved_models", "inference")
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading model artifact to '{local_dir}'...")
    model_path = model.download(local_dir)
    
    # Locate the joblib/pickle model file inside the downloaded artifact
    pkl_path = os.path.join(model_path, "model.pkl")
    if not os.path.exists(pkl_path):
        import glob
        pkl_files = glob.glob(os.path.join(model_path, "*.pkl")) + glob.glob(os.path.join(model_path, "*.joblib"))
        if not pkl_files:
            raise FileNotFoundError(f"Could not find a .pkl or .joblib model file in {model_path}")
        pkl_path = pkl_files[0]
        
    print(f"Loading model from '{pkl_path}'...")
    loaded_model = joblib.load(pkl_path)
    
    return loaded_model


def run_inference():
    """
    Fetch the latest features, load the model, and generate 24h, 48h, 72h forecasts.
    """
    feature_vector, current_aqi = get_latest_feature_vector()
    
    # Load model (Assuming the model predicts a multi-output array [24h, 48h, 72h])
    model = load_model_from_registry()
    
    print("Running inference...")
    predictions = model.predict(feature_vector)
    
    # Handle both 2D (batch) and 1D shapes for multi-output regressors
    preds = predictions[0] if len(predictions.shape) > 1 else predictions
    
    pred_24h = float(preds[0]) if len(preds) > 0 else None
    pred_48h = float(preds[1]) if len(preds) > 1 else None
    pred_72h = float(preds[2]) if len(preds) > 2 else None
    
    current_timestamp = datetime.utcnow().isoformat() + "Z"
    
    result = {
        "execution_time": current_timestamp,
        "current_aqi": float(current_aqi),
        "forecast_24h": pred_24h,
        "forecast_48h": pred_48h,
        "forecast_72h": pred_72h
    }
    
    return result


if __name__ == "__main__":
    try:
        print("=== Starting Batch Inference Pipeline ===")
        prediction_result = run_inference()
        print("\n=== Prediction Results ===")
        print(json.dumps(prediction_result, indent=2))
    except Exception as e:
        print(f"\n[ERROR] Inference pipeline failed: {e}")
