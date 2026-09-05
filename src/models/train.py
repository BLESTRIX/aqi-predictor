import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from src.config import CONFIG, HOPSWORKS_API_KEY
from src.utils.logger import logger
from src.models.evaluate import calculate_metrics
from src.models.registry import upload_model

def get_training_data():
    """Fetches historical data directly from the Hopsworks feature group."""
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set.")

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    
    fg_name = CONFIG["feature_store"]["feature_group_name"]
    fg_version = CONFIG["feature_store"]["feature_group_version"]
    
    logger.info(f"Fetching feature group '{fg_name}' version {fg_version}...")
    fg = fs.get_feature_group(name=fg_name, version=fg_version)
    
    df = fg.read()
    return df

def train_model():
    df = get_training_data()
    
    # Sort and clean
    df = df.sort_values(by="time")
    
    # Identify target columns
    target_cols = [c for c in df.columns if c.startswith("target_")]
    
    # The feature columns are everything except target cols and metadata
    metadata_cols = ["time", "location", "event_timestamp", "record_type"]
    feature_cols = [c for c in df.columns if c not in target_cols and c not in metadata_cols]
    
    # Drop rows where target variables are NaN (future dates)
    df_clean = df.dropna(subset=target_cols).copy()
    
    X = df_clean[feature_cols]
    y = df_clean[target_cols]
    
    logger.info(f"Training data shape: {X.shape}")
    
    # Chronological train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train MultiOutput Random Forest
    logger.info("Training MultiOutput RandomForestRegressor...")
    base_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    multi_target_model = MultiOutputRegressor(base_model)
    
    multi_target_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = multi_target_model.predict(X_test)
    
    # Calculate combined metrics across all horizons
    metrics = calculate_metrics(y_test.values, y_pred)
    logger.info(f"Evaluation Metrics: {metrics}")
    
    # Upload to registry
    model_name = "islamabad_aqi_model_24h"
    description = "RandomForest predicting 24h, 48h, and 72h AQI."
    upload_model(multi_target_model, model_name, metrics, description)

if __name__ == "__main__":
    train_model()
