import os
import json
from datetime import datetime, timezone
import pandas as pd
import joblib
import hopsworks

from src.config import CONFIG, HOPSWORKS_API_KEY

# Must match src/models/train.py's DEAD_COLS exactly, or the feature vector
# handed to the model at inference time will have different columns than
# what it was fit on, and sklearn/XGBoost will raise a shape/column mismatch.
DEAD_COLS = ["pm2_5","pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide",
             "ozone", "temperature", "humidity", "pressure", "wind_speed"]


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

    df = fg.read(read_options={"use_hive": False})
    if df.empty:
        raise ValueError(f"Feature group {fg_name} is empty.")

    df = df.sort_values(by="time", ascending=True)
    latest_row = df.tail(1).copy()

    current_aqi = latest_row["us_aqi"].iloc[0]

    metadata_cols = ["time", "location", "event_timestamp", "record_type"]
    target_cols = [col for col in latest_row.columns if col.startswith("target_")]
    cols_to_drop = metadata_cols + target_cols + DEAD_COLS

    feature_vector = latest_row.drop(columns=cols_to_drop, errors="ignore")

    return feature_vector, current_aqi


def load_model_from_registry(model_name="islamabad_aqi_model_24h"):
    """
    Connect to the Hopsworks Model Registry, download, and load the LATEST
    version of the specified model (by version number, not registry default).
    """
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set in the environment.")

    print(f"Connecting to Model Registry to fetch '{model_name}'...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=CONFIG["feature_store"]["project_name"])
    mr = project.get_model_registry()

    all_versions = mr.get_models(name=model_name)
    if not all_versions:
        raise ValueError(f"No versions of model '{model_name}' found in registry.")

    latest_model_meta = max(all_versions, key=lambda m: m.version)
    print(f"Found {len(all_versions)} version(s). Using latest: version {latest_model_meta.version}")

    model = mr.get_model(model_name, version=latest_model_meta.version)

    print("Downloading model artifact (using Hopsworks' managed cache)...")
    model_path = model.download()

    pkl_path = os.path.join(model_path, "model.pkl")
    if not os.path.exists(pkl_path):
        import glob
        pkl_files = glob.glob(os.path.join(model_path, "*.pkl")) + glob.glob(os.path.join(model_path, "*.joblib"))
        if not pkl_files:
            raise FileNotFoundError(f"Could not find a .pkl or .joblib model file in {model_path}")
        pkl_path = pkl_files[0]

    print(f"Loading model from '{pkl_path}' (version {latest_model_meta.version})...")
    loaded_model = joblib.load(pkl_path)

    return loaded_model, latest_model_meta.version


def get_background_sample(n: int = 100) -> pd.DataFrame:
    """
    Fetches a sample of recent historical feature rows to use as the SHAP
    explainer's reference distribution. Uses the same column filtering as
    get_latest_feature_vector() so columns line up with what the model expects.
    """
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=CONFIG["feature_store"]["project_name"])
    fs = project.get_feature_store()

    fg_name = CONFIG["feature_store"]["feature_group_name"]
    fg_version = CONFIG["feature_store"]["feature_group_version"]
    fg = fs.get_feature_group(name=fg_name, version=fg_version)

    df = fg.read(read_options={"use_hive": False})
    df = df.sort_values(by="time", ascending=True).tail(n).copy()

    metadata_cols = ["time", "location", "event_timestamp", "record_type"]
    target_cols = [col for col in df.columns if col.startswith("target_")]
    cols_to_drop = metadata_cols + target_cols + DEAD_COLS

    return df.drop(columns=cols_to_drop, errors="ignore")


def run_inference():
    """
    Fetch the latest features, load the latest model, generate 24h/48h/72h
    forecasts, and compute SHAP explanations for each horizon.
    """
    feature_vector, current_aqi = get_latest_feature_vector()
    model, model_version = load_model_from_registry()

    print("Running inference...")
    predictions = model.predict(feature_vector)
    preds = predictions[0] if len(predictions.shape) > 1 else predictions

    pred_24h = float(preds[0]) if len(preds) > 0 else None
    pred_48h = float(preds[1]) if len(preds) > 1 else None
    pred_72h = float(preds[2]) if len(preds) > 2 else None

    # SHAP explanations — best-effort; failures here shouldn't break the forecast
    shap_explanations = {}
    try:
        from src.models.evaluate import explain_forecast
        background_df = get_background_sample(n=100)
        target_cols = CONFIG["model"]["target_columns"]
        shap_explanations = explain_forecast(model, feature_vector, background_df, target_cols)
    except Exception as e:
        print(f"[WARN] SHAP explanation skipped: {e}")

    current_timestamp = datetime.now(timezone.utc).isoformat()

    result = {
        "execution_time": current_timestamp,
        "model_version": model_version,
        "current_aqi": float(current_aqi),
        "forecast_24h": pred_24h,
        "forecast_48h": pred_48h,
        "forecast_72h": pred_72h,
        "shap_explanations": shap_explanations,
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