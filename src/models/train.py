import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.config import CONFIG, HOPSWORKS_API_KEY
from src.utils.logger import logger
from src.models.evaluate import calculate_metrics
from src.models.registry import upload_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def get_training_data():
    """Fetches historical data directly from the Hopsworks feature group."""
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set.")

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=CONFIG["feature_store"]["project_name"])
    fs = project.get_feature_store()

    fg_name = CONFIG["feature_store"]["feature_group_name"]
    fg_version = CONFIG["feature_store"]["feature_group_version"]

    logger.info(f"Fetching feature group '{fg_name}' version {fg_version}...")
    fg = fs.get_feature_group(name=fg_name, version=fg_version)

    df = fg.read()
    return df


def evaluate_per_horizon(y_test: pd.DataFrame, y_pred: np.ndarray, target_cols: list, model_label: str) -> dict:
    """Computes and logs metrics separately for each forecast horizon."""
    per_horizon = {}
    for i, col in enumerate(target_cols):
        m = calculate_metrics(y_test.values[:, i], y_pred[:, i])
        per_horizon[col] = m
        logger.info(f"[{model_label}] {col}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2_score']:.3f}")
    return per_horizon


def train_and_score(model, model_label, X_train, y_train, X_test, y_test, target_cols):
    """Fits a model, scores it overall and per-horizon, returns everything needed for comparison."""
    logger.info(f"Training {model_label}...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    overall = calculate_metrics(y_test.values, pred)
    logger.info(f"[{model_label}] Overall: {overall}")
    evaluate_per_horizon(y_test, pred, target_cols, model_label)

    return model, overall


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

    # Chronological train-test split (identical split reused across all models)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    candidates = {}

    # ── Model 1: Random Forest ────────────────────────────────────────────
    rf_base = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model = MultiOutputRegressor(rf_base)
    candidates["RandomForest"] = train_and_score(
        rf_model, "RandomForest", X_train, y_train, X_test, y_test, target_cols
    )

    # ── Model 2: XGBoost ──────────────────────────────────────────────────
    xgb_base = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model = MultiOutputRegressor(xgb_base)
    candidates["XGBoost"] = train_and_score(
        xgb_model, "XGBoost", X_train, y_train, X_test, y_test, target_cols
    )

    # ── Model 3: Ridge Regression (statistical baseline) ─────────────────s
    ridge_pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))
    ridge_model = MultiOutputRegressor(ridge_pipeline)

    candidates["Ridge"] = train_and_score(
        ridge_model, "Ridge", X_train, y_train, X_test, y_test, target_cols
    )

    # ── Model selection: pick the best by overall R² ─────────────────────
    best_name = max(candidates, key=lambda name: candidates[name][1]["r2_score"])
    best_model, best_metrics = candidates[best_name]

    logger.info("── Model comparison summary ──")
    for name, (_, metrics) in candidates.items():
        logger.info(f"{name}: R2={metrics['r2_score']:.3f}  RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}")

    logger.info(f"Selected best model: {best_name} (R2={best_metrics['r2_score']:.3f})")

    # Upload to registry
    model_name = "islamabad_aqi_model_24h"
    description = f"{best_name} predicting 24h, 48h, and 72h AQI. Compared against RandomForest, XGBoost, and Ridge."
    upload_model(best_model, model_name, best_metrics, description)


if __name__ == "__main__":
    train_model()