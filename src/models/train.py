import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.config import CONFIG, HOPSWORKS_API_KEY
from src.utils.logger import logger
from src.models.evaluate import calculate_metrics
from src.models.registry import upload_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# v2 feature group (AQICN, PM2.5-only) carries these columns as constant 0.0
# placeholders inherited from the old multi-pollutant/weather schema. They add
# no signal and can pick up spurious SHAP "importance" from floating point
# noise, so they're excluded from the model's feature set.
DEAD_COLS = ["pm2_5","pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide",
             "ozone", "temperature", "humidity", "pressure", "wind_speed"]

# ── Model comparison context (see notebooks/AQI_Model_Comparison.ipynb) ─────
#
# That notebook trained Persistence, Ridge, RandomForest, XGBoost, MLP, and
# SARIMA on the same v2 data/split used here and scored actual vs. predicted
# curves, tail-focused MAE (>150 AQI), and peak-lag — not just RMSE/MAE/R².
# Two results from that comparison are reflected below:
#
# 1. Ridge won on R² at every horizon (24h/48h/72h) — RandomForest and
#    XGBoost were consistently NOT the best performers on this dataset.
#    Nothing is hardcoded to "pick Ridge" here; the existing max-R² selection
#    logic already surfaces it correctly, we just now log it explicitly so
#    that stops being a surprise if it changes on a future retrain.
# 2. MLPRegressor (a genuinely different model family — neural net vs.
#    trees/linear) was competitive, beating both RandomForest and XGBoost at
#    the 48h/72h horizons in the notebook, so it's added below as a fourth
#    real candidate.
#
# SARIMA was also evaluated in the notebook (it did not win any horizon) but
# is intentionally NOT added as a production candidate here: this repo's
# serving path (src/models/predict.py) calls `model.predict(feature_vector)`
# on a single-row DataFrame, and SHAP explanations (src/models/evaluate.py)
# require `.estimators_` on a MultiOutputRegressor. SARIMA instead requires
# walk-forward sequential fitting against the full time series and produces
# forecasts very differently — it cannot be dropped into that contract
# without a separate inference path. If SARIMA (or another sequence model)
# is ever wanted in production, it needs its own serving code, not a slot in
# this candidates dict.


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


def compute_persistence_baseline(X_test: pd.DataFrame, y_test: pd.DataFrame, target_cols: list) -> dict:
    """
    'Tomorrow's AQI = today's AQI', evaluated on the same test split as every
    trained candidate. Not a deployable model — used purely as a sanity floor.
    Per the comparison notebook, this is a genuinely strong competitor at 24h,
    so any trained model that can't clearly beat it isn't earning its
    complexity, especially at longer horizons.
    """
    persistence_pred = np.repeat(X_test["us_aqi"].values.reshape(-1, 1), len(target_cols), axis=1)
    return evaluate_per_horizon(y_test, persistence_pred, target_cols, "Persistence (baseline)")


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

    # The feature columns are everything except targets, metadata, and dead columns
    metadata_cols = ["time", "location", "event_timestamp", "record_type"]
    feature_cols = [
        c for c in df.columns
        if c not in target_cols and c not in metadata_cols and c not in DEAD_COLS
    ]

    # Drop rows where target variables are NaN (future dates)
    df_clean = df.dropna(subset=target_cols).copy()

    X = df_clean[feature_cols]
    y = df_clean[target_cols]

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Feature columns: {feature_cols}")

    # Chronological train-test split (identical split reused across all models)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # ── Persistence baseline (logged, not a candidate for deployment) ──────
    persistence_metrics = compute_persistence_baseline(X_test, y_test, target_cols)

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

    # ── Model 3: Ridge Regression (statistical baseline) ─────────────────
    ridge_pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))
    ridge_model = MultiOutputRegressor(ridge_pipeline)

    candidates["Ridge"] = train_and_score(
        ridge_model, "Ridge", X_train, y_train, X_test, y_test, target_cols
    )

    # ── Model 4: MLP (neural network) ─────────────────────────────────────
    # Added per the model-comparison notebook: a genuinely different model
    # family from trees/linear regression, and competitive with RF/XGBoost
    # at the 48h/72h horizons there. Wrapped in the same
    # StandardScaler-then-model pipeline pattern as Ridge, since MLP is
    # scale-sensitive.
    mlp_base = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=3000,
        random_state=42,
        early_stopping=True,
    )
    mlp_pipeline = make_pipeline(StandardScaler(), mlp_base)
    mlp_model = MultiOutputRegressor(mlp_pipeline)

    candidates["MLP"] = train_and_score(
        mlp_model, "MLP", X_train, y_train, X_test, y_test, target_cols
    )

    # ── Model selection: pick the best by overall R² ─────────────────────
    best_name = max(candidates, key=lambda name: candidates[name][1]["r2_score"])
    best_model, best_metrics = candidates[best_name]

    logger.info("── Model comparison summary ──")
    logger.info(
        f"Persistence (baseline, not deployable): "
        f"{ {h: round(m['r2_score'], 3) for h, m in persistence_metrics.items()} }"
    )
    for name, (_, metrics) in candidates.items():
        logger.info(f"{name}: R2={metrics['r2_score']:.3f}  RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}")

    logger.info(f"Selected best model: {best_name} (R2={best_metrics['r2_score']:.3f})")

    # Sanity check against the persistence baseline, per-horizon. This won't
    # block an upload (a slightly-worse-than-persistence model may still be
    # the least-bad option available), but it should never pass silently.
    persistence_overall_r2 = np.mean([m["r2_score"] for m in persistence_metrics.values()])
    best_overall_per_horizon_r2 = np.mean([m["r2_score"] for m in
                                            evaluate_per_horizon(y_test, best_model.predict(X_test), target_cols, best_name).values()])
    if best_overall_per_horizon_r2 <= persistence_overall_r2:
        logger.warning(
            f"Selected model '{best_name}' (avg per-horizon R2={best_overall_per_horizon_r2:.3f}) "
            f"does NOT clearly beat the naive persistence baseline "
            f"(avg per-horizon R2={persistence_overall_r2:.3f}). Consider investigating "
            f"before relying on this forecast in production."
        )

    # Upload to registry
    model_name = "islamabad_aqi_model_24h"
    description = (
        f"{best_name} predicting 24h, 48h, and 72h AQI (v2: AQICN daily PM2.5). "
        f"Compared against RandomForest, XGBoost, Ridge, MLP, and a naive "
        f"persistence baseline (see notebooks/AQI_Model_Comparison.ipynb; "
        f"SARIMA was also evaluated there but is not a production candidate "
        f"due to incompatible serving architecture)."
    )
    upload_model(best_model, model_name, best_metrics, description)


if __name__ == "__main__":
    train_model()