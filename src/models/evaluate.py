import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
from src.utils.logger import logger

def calculate_metrics(y_true, y_pred) -> dict:
    """Calculates standard regression metrics."""
    valid_idx = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_cln = y_true[valid_idx]
    y_pred_cln = y_pred[valid_idx]

    mse = mean_squared_error(y_true_cln, y_pred_cln)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_cln, y_pred_cln)
    r2 = r2_score(y_true_cln, y_pred_cln)

    return {"rmse": rmse, "mae": mae, "r2_score": r2}


def explain_forecast(model, feature_vector: pd.DataFrame, background_df: pd.DataFrame,
                      target_cols: list, top_n: int = 5) -> dict:
    """
    Computes SHAP explanations for a single-row prediction, per forecast horizon.
    """
    explanations = {}

    if not hasattr(model, "estimators_"):
        logger.warning("Model has no .estimators_ (not a MultiOutputRegressor) — skipping SHAP.")
        return explanations

    for target_name, estimator in zip(target_cols, model.estimators_):
        try:
            # Use estimator.predict (a plain callable) rather than the estimator
            # object itself — this works uniformly across raw sklearn models,
            # XGBoost, AND sklearn Pipelines (e.g. StandardScaler + Ridge),
            # which shap.Explainer can't auto-detect as a model type directly.
            explainer = shap.Explainer(estimator.predict, background_df)
            shap_result = explainer(feature_vector)

            values = shap_result.values[0]
            feature_names = feature_vector.columns.tolist()

            contributions = sorted(
                zip(feature_names, values),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_n]

            explanations[target_name] = [
                {"feature": feat, "shap_value": float(val)} for feat, val in contributions
            ]
        except Exception as e:
            logger.warning(f"SHAP explanation failed for {target_name}: {e}")
            explanations[target_name] = []

    return explanations