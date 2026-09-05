import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
from src.utils.logger import logger

def calculate_metrics(y_true, y_pred) -> dict:
    """Calculates standard regression metrics."""
    # Ensure no NaNs before calculating
    valid_idx = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_cln = y_true[valid_idx]
    y_pred_cln = y_pred[valid_idx]
    
    mse = mean_squared_error(y_true_cln, y_pred_cln)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_cln, y_pred_cln)
    r2 = r2_score(y_true_cln, y_pred_cln)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2
    }

def calculate_shap_values(model, X_train: pd.DataFrame):
    """Calculates SHAP values for tree-based models."""
    logger.info("Calculating SHAP feature importance...")
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate for a sample to avoid massive compute times
    sample = X_train.sample(min(100, len(X_train)), random_state=42)
    shap_values = explainer.shap_values(sample)
    
    return shap_values, sample
