import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestRegressor
from src.models.registry import HopsworksModelRegistry


def test_registry_offline_save_and_load(tmp_path):
    registry = HopsworksModelRegistry(api_key=None)

    # Train dummy model
    X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
    y = pd.Series([10, 20, 30])
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    model.fit(X, y)

    metrics = {"rmse": 0.5, "mae": 0.4}
    saved = registry.save_model(model=model, metrics=metrics, input_example=X)

    # Offline mode returns False for Hopsworks upload, but saves local pkl file
    assert os.path.exists(os.path.join("models", f"{registry.model_name}.pkl"))

    loaded_model = registry.load_latest_model()
    assert loaded_model is not None
    preds = loaded_model.predict(X)
    assert len(preds) == 3
