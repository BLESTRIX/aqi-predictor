import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestRegressor
from src.models.registry import HopsworksModelRegistry, upload_model


def test_registry_load_latest_model_falls_back_to_local(tmp_path, monkeypatch):
    """If Hopsworks is unreachable and no HOPSWORKS_API_KEY is set,
    load_latest_model should fall back to a local saved_models/model.pkl."""
    import joblib

    fallback_dir = tmp_path / "saved_models"
    fallback_dir.mkdir()
    fallback_path = fallback_dir / "model.pkl"

    model = RandomForestRegressor(n_estimators=5, random_state=42)
    X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
    y = [10, 20, 30]
    model.fit(X, y)
    joblib.dump(model, fallback_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.models.registry.HOPSWORKS_API_KEY", None)

    registry = HopsworksModelRegistry(model_name="test_model")
    loaded = registry.load_latest_model()

    assert loaded is not None
    preds = loaded.predict(X)
    assert len(preds) == 3


def test_registry_load_latest_model_returns_none_when_nothing_available(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.models.registry.HOPSWORKS_API_KEY", None)

    registry = HopsworksModelRegistry(model_name="nonexistent_model")
    result = registry.load_latest_model()

    assert result is None


@patch("src.models.registry.HOPSWORKS_API_KEY", None)
def test_upload_model_raises_without_api_key():
    model = RandomForestRegressor(n_estimators=5)
    with pytest.raises(ValueError):
        upload_model(model, "test_model", {"rmse": 1.0}, "test description")


@patch("src.models.registry.HOPSWORKS_API_KEY", "fake_key")
@patch("src.models.registry.hopsworks.login")
def test_upload_model_saves_local_pkl_before_upload(mock_login, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    mock_project = MagicMock()
    mock_mr = MagicMock()
    mock_login.return_value = mock_project
    mock_project.get_model_registry.return_value = mock_mr

    model = RandomForestRegressor(n_estimators=5, random_state=42)
    X = pd.DataFrame({"f1": [1, 2, 3]})
    y = [10, 20, 30]
    model.fit(X, y)

    upload_model(model, "test_model", {"rmse": 1.2}, "test")

    assert os.path.exists(os.path.join("saved_models", "model.pkl"))
    mock_mr.python.create_model.assert_called_once()