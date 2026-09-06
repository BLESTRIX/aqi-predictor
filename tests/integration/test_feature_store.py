import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.features.feature_store import push_to_feature_store


@patch("src.features.feature_store.HOPSWORKS_API_KEY", None)
def test_push_to_feature_store_raises_without_api_key():
    df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=3), "us_aqi": [1, 2, 3]})
    with pytest.raises(ValueError):
        push_to_feature_store(df)


@patch("src.features.feature_store.HOPSWORKS_API_KEY", "fake_key")
@patch("src.features.feature_store.hopsworks.login")
def test_push_to_feature_store_calls_insert(mock_login):
    mock_project = MagicMock()
    mock_fs = MagicMock()
    mock_fg = MagicMock()

    mock_login.return_value = mock_project
    mock_project.get_feature_store.return_value = mock_fs
    mock_fs.get_or_create_feature_group.return_value = mock_fg
    mock_fg.features = []

    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=3),
        "location": ["islamabad"] * 3,
        "event_timestamp": [1, 2, 3],
        "us_aqi": [80, 90, 100],
    })

    push_to_feature_store(df)

    mock_login.assert_called_once()
    mock_fs.get_or_create_feature_group.assert_called_once()
    mock_fg.insert.assert_called_once()


@patch("src.features.feature_store.HOPSWORKS_API_KEY", "fake_key")
@patch("src.features.feature_store.hopsworks.login")
def test_push_to_feature_store_uses_configured_feature_group_name(mock_login):
    mock_project = MagicMock()
    mock_fs = MagicMock()
    mock_fg = MagicMock()

    mock_login.return_value = mock_project
    mock_project.get_feature_store.return_value = mock_fs
    mock_fs.get_or_create_feature_group.return_value = mock_fg
    mock_fg.features = []

    df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=2), "us_aqi": [1, 2]})
    push_to_feature_store(df)

    _, kwargs = mock_fs.get_or_create_feature_group.call_args
    assert kwargs["name"] == "islamabad_aqi_features"
    assert kwargs["primary_key"] == ["location", "event_timestamp"]