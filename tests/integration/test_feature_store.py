import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.features.feature_store import HopsworksFeatureStore


def test_feature_store_offline_mode():
    fs = HopsworksFeatureStore(api_key=None)
    df_loaded = fs.load_features()
    assert isinstance(df_loaded, pd.DataFrame)
    assert df_loaded.empty


@patch("src.features.feature_store.HopsworksFeatureStore.get_feature_store")
def test_feature_store_save_mock(mock_get_fs):
    mock_fs = MagicMock()
    mock_fg = MagicMock()
    mock_fs.get_or_create_feature_group.return_value = mock_fg
    mock_get_fs.return_value = mock_fs

    fs = HopsworksFeatureStore(api_key="mock_key", project_name="mock_project")
    df_dummy = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3),
        "pm2_5": [10, 15, 20]
    })

    result = fs.save_features(df_dummy)
    assert result is True
    mock_fg.insert.assert_called_once()
