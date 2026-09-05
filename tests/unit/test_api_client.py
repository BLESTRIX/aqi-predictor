import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.data.api_client import OpenMeteoAPIClient


@patch("src.data.api_client.requests.get")
def test_fetch_air_quality(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "hourly": {
            "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
            "pm2_5": [12.5, 14.2],
            "pm10": [25.0, 28.1]
        }
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    client = OpenMeteoAPIClient(latitude=51.5074, longitude=-0.1278)
    df = client.fetch_air_quality(past_days=1, forecast_days=0)

    assert not df.empty
    assert "timestamp" in df.columns
    assert "pm2_5" in df.columns
    assert len(df) == 2


@patch("src.data.api_client.requests.get")
def test_fetch_weather(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "hourly": {
            "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
            "temperature_2m": [15.2, 14.8],
            "relative_humidity_2m": [80, 82]
        }
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    client = OpenMeteoAPIClient(latitude=51.5074, longitude=-0.1278)
    df = client.fetch_weather(past_days=1, forecast_days=0)

    assert not df.empty
    assert "timestamp" in df.columns
    assert "temperature_2m" in df.columns
    assert len(df) == 2
