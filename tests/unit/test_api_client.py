import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.data.api_client import parse_aqicn_payload, fetch_raw_air_quality


def _fake_aqicn_payload():
    """Mimics the real AQICN API response shape used by parse_aqicn_payload."""
    return {
        "status": "ok",
        "data": {
            "aqi": 145,
            "time": {"s": "2026-01-01 12:00:00"},
            "iaqi": {
                "pm25": {"v": 55.0},
                "pm10": {"v": 80.0},
                "no2": {"v": 12.0},
                "so2": {"v": 5.0},
                "co": {"v": 0.4},
                "o3": {"v": 30.0},
                "t": {"v": 20.0},
                "h": {"v": 45.0},
                "p": {"v": 1012.0},
                "w": {"v": 3.5},
            },
            "forecast": {
                "daily": {
                    "pm25": [{"day": "2026-01-02", "avg": 60.0, "max": 120.0}],
                    "pm10": [{"day": "2026-01-02", "avg": 85.0}],
                    "o3": [{"day": "2026-01-02", "avg": 33.0}],
                }
            },
        },
    }


def test_parse_aqicn_payload_returns_dataframe():
    df = parse_aqicn_payload(_fake_aqicn_payload())

    assert isinstance(df, pd.DataFrame)
    assert "us_aqi" in df.columns
    assert "pm2_5" in df.columns
    assert "record_type" in df.columns
    assert "pm2_5_max" not in df.columns
    # 1 realtime row + 1 forecast row
    assert len(df) == 2


def test_parse_aqicn_payload_realtime_row_values():
    df = parse_aqicn_payload(_fake_aqicn_payload())
    realtime_row = df[df["record_type"] == "realtime"].iloc[0]

    assert realtime_row["us_aqi"] == 145.0
    assert realtime_row["pm2_5"] == 55.0
    assert realtime_row["temperature"] == 20.0


def test_parse_aqicn_payload_forecast_row_values():
    df = parse_aqicn_payload(_fake_aqicn_payload())
    forecast_row = df[df["record_type"] == "forecast"].iloc[0]

    assert forecast_row["pm2_5"] == 60.0
    assert forecast_row["pm10"] == 85.0
    assert forecast_row["ozone"] == 33.0


def test_parse_aqicn_payload_raises_on_error_status():
    bad_payload = {"status": "error", "data": "Invalid key"}
    with pytest.raises(ValueError):
        parse_aqicn_payload(bad_payload)


@patch("src.data.api_client.AQICN_API_KEY", "fake_key")
@patch("src.data.api_client.requests.get")
def test_fetch_raw_air_quality_calls_api_and_parses(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = _fake_aqicn_payload()
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    df = fetch_raw_air_quality(lat=33.6844, lon=73.0479)

    assert not df.empty
    assert mock_get.called


@patch("src.data.api_client.AQICN_API_KEY", None)
def test_fetch_raw_air_quality_raises_without_api_key():
    with pytest.raises(ValueError):
        fetch_raw_air_quality(lat=33.6844, lon=73.0479)