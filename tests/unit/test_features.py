import pytest
import pandas as pd
import numpy as np
from src.features.build_features import (
    compute_time_features,
    compute_lag_features,
    compute_rolling_features,
    calculate_aqi_subindex,
    engineer_all_features
)


def test_calculate_aqi_subindex():
    assert calculate_aqi_subindex(0.0) == 0.0
    assert calculate_aqi_subindex(12.0) == 50.0
    assert calculate_aqi_subindex(35.4) == 100.0
    assert calculate_aqi_subindex(55.4) == 150.0


def test_compute_time_features():
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2026-01-01 00:00:00", periods=5, freq="h")
    })
    df_res = compute_time_features(df)
    assert "hour" in df_res.columns
    assert "hour_sin" in df_res.columns
    assert "hour_cos" in df_res.columns
    assert "dayofweek" in df_res.columns
    assert "month" in df_res.columns
    assert len(df_res) == 5


def test_compute_lag_features():
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2026-01-01 00:00:00", periods=10, freq="h"),
        "pm2_5": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    })
    df_res = compute_lag_features(df, target_cols=["pm2_5"], lags=[1, 2])
    assert "pm2_5_lag_1h" in df_res.columns
    assert "pm2_5_lag_2h" in df_res.columns
    assert pd.isna(df_res.iloc[0]["pm2_5_lag_1h"])
    assert df_res.iloc[1]["pm2_5_lag_1h"] == 10.0


def test_engineer_all_features():
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2026-01-01 00:00:00", periods=5, freq="h"),
        "pm2_5": [10.0, 15.0, 20.0, 25.0, 30.0],
        "pm10": [20.0, 30.0, 40.0, 50.0, 60.0]
    })
    df_res = engineer_all_features(df)
    assert "calculated_aqi" in df_res.columns
    assert "hour_sin" in df_res.columns
    assert "pm2_5_lag_1h" in df_res.columns
