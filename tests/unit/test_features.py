import pytest
import pandas as pd
import numpy as np
from src.features.build_features import (
    enforce_daily_cadence,
    create_time_features,
    create_daily_lags_and_rolling,
    create_target_variables,
    engineer_features,
)


def _sample_daily_df(n_days=60, start="2026-01-01"):
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    return pd.DataFrame({
        "time": dates,
        "us_aqi": np.linspace(50, 150, n_days),
        "pm2_5": np.linspace(10, 60, n_days),
    })


def test_enforce_daily_cadence_resamples_to_daily_grid():
    df = _sample_daily_df(n_days=10)
    result = enforce_daily_cadence(df)

    assert "time" in result.columns
    assert len(result) == 10
    diffs = result["time"].diff().dropna()
    assert all(diffs == pd.Timedelta(days=1))


def test_create_time_features_adds_expected_columns():
    df = _sample_daily_df(n_days=24)
    df = enforce_daily_cadence(df)
    result = create_time_features(df)

    for col in ["day_of_week", "day", "month", "month_sin", "month_cos"]:
        assert col in result.columns
    assert result["month_sin"].between(-1, 1).all()
    assert result["month_cos"].between(-1, 1).all()


def test_create_daily_lags_and_rolling_adds_lag_and_rolling_columns():
    df = _sample_daily_df(n_days=20)
    df = enforce_daily_cadence(df)
    result = create_daily_lags_and_rolling(df)

    assert "aqi_lag_1d" in result.columns
    assert "aqi_change_rate_1d" in result.columns
    assert "aqi_rolling_mean_3d" in result.columns
    assert "aqi_rolling_mean_7d" in result.columns
    assert "aqi_rolling_max_7d" in result.columns
    # first row's lag should be NaN (no prior value)
    assert pd.isna(result["aqi_lag_1d"].iloc[0])


def test_create_target_variables_shifts_correctly():
    df = _sample_daily_df(n_days=10)
    df = enforce_daily_cadence(df)
    result = create_target_variables(df)

    assert "target_aqi_24h" in result.columns
    assert "target_aqi_48h" in result.columns
    assert "target_aqi_72h" in result.columns

    # target_aqi_24h at row i should equal us_aqi at row i+1 (1 day ahead)
    non_null = result.dropna(subset=["target_aqi_24h"])
    idx = non_null.index[0]
    assert result.loc[idx, "target_aqi_24h"] == pytest.approx(result.loc[idx + 1, "us_aqi"])

    # target_aqi_72h at row i should equal us_aqi at row i+3 (3 days ahead)
    non_null_72 = result.dropna(subset=["target_aqi_72h"])
    idx72 = non_null_72.index[0]
    assert result.loc[idx72, "target_aqi_72h"] == pytest.approx(result.loc[idx72 + 3, "us_aqi"])


def test_engineer_features_end_to_end_drops_nan_feature_rows():
    df = _sample_daily_df(n_days=30)  # enough rows to survive dropna after shifting
    result = engineer_features(df)

    assert isinstance(result, pd.DataFrame)
    assert "location" in result.columns
    assert "event_timestamp" in result.columns
    feature_cols = [c for c in result.columns if not c.startswith("target_")]
    assert not result[feature_cols].isna().any().any()