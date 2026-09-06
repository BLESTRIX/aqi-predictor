import pandas as pd
import numpy as np
from src.config import CONFIG


def enforce_daily_cadence(df: pd.DataFrame) -> pd.DataFrame:
    """Resamples the DataFrame to an exact daily grid and interpolates missing days."""
    df = df.sort_values("time").set_index("time")

    resampled_df = df.resample("1D").mean(numeric_only=True)
    resampled_df = resampled_df.interpolate(method="linear")

    return resampled_df.reset_index()


def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily-cadence target steps (1 row = 1 day):
    shift(-1) = 24 hours ahead
    shift(-2) = 48 hours ahead
    shift(-3) = 72 hours ahead
    """
    df["target_aqi_24h"] = df["us_aqi"].shift(-1)
    df["target_aqi_48h"] = df["us_aqi"].shift(-2)
    df["target_aqi_72h"] = df["us_aqi"].shift(-3)
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts calendar features and cyclical encodings."""
    df["day_of_week"] = df["time"].dt.dayofweek
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month

    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    return df


def create_daily_lags_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Computes daily lag metrics and rolling statistics."""
    df["aqi_lag_1d"] = df["us_aqi"].shift(1)
    df["aqi_change_rate_1d"] = (df["us_aqi"] - df["aqi_lag_1d"]) / (df["aqi_lag_1d"] + 1e-5)

    df["aqi_rolling_mean_3d"] = df["us_aqi"].rolling(window=3, min_periods=1).mean()
    df["aqi_rolling_mean_7d"] = df["us_aqi"].rolling(window=7, min_periods=1).mean()
    df["aqi_rolling_max_7d"] = df["us_aqi"].rolling(window=7, min_periods=1).max()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main feature engineering entrypoint (daily cadence)."""
    # 1. Standardize to a daily grid first
    df = enforce_daily_cadence(df)

    # 2. Compute features and multi-step target shifts
    df = create_time_features(df)
    df = create_daily_lags_and_rolling(df)
    df = create_target_variables(df)

    # Hopsworks metadata and primary keys
    df["location"] = CONFIG["location"]["name"].lower()
    df["event_timestamp"] = df["time"].astype("int64") // 10**6

    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    return df.dropna(subset=feature_cols).reset_index(drop=True)


if __name__ == "__main__":
    from src.data.backfill import load_historical_aqicn_data

    raw_df = load_historical_aqicn_data("data/history_ISL_AQI.csv")
    processed_df = engineer_features(raw_df)

    print(f"Engineered {processed_df.shape[1]} features across {len(processed_df)} daily records for {CONFIG['location']['name']}:")
    print(processed_df[["time", "us_aqi", "aqi_lag_1d", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]].head(10))